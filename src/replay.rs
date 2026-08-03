use crate::game::{Board, Move, transform_index};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub board: Board,
    pub policy: Vec<(Move, f32)>,
    pub value: f32,
    pub value_wdl: Option<[f32; 3]>,
    pub generation: u64,
    pub policy_weight: f32,
    pub value_weight: f32,
    pub policy_surprise: f32,
    pub value_surprise: f32,
    pub predicted_value: f32,
}

impl Sample {
    pub(crate) fn transformed(&self, symmetry: usize) -> Self {
        Self {
            board: self.board.transformed(symmetry),
            policy: self
                .policy
                .iter()
                .map(|&(mv, probability)| (Move(transform_index(mv.0, symmetry)), probability))
                .collect(),
            value: self.value,
            value_wdl: self.value_wdl,
            generation: self.generation,
            policy_weight: self.policy_weight,
            value_weight: self.value_weight,
            policy_surprise: self.policy_surprise,
            value_surprise: self.value_surprise,
            predicted_value: self.predicted_value,
        }
    }
}

pub struct MixedSampleBatch {
    pub samples: Vec<Sample>,
    pub recent_quota: usize,
    pub actual_recent: usize,
}

pub fn sample_mixed_recent(
    pool: &[Sample],
    count: usize,
    recent_fraction: f32,
    recent_updates: u64,
    policy_surprise_fraction: f32,
    value_surprise_fraction: f32,
    seed: u64,
) -> MixedSampleBatch {
    if pool.is_empty() || count == 0 {
        return MixedSampleBatch {
            samples: Vec::new(),
            recent_quota: 0,
            actual_recent: 0,
        };
    }
    let newest = pool
        .iter()
        .map(|sample| sample.generation)
        .max()
        .unwrap_or(0);
    let oldest_recent = newest.saturating_sub(recent_updates.max(1).saturating_sub(1));
    let recent = pool
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| (sample.generation >= oldest_recent).then_some(index))
        .collect::<Vec<_>>();
    let historical = pool
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| (sample.generation < oldest_recent).then_some(index))
        .collect::<Vec<_>>();
    let recent_quota = if recent.is_empty() {
        0
    } else {
        ((count as f32) * recent_fraction.clamp(0.0, 1.0)).round() as usize
    }
    .min(count);
    let mut rng = SplitMix64(seed);
    let mut samples = Vec::with_capacity(count);
    let policy_quota = ((count as f32) * policy_surprise_fraction.clamp(0.0, 1.0)).round() as usize;
    let value_quota = ((count as f32) * value_surprise_fraction.clamp(0.0, 1.0)).round() as usize;
    let mut kinds = vec![0u8; count];
    for kind in &mut kinds[..policy_quota.min(count)] {
        *kind = 1;
    }
    for kind in
        &mut kinds[policy_quota.min(count)..policy_quota.saturating_add(value_quota).min(count)]
    {
        *kind = 2;
    }
    for index in (1..kinds.len()).rev() {
        let other = rng.index(index + 1);
        kinds.swap(index, other);
    }
    for slot in 0..count {
        let source = if slot < recent_quota {
            &recent
        } else if historical.is_empty() {
            &recent
        } else {
            &historical
        };
        let kind = kinds[slot] as usize;
        let selected = weighted_index(pool, source, kind, &mut rng);
        let sample = &pool[selected];
        samples.push(sample.transformed(rng.index(8)));
    }
    for index in (1..samples.len()).rev() {
        let other = rng.index(index + 1);
        samples.swap(index, other);
    }
    let actual_recent = samples
        .iter()
        .filter(|sample| sample.generation >= oldest_recent)
        .count();
    MixedSampleBatch {
        samples,
        recent_quota,
        actual_recent,
    }
}

fn weighted_index(pool: &[Sample], source: &[usize], kind: usize, rng: &mut SplitMix64) -> usize {
    if kind == 0 {
        return source[rng.index(source.len())];
    }
    let total = source
        .iter()
        .map(|&index| {
            if kind == 1 {
                pool[index].policy_surprise
            } else {
                pool[index].value_surprise
            }
            .max(1e-4)
        })
        .sum::<f32>();
    let mut draw = rng.unit() * total;
    for &index in source {
        draw -= if kind == 1 {
            pool[index].policy_surprise
        } else {
            pool[index].value_surprise
        }
        .max(1e-4);
        if draw <= 0.0 {
            return index;
        }
    }
    *source.last().expect("抽样来源不能为空")
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len.max(1)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}
pub fn load(path: impl AsRef<Path>) -> io::Result<Vec<Sample>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e),
    };
    BufReader::new(FrameDecoder::new(file))
        .lines()
        .map(|l| serde_json::from_str(&l?).map_err(io::Error::other))
        .collect()
}

pub fn save(path: impl AsRef<Path>, samples: &[Sample]) -> io::Result<()> {
    let path = path.as_ref();
    if samples.is_empty() {
        if path.exists() {
            fs::remove_file(path)?;
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let file = File::create(&temporary)?;
    let mut encoder = FrameEncoder::new(file);
    for sample in samples {
        serde_json::to_writer(&mut encoder, sample).map_err(io::Error::other)?;
        writeln!(encoder)?;
    }
    encoder.finish().map_err(io::Error::other)?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(generation: u64) -> Sample {
        Sample {
            board: Board::new(),
            policy: Vec::new(),
            value: 0.0,
            value_wdl: None,
            generation,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        }
    }

    #[test]
    fn mixed_sampling_reserves_recent_quota() {
        let mut pool = vec![sample(1); 100];
        pool.extend(vec![sample(10); 10]);
        let batch = sample_mixed_recent(&pool, 1000, 0.4, 2, 0.4, 0.1, 7);
        assert_eq!(batch.samples.len(), 1000);
        assert_eq!(batch.recent_quota, 400);
        assert_eq!(batch.actual_recent, 400);
        let generations = |samples: &[Sample]| {
            samples
                .iter()
                .map(|sample| sample.generation)
                .collect::<Vec<_>>()
        };
        let again = sample_mixed_recent(&pool, 1000, 0.4, 2, 0.4, 0.1, 7);
        assert_eq!(generations(&batch.samples), generations(&again.samples));
    }

    #[test]
    fn mixed_sampling_falls_back_when_there_is_no_history() {
        let pool = vec![sample(10); 10];
        let batch = sample_mixed_recent(&pool, 100, 0.4, 5, 0.4, 0.1, 7);
        assert_eq!(batch.samples.len(), 100);
        assert_eq!(batch.recent_quota, 40);
        assert_eq!(batch.actual_recent, 100);
    }

    #[test]
    fn surprise_sampling_prefers_informative_samples() {
        let mut pool = vec![sample(10); 20];
        pool[0].policy_surprise = 1000.0;
        let batch = sample_mixed_recent(&pool, 1000, 0.0, 1, 1.0, 0.0, 19);
        let selected_high_surprise = batch
            .samples
            .iter()
            .filter(|sample| sample.policy_surprise > 100.0)
            .count();
        assert!(selected_high_surprise > 900);
    }

    #[test]
    fn lz4_snapshot_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "gomoku-replay-{}-{}.lz4",
            std::process::id(),
            20260730
        ));
        let samples = vec![sample(3), sample(4)];
        save(&path, &samples).unwrap();
        let restored = load(&path).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].generation, 3);
        assert_eq!(restored[1].generation, 4);
        fs::remove_file(path).unwrap();
    }
}
