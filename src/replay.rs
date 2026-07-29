use crate::game::{Board, Move};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::Path,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub board: Board,
    pub policy: Vec<(Move, f32)>,
    pub value: f32,
    #[serde(default)]
    pub moves_left: f32,
    #[serde(default)]
    pub generation: u64,
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
    let recent_quota = if recent.is_empty() {
        0
    } else {
        ((count as f32) * recent_fraction.clamp(0.0, 1.0)).round() as usize
    }
    .min(count);
    let mut rng = SplitMix64(seed);
    let mut samples = Vec::with_capacity(count);
    for _ in 0..recent_quota {
        samples.push(pool[recent[rng.index(recent.len())]].clone());
    }
    for _ in recent_quota..count {
        samples.push(pool[rng.index(pool.len())].clone());
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
}
pub fn append(path: impl AsRef<Path>, samples: &[Sample]) -> io::Result<()> {
    if let Some(p) = path.as_ref().parent() {
        fs::create_dir_all(p)?
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    for s in samples {
        serde_json::to_writer(&mut f, s).map_err(io::Error::other)?;
        writeln!(f)?
    }
    Ok(())
}
pub fn load(path: impl AsRef<Path>) -> io::Result<Vec<Sample>> {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e),
    };
    BufReader::new(f)
        .lines()
        .map(|l| serde_json::from_str(&l?).map_err(io::Error::other))
        .collect()
}

pub fn save(path: impl AsRef<Path>, samples: &[Sample]) -> io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    for sample in samples {
        serde_json::to_writer(&mut file, sample).map_err(io::Error::other)?;
        writeln!(file)?;
    }
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
            moves_left: 1.0,
            generation,
        }
    }

    #[test]
    fn mixed_sampling_reserves_recent_quota() {
        let mut pool = vec![sample(1); 100];
        pool.extend(vec![sample(10); 10]);
        let batch = sample_mixed_recent(&pool, 1000, 0.4, 2, 7);
        assert_eq!(batch.samples.len(), 1000);
        assert_eq!(batch.recent_quota, 400);
        assert!(batch.actual_recent >= 400);
        let generations = |samples: &[Sample]| {
            samples
                .iter()
                .map(|sample| sample.generation)
                .collect::<Vec<_>>()
        };
        let again = sample_mixed_recent(&pool, 1000, 0.4, 2, 7);
        assert_eq!(generations(&batch.samples), generations(&again.samples));
    }
}
