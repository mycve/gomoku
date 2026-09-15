use crate::game::{Board, Move, transform_index};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub board: Board,
    pub policy: Vec<(Move, f32)>,
    /// 当前行棋方的真实终局结果；v32 必填，旧 TD 回放不可直接复用。
    #[serde(rename = "mc_value")]
    pub value: f32,
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
    crate::scope_profile!("replay.sample_total");
    let (oldest_recent, recent, historical) = {
        crate::scope_profile!("replay.partition");
        let newest = pool
            .iter()
            .map(|sample| sample.generation)
            .max()
            .unwrap_or(0);
        let oldest_recent = newest.saturating_sub(recent_updates.max(1).saturating_sub(1));
        let (recent, historical): (Vec<_>, Vec<_>) =
            (0..pool.len()).partition(|&index| pool[index].generation >= oldest_recent);
        (oldest_recent, recent, historical)
    };
    let recent_quota = if recent.is_empty() {
        0
    } else {
        ((count as f32) * recent_fraction.clamp(0.0, 1.0)).round() as usize
    }
    .min(count);
    let mut rng = SplitMix64(seed);
    let (recent_weights, historical_weights) = {
        crate::scope_profile!("replay.weights");
        (
            WeightedSource::new(pool, &recent),
            WeightedSource::new(pool, &historical),
        )
    };
    let mut plan = {
        crate::scope_profile!("replay.select");
        let mut plan = Vec::with_capacity(count);
        let policy_quota =
            ((count as f32) * policy_surprise_fraction.clamp(0.0, 1.0)).round() as usize;
        let value_quota =
            ((count as f32) * value_surprise_fraction.clamp(0.0, 1.0)).round() as usize;
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
            let weights = if slot < recent_quota || historical.is_empty() {
                &recent_weights
            } else {
                &historical_weights
            };
            let selected = weights.sample(kinds[slot] as usize, &mut rng);
            plan.push((selected, rng.index(8)));
        }
        plan
    };
    {
        crate::scope_profile!("replay.shuffle");
        for index in (1..plan.len()).rev() {
            let other = rng.index(index + 1);
            plan.swap(index, other);
        }
    }
    let actual_recent = plan
        .iter()
        .filter(|&&(index, _)| pool[index].generation >= oldest_recent)
        .count();
    let samples = {
        crate::scope_profile!("replay.materialize");
        plan.into_iter()
            .map(|(index, symmetry)| pool[index].transformed(symmetry))
            .collect()
    };
    MixedSampleBatch {
        samples,
        recent_quota,
        actual_recent,
    }
}

struct WeightedSource<'a> {
    indices: &'a [usize],
    policy_cdf: Vec<f32>,
    policy_total: f32,
    value_cdf: Vec<f32>,
    value_total: f32,
}

impl<'a> WeightedSource<'a> {
    fn new(pool: &[Sample], indices: &'a [usize]) -> Self {
        let mut policy_cdf = Vec::with_capacity(indices.len());
        let mut value_cdf = Vec::with_capacity(indices.len());
        let mut policy_total = 0.0;
        let mut value_total = 0.0;
        for &index in indices {
            policy_total += pool[index].policy_surprise.max(1e-4);
            value_total += pool[index].value_surprise.max(1e-4);
            policy_cdf.push(policy_total);
            value_cdf.push(value_total);
        }
        Self {
            indices,
            policy_cdf,
            policy_total,
            value_cdf,
            value_total,
        }
    }

    fn sample(&self, kind: usize, rng: &mut SplitMix64) -> usize {
        if kind == 0 {
            return self.indices[rng.index(self.indices.len())];
        }
        let (cdf, total) = if kind == 1 {
            (&self.policy_cdf, self.policy_total)
        } else {
            (&self.value_cdf, self.value_total)
        };
        let draw = rng.unit() * total;
        let position = cdf.partition_point(|&cumulative| cumulative <= draw);
        self.indices[position.min(self.indices.len() - 1)]
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
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}
const SNAPSHOT_MAGIC: &[u8; 8] = b"GO19RP01";

#[derive(Serialize, Deserialize)]
struct StoredSample {
    board: crate::game::SnapshotBoard,
    policy: Vec<(Move, f32)>,
    generation: u64,
    stats: [f32; 6],
}
fn snapshot_file(path: &Path) -> io::Result<Option<File>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    if &magic != SNAPSHOT_MAGIC {
        return Err(io::Error::other(format!(
            "回放 {} 不是 GO19RP01 二进制快照；旧回放不兼容，请保留旧文件并改用新的 replay_path。模型无需重练。",
            path.display()
        )));
    }
    Ok(Some(file))
}
pub fn check_format(path: impl AsRef<Path>) -> io::Result<()> {
    snapshot_file(path.as_ref()).map(|_| ())
}
pub fn load(path: impl AsRef<Path>) -> io::Result<Vec<Sample>> {
    let Some(file) = snapshot_file(path.as_ref())? else {
        return Ok(Vec::new());
    };
    let mut reader = BufReader::new(FrameDecoder::new(file));
    let count: u64 = bincode::deserialize_from(&mut reader).map_err(io::Error::other)?;
    if count > 10_000_000 {
        return Err(io::Error::other("回放样本数量无效"));
    }
    let mut history = crate::game::SnapshotDecoder::new();
    let mut samples = Vec::with_capacity((count as usize).min(500_000));
    for _ in 0..count {
        let nodes = bincode::deserialize_from(&mut reader).map_err(io::Error::other)?;
        let saved: StoredSample =
            bincode::deserialize_from(&mut reader).map_err(io::Error::other)?;
        let [
            value,
            policy_weight,
            value_weight,
            policy_surprise,
            value_surprise,
            predicted_value,
        ] = saved.stats;
        samples.push(Sample {
            board: history.decode(nodes, saved.board)?,
            policy: saved.policy,
            generation: saved.generation,
            value,
            policy_weight,
            value_weight,
            policy_surprise,
            value_surprise,
            predicted_value,
        });
    }
    let mut trailing = [0];
    if reader.read(&mut trailing)? != 0 {
        return Err(io::Error::other("回放包含多余数据"));
    }
    Ok(samples)
}

pub fn save(path: impl AsRef<Path>, samples: &[Sample]) -> io::Result<()> {
    save_controlled(path, samples, None).map(|_| ())
}
/// 先写临时快照，完整落盘后原子替换。取消不破坏已有文件。
pub fn save_controlled(
    path: impl AsRef<Path>,
    samples: &[Sample],
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<bool> {
    use std::sync::atomic::Ordering;
    let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
    if cancelled() {
        return Ok(false);
    }
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".go19-replay-")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    let started = std::time::Instant::now();
    let mut reported = started;
    let mut history = crate::game::SnapshotEncoder::default();
    eprintln!(
        "replay   : saving {} samples to {} (second Ctrl+C skips replay)",
        samples.len(),
        path.display()
    );
    {
        temporary.as_file_mut().write_all(SNAPSHOT_MAGIC)?;
        let writer = BufWriter::with_capacity(1024 * 1024, temporary.as_file_mut());
        let mut encoder = FrameEncoder::new(writer);
        bincode::serialize_into(&mut encoder, &(samples.len() as u64)).map_err(io::Error::other)?;
        for (index, sample) in samples.iter().enumerate() {
            if cancelled() {
                eprintln!("replay   : cancelled; previous snapshot preserved");
                return Ok(false);
            }
            let (nodes, board) = history.encode(&sample.board);
            let stored = StoredSample {
                board,
                policy: sample.policy.clone(),
                generation: sample.generation,
                stats: [
                    sample.value,
                    sample.policy_weight,
                    sample.value_weight,
                    sample.policy_surprise,
                    sample.value_surprise,
                    sample.predicted_value,
                ],
            };
            bincode::serialize_into(&mut encoder, &nodes).map_err(io::Error::other)?;
            bincode::serialize_into(&mut encoder, &stored).map_err(io::Error::other)?;
            if reported.elapsed().as_secs() >= 1 {
                eprintln!(
                    "replay   : saved {}/{} history_nodes={} elapsed={:.1}s",
                    index + 1,
                    samples.len(),
                    history.history_nodes(),
                    started.elapsed().as_secs_f32()
                );
                reported = std::time::Instant::now();
            }
        }
        encoder.finish().map_err(io::Error::other)?.flush()?;
    }
    if cancelled() {
        return Ok(false);
    }
    temporary.as_file().sync_all()?;
    if cancelled() {
        return Ok(false);
    }
    let bytes = temporary.as_file().metadata()?.len();
    temporary.persist(path).map_err(|error| error.error)?;
    eprintln!(
        "replay   : complete samples={} history_nodes={} bytes={} elapsed={:.2}s",
        samples.len(),
        history.history_nodes(),
        bytes,
        started.elapsed().as_secs_f32()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(generation: u64) -> Sample {
        Sample {
            board: Board::new(),
            policy: Vec::new(),
            value: 0.0,
            generation,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        }
    }

    #[test]
    fn old_td_replay_is_rejected() {
        let sample = Sample {
            board: Board::new(),
            policy: vec![],
            value: 1.0,
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        };
        let mut json = serde_json::to_value(&sample).unwrap();
        assert_eq!(json["mc_value"], 1.0);
        assert!(serde_json::from_value::<Sample>(json.clone()).is_ok());
        json.as_object_mut().unwrap().remove("mc_value");
        json["value"] = serde_json::json!(0.2);
        assert!(serde_json::from_value::<Sample>(json).is_err());
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

    #[test]
    fn compact_snapshot_preserves_shared_histories_symmetries_and_all_labels() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replay.bin.lz4");
        let mut board = Board::new();
        let mut samples = Vec::new();
        let mut encoder = crate::game::SnapshotEncoder::default();
        for ply in 0..120 {
            let moves = board.rule_legal_moves();
            let mv = moves[(ply * 17) % (moves.len() - 1)];
            assert!(board.play(mv));
            let mut s = sample(ply as u64);
            s.board = board.clone();
            s.policy = vec![(Move::PASS, 0.5), (Move(0), 0.5)];
            s.value = -1.0;
            s.policy_surprise = 0.25;
            s.predicted_value = 0.125;
            for symmetry in 0..8 {
                let transformed = s.transformed(symmetry);
                encoder.encode(&transformed.board);
                samples.push(transformed);
            }
        }
        assert_eq!(encoder.history_nodes(), 121);
        save(&path, &samples).unwrap();
        let restored = load(&path).unwrap();
        assert_eq!(restored.len(), samples.len());
        for (before, after) in samples.iter().zip(&restored) {
            assert_eq!(
                serde_json::to_vec(before).unwrap(),
                serde_json::to_vec(after).unwrap()
            );
            assert_eq!(
                before.board.rule_legal_moves(),
                after.board.rule_legal_moves()
            );
        }
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        assert!(load(&path).is_err());
    }

    #[test]
    fn cancelled_snapshot_keeps_previous_file_and_legacy_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replay.bin.lz4");
        save(&path, &[sample(7)]).unwrap();
        let previous = fs::read(&path).unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert!(!save_controlled(&path, &[sample(9)], Some(&cancel)).unwrap());
        assert_eq!(fs::read(&path).unwrap(), previous);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
        // 等临时文件出现后取消，验证写入过程中也不会替换上一份完整快照。
        use std::sync::atomic::{AtomicBool, Ordering};
        let cancel = AtomicBool::new(false);
        let finished = AtomicBool::new(false);
        let batch = vec![sample(9); 100_000];
        let result = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !finished.load(Ordering::Acquire) {
                    if fs::read_dir(directory.path()).unwrap().any(|entry| {
                        entry
                            .unwrap()
                            .path()
                            .extension()
                            .is_some_and(|ext| ext == "tmp")
                    }) {
                        cancel.store(true, Ordering::Release);
                        break;
                    }
                    std::thread::yield_now();
                }
            });
            let result = save_controlled(&path, &batch, Some(&cancel));
            finished.store(true, Ordering::Release);
            result
        });
        assert!(!result.unwrap());
        assert_eq!(fs::read(&path).unwrap(), previous);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
        fs::write(&path, b"old jsonl snapshot").unwrap();
        assert!(
            check_format(&path)
                .unwrap_err()
                .to_string()
                .contains("旧回放不兼容")
        );
    }
}
