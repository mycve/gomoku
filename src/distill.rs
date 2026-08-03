//! KataGo Gomoku NPZ 蒸馏数据读取。

use crate::{
    game::{Board, CELL_COUNT, Move, Player},
    replay::Sample,
};
use candle_core::{Device, Tensor, npy::NpzTensors};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

const PACKED_BYTES: usize = (CELL_COUNT + 7) / 8;

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    pub rows: usize,
    pub accepted: usize,
    pub policy_mass_total: f64,
    pub policy_mass_kept: f64,
    pub top1_kept: usize,
}

impl LoadStats {
    pub fn policy_mass_retention(self) -> f64 {
        self.policy_mass_kept / self.policy_mass_total.max(f64::EPSILON)
    }
    pub fn top1_retention(self) -> f64 {
        self.top1_kept as f64 / self.accepted.max(1) as f64
    }
}

pub fn npz_files(path: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|x| x == "npz"))
        .collect::<Vec<_>>();
    files.sort_by_key(|path| {
        path.file_stem()
            .and_then(|x| x.to_str())
            .and_then(|x| x.strip_prefix("data"))
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    });
    Ok(files)
}

pub fn is_lfs_pointer(path: impl AsRef<Path>) -> io::Result<bool> {
    let data = fs::read(path)?;
    Ok(data.len() < 1024 && data.starts_with(b"version https://git-lfs.github.com/spec/v1"))
}

pub fn load_npz(path: impl AsRef<Path>, limit: Option<usize>) -> io::Result<Vec<Sample>> {
    load_npz_with_stats(path, limit).map(|x| x.0)
}

pub fn load_npz_with_stats(
    path: impl AsRef<Path>,
    limit: Option<usize>,
) -> io::Result<(Vec<Sample>, LoadStats)> {
    let path = path.as_ref();
    if is_lfs_pointer(path)? {
        return Err(io::Error::other(format!(
            "{} 仍是 Git LFS 占位符，尚未下载数据实体",
            path.display()
        )));
    }
    let npz = NpzTensors::new(path).map_err(err)?;
    let binary = required(&npz, "binaryInputNCHWPacked")?
        .to_device(&Device::Cpu)
        .map_err(err)?;
    let policy = required(&npz, "policyTargetsNCMove")?
        .to_device(&Device::Cpu)
        .map_err(err)?;
    let values = required(&npz, "globalTargetsNC")?
        .to_device(&Device::Cpu)
        .map_err(err)?;
    let (n, channels, packed) = binary.dims3().map_err(err)?;
    if channels < 3 || packed < PACKED_BYTES {
        return Err(io::Error::other(format!(
            "binaryInput 形状不支持: {:?}",
            binary.dims()
        )));
    }
    let (pn, heads, moves) = policy.dims3().map_err(err)?;
    let (vn, value_channels) = values.dims2().map_err(err)?;
    if pn != n || vn != n || heads < 1 || moves < CELL_COUNT || value_channels < 3 {
        return Err(io::Error::other(
            "NPZ 的 policy/value 形状与 15x15 数据格式不匹配",
        ));
    }
    let binary = binary
        .flatten_all()
        .map_err(err)?
        .to_vec1::<u8>()
        .map_err(err)?;
    let policy = policy
        .flatten_all()
        .map_err(err)?
        .to_vec1::<i16>()
        .map_err(err)?;
    let values = values
        .flatten_all()
        .map_err(err)?
        .to_vec1::<f32>()
        .map_err(err)?;
    let take = limit.unwrap_or(n).min(n);
    let mut stats = LoadStats {
        rows: take,
        ..Default::default()
    };
    let mut samples = Vec::with_capacity(take);
    for row in 0..take {
        let bit = |channel: usize, sq: usize| {
            let byte = binary[(row * channels + channel) * packed + sq / 8];
            (byte >> (7 - sq % 8)) & 1 != 0
        };
        let own_count = (0..CELL_COUNT).filter(|&sq| bit(1, sq)).count();
        let opp_count = (0..CELL_COUNT).filter(|&sq| bit(2, sq)).count();
        let own = if own_count == opp_count {
            Player::Black
        } else {
            Player::White
        };
        let mut stones = Vec::with_capacity(own_count + opp_count);
        for sq in 0..CELL_COUNT {
            if bit(1, sq) {
                stones.push((Move(sq), own));
            } else if bit(2, sq) {
                stones.push((Move(sq), own.other()));
            }
        }
        let Some(board) = Board::from_stones(&stones) else {
            continue;
        };
        if board.to_move() != own || board.outcome().is_some() {
            continue;
        }
        let allowed = board.search_candidates();
        let mut target = Vec::with_capacity(allowed.len());
        let policy_base = (row * heads) * moves;
        let total_mass = (0..CELL_COUNT)
            .map(|sq| policy[policy_base + sq].max(0) as f64)
            .sum::<f64>();
        let top1 = (0..CELL_COUNT)
            .max_by_key(|&sq| policy[policy_base + sq])
            .unwrap();
        let mut kept_mass = 0.0;
        let mut keeps_top1 = false;
        for mv in allowed {
            let weight = policy[policy_base + mv.0].max(0) as f32;
            if weight > 0.0 {
                target.push((mv, weight));
                kept_mass += weight as f64;
            }
            keeps_top1 |= mv.0 == top1;
        }
        if target.is_empty() {
            continue;
        }
        let value_base = row * value_channels;
        let raw = [
            values[value_base],
            values[value_base + 2],
            values[value_base + 1],
        ];
        let sum: f32 = raw.iter().map(|x| x.max(0.0)).sum();
        if sum <= 1e-12 {
            continue;
        }
        let wdl = raw.map(|x| x.max(0.0) / sum);
        samples.push(Sample {
            board,
            policy: target,
            value: wdl[0] - wdl[2],
            value_wdl: Some(wdl),
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        });
        stats.accepted += 1;
        stats.policy_mass_total += total_mass;
        stats.policy_mass_kept += kept_mass;
        stats.top1_kept += usize::from(keeps_top1);
    }
    Ok((samples, stats))
}

pub fn augment_and_shuffle(samples: &mut [Sample], seed: u64) {
    let mut rng = SplitMix64(seed);
    for sample in samples.iter_mut() {
        *sample = sample.transformed(rng.index(8));
    }
    for index in (1..samples.len()).rev() {
        let other = rng.index(index + 1);
        samples.swap(index, other);
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

fn required(npz: &NpzTensors, name: &str) -> io::Result<Tensor> {
    npz.get(name)
        .map_err(err)?
        .ok_or_else(|| io::Error::other(format!("NPZ 缺少 {name}")))
}
fn err(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
