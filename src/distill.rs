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
        for mv in allowed {
            let weight = policy[policy_base + mv.0].max(0) as f32;
            if weight > 0.0 {
                target.push((mv, weight));
            }
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
        });
    }
    Ok(samples)
}

fn required(npz: &NpzTensors, name: &str) -> io::Result<Tensor> {
    npz.get(name)
        .map_err(err)?
        .ok_or_else(|| io::Error::other(format!("NPZ 缺少 {name}")))
}
fn err(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
