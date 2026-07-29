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
