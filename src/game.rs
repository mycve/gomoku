use serde::{Deserialize, Serialize};
use std::fmt;

pub const BOARD_SIZE: usize = 15;
pub const CELL_COUNT: usize = BOARD_SIZE * BOARD_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Player {
    Black,
    White,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
    pub fn stone(self) -> i8 {
        match self {
            Self::Black => 1,
            Self::White => -1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Move(pub usize);

impl Move {
    pub fn new(row: usize, col: usize) -> Option<Self> {
        (row < BOARD_SIZE && col < BOARD_SIZE).then_some(Self(row * BOARD_SIZE + col))
    }
    pub fn row(self) -> usize {
        self.0 / BOARD_SIZE
    }
    pub fn col(self) -> usize {
        self.0 % BOARD_SIZE
    }
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        let mut chars = s.chars();
        let col = chars.next()? as usize - 'a' as usize;
        let row = chars.as_str().parse::<usize>().ok()?.checked_sub(1)?;
        Self::new(row, col)
    }
    pub fn notation(self) -> String {
        format!("{}{}", (b'a' + self.col() as u8) as char, self.row() + 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Win(Player),
    Draw,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Board {
    cells: Vec<i8>,
    to_move: Player,
    moves: usize,
    last: Option<Move>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: vec![0; CELL_COUNT],
            to_move: Player::Black,
            moves: 0,
            last: None,
        }
    }
    pub fn cells(&self) -> &[i8] {
        &self.cells
    }
    pub fn to_move(&self) -> Player {
        self.to_move
    }
    pub fn move_count(&self) -> usize {
        self.moves
    }
    pub fn is_legal(&self, mv: Move) -> bool {
        mv.0 < CELL_COUNT && self.cells[mv.0] == 0 && self.outcome().is_none()
    }
    pub fn play(&mut self, mv: Move) -> bool {
        if !self.is_legal(mv) {
            return false;
        }
        self.cells[mv.0] = self.to_move.stone();
        self.moves += 1;
        self.last = Some(mv);
        self.to_move = self.to_move.other();
        true
    }
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.outcome().is_some() {
            return vec![];
        }
        if self.moves == 0 {
            return vec![Move::new(BOARD_SIZE / 2, BOARD_SIZE / 2).unwrap()];
        }
        let mut near = [false; CELL_COUNT];
        for i in 0..CELL_COUNT {
            if self.cells[i] == 0 {
                continue;
            }
            let r = i / BOARD_SIZE;
            let c = i % BOARD_SIZE;
            for dr in -2i32..=2 {
                for dc in -2i32..=2 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nc >= 0 && nr < BOARD_SIZE as i32 && nc < BOARD_SIZE as i32 {
                        let j = nr as usize * BOARD_SIZE + nc as usize;
                        if self.cells[j] == 0 {
                            near[j] = true;
                        }
                    }
                }
            }
        }
        near.iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(Move(i)))
            .collect()
    }
    pub fn outcome(&self) -> Option<Outcome> {
        if let Some(mv) = self.last {
            let stone = self.cells[mv.0];
            for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                let mut n = 1;
                for sign in [-1, 1] {
                    let mut r = mv.row() as i32 + dr * sign;
                    let mut c = mv.col() as i32 + dc * sign;
                    while r >= 0
                        && c >= 0
                        && r < BOARD_SIZE as i32
                        && c < BOARD_SIZE as i32
                        && self.cells[r as usize * BOARD_SIZE + c as usize] == stone
                    {
                        n += 1;
                        r += dr * sign;
                        c += dc * sign;
                    }
                }
                if n >= 5 {
                    return Some(Outcome::Win(if stone == 1 {
                        Player::Black
                    } else {
                        Player::White
                    }));
                }
            }
        }
        (self.moves == CELL_COUNT).then_some(Outcome::Draw)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for c in 0..BOARD_SIZE {
            write!(f, " {}", (b'A' + c as u8) as char)?;
        }
        writeln!(f)?;
        for r in 0..BOARD_SIZE {
            write!(f, "{:>2} ", r + 1)?;
            for c in 0..BOARD_SIZE {
                let x = self.cells[r * BOARD_SIZE + c];
                write!(
                    f,
                    " {}",
                    match x {
                        1 => '●',
                        -1 => '○',
                        _ => '·',
                    }
                )?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn horizontal_win() {
        let mut b = Board::new();
        for c in 0..5 {
            assert!(b.play(Move::new(7, c).unwrap()));
            if c < 4 {
                assert!(b.play(Move::new(8, c).unwrap()));
            }
        }
        assert_eq!(b.outcome(), Some(Outcome::Win(Player::Black)));
    }
    #[test]
    fn notation_roundtrip() {
        for s in ["a1", "h8", "o15"] {
            let m = Move::parse(s).unwrap();
            assert_eq!(m.notation(), s);
        }
    }
}
