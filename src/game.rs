use serde::{Deserialize, Serialize};
use std::fmt;

pub const BOARD_SIZE: usize = 15;
pub const CELL_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
pub const TACTICAL_FEATURES: usize = 24;

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
    pub(crate) fn transformed(&self, symmetry: usize) -> Self {
        let mut cells = vec![0; CELL_COUNT];
        for (index, &stone) in self.cells.iter().enumerate() {
            cells[transform_index(index, symmetry)] = stone;
        }
        Self {
            cells,
            to_move: self.to_move,
            moves: self.moves,
            last: self.last.map(|mv| Move(transform_index(mv.0, symmetry))),
        }
    }
    pub fn is_legal(&self, mv: Move) -> bool {
        mv.0 < CELL_COUNT && self.cells[mv.0] == 0 && self.outcome().is_none()
    }
    pub fn play(&mut self, mv: Move) -> bool {
        crate::scope_profile!("game.play");
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
        crate::scope_profile!("game.legal_moves");
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
        crate::scope_profile!("game.outcome");
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

pub(crate) fn tactical_features(board: &Board, mv: Move) -> [f32; TACTICAL_FEATURES] {
    let mut features = [0.0; TACTICAL_FEATURES];
    if mv.0 >= CELL_COUNT || board.cells[mv.0] != 0 {
        return features;
    }
    let us = board.to_move().stone();
    let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let mut own_four = 0;
    let mut opponent_four = 0;
    let mut own_open_three = 0;
    let mut opponent_open_three = 0;
    for (direction, (dr, dc)) in directions.into_iter().enumerate() {
        let (own_run, own_open) = line_shape(board, mv, us, dr, dc);
        let (opponent_run, opponent_open) = line_shape(board, mv, -us, dr, dc);
        let base = direction * 4;
        features[base] = own_run.min(5) as f32 / 5.0;
        features[base + 1] = own_open as f32 / 2.0;
        features[base + 2] = opponent_run.min(5) as f32 / 5.0;
        features[base + 3] = opponent_open as f32 / 2.0;
        own_four += usize::from(own_run >= 4);
        opponent_four += usize::from(opponent_run >= 4);
        own_open_three += usize::from(own_run >= 3 && own_open == 2);
        opponent_open_three += usize::from(opponent_run >= 3 && opponent_open == 2);
    }
    features[16] = f32::from(
        own_four > 0
            && directions
                .into_iter()
                .any(|(dr, dc)| line_shape(board, mv, us, dr, dc).0 >= 5),
    );
    features[17] = f32::from(
        opponent_four > 0
            && directions
                .into_iter()
                .any(|(dr, dc)| line_shape(board, mv, -us, dr, dc).0 >= 5),
    );
    features[18] = own_four as f32 / 4.0;
    features[19] = opponent_four as f32 / 4.0;
    features[20] = own_open_three as f32 / 4.0;
    features[21] = opponent_open_three as f32 / 4.0;
    for dr in -2i32..=2 {
        for dc in -2i32..=2 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let row = mv.row() as i32 + dr;
            let col = mv.col() as i32 + dc;
            if row >= 0 && col >= 0 && row < BOARD_SIZE as i32 && col < BOARD_SIZE as i32 {
                let stone = board.cells[row as usize * BOARD_SIZE + col as usize];
                features[if stone == us { 22 } else { 23 }] += f32::from(stone != 0) / 24.0;
            }
        }
    }
    features
}

fn line_shape(board: &Board, mv: Move, stone: i8, dr: i32, dc: i32) -> (usize, usize) {
    let mut run = 1;
    let mut open = 0;
    for sign in [-1, 1] {
        let mut row = mv.row() as i32 + dr * sign;
        let mut col = mv.col() as i32 + dc * sign;
        while row >= 0
            && col >= 0
            && row < BOARD_SIZE as i32
            && col < BOARD_SIZE as i32
            && board.cells[row as usize * BOARD_SIZE + col as usize] == stone
        {
            run += 1;
            row += dr * sign;
            col += dc * sign;
        }
        if row >= 0
            && col >= 0
            && row < BOARD_SIZE as i32
            && col < BOARD_SIZE as i32
            && board.cells[row as usize * BOARD_SIZE + col as usize] == 0
        {
            open += 1;
        }
    }
    (run, open)
}

pub(crate) fn transform_index(index: usize, symmetry: usize) -> usize {
    let mut row = index / BOARD_SIZE;
    let mut col = index % BOARD_SIZE;
    if symmetry & 4 != 0 {
        col = BOARD_SIZE - 1 - col;
    }
    for _ in 0..(symmetry & 3) {
        (row, col) = (col, BOARD_SIZE - 1 - row);
    }
    row * BOARD_SIZE + col
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
    #[test]
    fn eight_symmetries_are_distinct_and_invertible() {
        let point = Move::new(2, 4).unwrap().0;
        let mut mapped = (0..8)
            .map(|s| transform_index(point, s))
            .collect::<Vec<_>>();
        mapped.sort_unstable();
        mapped.dedup();
        assert_eq!(mapped.len(), 8);
        for symmetry in 0..8 {
            assert!(transform_index(point, symmetry) < CELL_COUNT);
        }
    }
    #[test]
    fn tactical_features_detect_win_and_survive_symmetry() {
        let mut board = Board::new();
        for col in 3..7 {
            assert!(board.play(Move::new(7, col).unwrap()));
            assert!(board.play(Move::new(0, col).unwrap()));
        }
        let mv = Move::new(7, 7).unwrap();
        let features = tactical_features(&board, mv);
        assert_eq!(features[16], 1.0);
        assert!(features[18] > 0.0);
        for symmetry in 0..8 {
            let transformed = board.transformed(symmetry);
            let transformed_move = Move(transform_index(mv.0, symmetry));
            let other = tactical_features(&transformed, transformed_move);
            assert_eq!(&features[16..22], &other[16..22]);
        }
    }
}
