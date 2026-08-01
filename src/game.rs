use serde::{Deserialize, Serialize};
use std::fmt;

pub const BOARD_SIZE: usize = 15;
pub const CELL_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
pub const SEARCH_CANDIDATE_RADIUS: i32 = 3;

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
        let col = (chars.next()? as usize).checked_sub('a' as usize)?;
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
    pub fn last_move(&self) -> Option<Move> {
        self.last
    }
    pub fn from_stones(stones: &[(Move, Player)]) -> Option<Self> {
        let mut cells = vec![0; CELL_COUNT];
        let mut black = 0usize;
        let mut white = 0usize;
        for &(mv, player) in stones {
            if mv.0 >= CELL_COUNT || cells[mv.0] != 0 {
                return None;
            }
            cells[mv.0] = player.stone();
            match player {
                Player::Black => black += 1,
                Player::White => white += 1,
            }
        }
        if black != white && black != white + 1 {
            return None;
        }
        Some(Self {
            cells,
            to_move: if black == white {
                Player::Black
            } else {
                Player::White
            },
            moves: stones.len(),
            last: None,
        })
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
    /// 返回规则允许的全部落点，不包含任何搜索剪枝或启发式。
    pub fn rule_legal_moves(&self) -> Vec<Move> {
        crate::scope_profile!("game.rule_legal_moves");
        if self.outcome().is_some() {
            return vec![];
        }
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(index, &stone)| (stone == 0).then_some(Move(index)))
            .collect()
    }
    /// 返回引擎实际评估的候选点；这是性能启发式，不代表规则合法性。
    pub fn search_candidates(&self) -> Vec<Move> {
        if self.outcome().is_some() {
            return vec![];
        }
        if self.moves == 0 {
            return vec![Move::new(BOARD_SIZE / 2, BOARD_SIZE / 2).unwrap()];
        }
        let mut near = [false; CELL_COUNT];
        for index in 0..CELL_COUNT {
            if self.cells[index] == 0 {
                continue;
            }
            let row = index / BOARD_SIZE;
            let col = index % BOARD_SIZE;
            for dr in -SEARCH_CANDIDATE_RADIUS..=SEARCH_CANDIDATE_RADIUS {
                for dc in -SEARCH_CANDIDATE_RADIUS..=SEARCH_CANDIDATE_RADIUS {
                    let next_row = row as i32 + dr;
                    let next_col = col as i32 + dc;
                    if next_row >= 0
                        && next_col >= 0
                        && next_row < BOARD_SIZE as i32
                        && next_col < BOARD_SIZE as i32
                    {
                        let candidate = next_row as usize * BOARD_SIZE + next_col as usize;
                        if self.cells[candidate] == 0 {
                            near[candidate] = true;
                        }
                    }
                }
            }
        }
        let candidates = near
            .iter()
            .enumerate()
            .filter_map(|(index, &is_near)| is_near.then_some(Move(index)))
            .collect::<Vec<_>>();
        assert!(
            self.moves == CELL_COUNT || !candidates.is_empty(),
            "非终局且棋盘未满时，半径 {SEARCH_CANDIDATE_RADIUS} 搜索候选不能为空"
        );
        candidates
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
        if self.last.is_none() {
            for index in 0..CELL_COUNT {
                let stone = self.cells[index];
                if stone == 0 {
                    continue;
                }
                let mv = Move(index);
                for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                    if line_shape(self, mv, stone, dr, dc).0 >= 5 {
                        return Some(Outcome::Win(if stone == 1 {
                            Player::Black
                        } else {
                            Player::White
                        }));
                    }
                }
            }
        }
        (self.moves == CELL_COUNT).then_some(Outcome::Draw)
    }
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
    fn rule_legal_moves_include_every_empty_point() {
        let mut board = Board::new();
        assert_eq!(board.rule_legal_moves().len(), CELL_COUNT);
        assert!(board.play(Move::new(7, 7).unwrap()));
        let legal = board.rule_legal_moves();
        assert_eq!(legal.len(), CELL_COUNT - 1);
        assert!(legal.contains(&Move::new(0, 0).unwrap()));
        assert!(legal.contains(&Move::new(14, 14).unwrap()));
        assert!(!legal.contains(&Move::new(7, 7).unwrap()));
    }

    #[test]
    fn search_candidates_use_explicit_radius_three_without_changing_rules() {
        let board = Board::new();
        assert_eq!(board.search_candidates(), vec![Move::new(7, 7).unwrap()]);
        let mut board = board;
        assert!(board.play(Move::new(7, 7).unwrap()));
        assert_eq!(board.search_candidates().len(), 48);
        assert!(
            !board
                .search_candidates()
                .contains(&Move::new(0, 0).unwrap())
        );
        assert!(board.rule_legal_moves().contains(&Move::new(0, 0).unwrap()));
    }
    #[test]
    fn notation_roundtrip() {
        for s in ["a1", "h8", "o15"] {
            let m = Move::parse(s).unwrap();
            assert_eq!(m.notation(), s);
        }
    }

    #[test]
    fn notation_parser_rejects_out_of_range_input_without_panicking() {
        assert!(Move::parse("@1").is_none());
        assert!(Move::parse("p1").is_none());
        assert!(Move::parse("a0").is_none());
        assert!(Move::parse("中1").is_none());
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
    fn restores_protocol_position_without_move_history() {
        let stones = [
            (Move::new(7, 3).unwrap(), Player::Black),
            (Move::new(0, 0).unwrap(), Player::White),
            (Move::new(7, 4).unwrap(), Player::Black),
            (Move::new(0, 1).unwrap(), Player::White),
            (Move::new(7, 5).unwrap(), Player::Black),
            (Move::new(0, 2).unwrap(), Player::White),
            (Move::new(7, 6).unwrap(), Player::Black),
            (Move::new(0, 3).unwrap(), Player::White),
            (Move::new(7, 7).unwrap(), Player::Black),
        ];
        let board = Board::from_stones(&stones).unwrap();
        assert_eq!(board.to_move(), Player::White);
        assert_eq!(board.outcome(), Some(Outcome::Win(Player::Black)));
    }
}
