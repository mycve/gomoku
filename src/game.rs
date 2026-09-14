use serde::{Deserialize, Serialize};
use std::fmt;

pub const BOARD_SIZE: usize = 19;
pub const COORDINATES: &str = "abcdefghjklmnopqrst";
type PointSet = bitvec::array::BitArray<[u64; CELL_COUNT.div_ceil(64)]>;
pub const CELL_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
pub const ACTION_COUNT: usize = CELL_COUNT + 1;
pub const KOMI: f32 = 7.5;
/// 超过此上限中止对局，不生成胜负训练标签。
pub const MAX_MOVES: usize = CELL_COUNT * 4;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Move(pub usize);
impl Move {
    pub const PASS: Self = Self(CELL_COUNT);
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
        if s == "pass" {
            return Some(Self::PASS);
        }
        let mut chars = s.chars();
        let col = COORDINATES.find(chars.next()?)?;
        let row = chars.as_str().parse::<usize>().ok()?.checked_sub(1)?;
        Self::new(row, col)
    }
    pub fn notation(self) -> String {
        if self == Self::PASS {
            return "pass".into();
        }
        format!(
            "{}{}",
            COORDINATES.as_bytes()[self.col()] as char,
            self.row() + 1
        )
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Win(Player),
    Draw,
    Aborted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Board {
    #[serde(deserialize_with = "deserialize_cells")]
    cells: Vec<i8>,
    to_move: Player,
    moves: usize,
    passes: usize,
    komi_milli: i32,
    #[serde(deserialize_with = "deserialize_cells")]
    previous: Vec<i8>,
    /// 精确盘面历史，用于位置超级劫；pass 不受超级劫限制。
    #[serde(deserialize_with = "deserialize_history")]
    history: Vec<Vec<i8>>,
}
/// 单个候选的精确提子与落子后气数，不分配临时棋盘。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlacementInfo {
    pub captured: PointSet,
    pub liberties: usize,
}

/// 每个局面只建立一次棋块索引，用位集合合并相邻棋块和气。
pub(crate) struct MoveAnalysis {
    at: [usize; CELL_COUNT],
    stones: Vec<PointSet>,
    liberties: Vec<PointSet>,
}
impl MoveAnalysis {
    pub(crate) fn new(groups: &[crate::scoring::Chain]) -> Self {
        let mut at = [usize::MAX; CELL_COUNT];
        let mut stones = Vec::with_capacity(groups.len());
        let mut liberties = Vec::with_capacity(groups.len());
        for (id, chain) in groups.iter().enumerate() {
            let mut mask = PointSet::ZERO;
            for &p in &chain.stones {
                at[p] = id;
                mask.set(p, true);
            }
            stones.push(mask);
            let mut liberty_mask = PointSet::ZERO;
            for &p in &chain.liberties {
                liberty_mask.set(p, true);
            }
            liberties.push(liberty_mask);
        }
        Self {
            at,
            stones,
            liberties,
        }
    }
    pub(crate) fn placement(
        &self,
        board: &Board,
        mv: Move,
        player: Player,
    ) -> Option<PlacementInfo> {
        let point = mv.0;
        if point >= CELL_COUNT || board.cells[point] != 0 {
            return None;
        }
        let mut bit = PointSet::ZERO;
        bit.set(point, true);
        let mut own = bit;
        let mut liberties = PointSet::ZERO;
        let mut captured = PointSet::ZERO;
        for n in neighbors(point) {
            if board.cells[n] == 0 {
                liberties.set(n, true);
            } else {
                let id = self.at[n];
                if board.cells[n] == player.stone() {
                    own |= self.stones[id];
                    liberties |= self.liberties[id];
                } else if self.liberties[id] == bit {
                    captured |= self.stones[id];
                }
            }
        }
        for p in captured.iter_ones() {
            if neighbors(p).any(|n| own[n]) {
                liberties.set(p, true);
            }
        }
        liberties.set(point, false);
        if liberties.not_any() {
            return None;
        }
        // 精确比较历史，保持位置超级劫语义，不依赖有碰撞风险的哈希。
        let mut after = [0; CELL_COUNT];
        after.copy_from_slice(&board.cells);
        after[point] = player.stone();
        for p in captured.iter_ones() {
            after[p] = 0;
        }
        if board.history.iter().any(|old| old.as_slice() == after) {
            return None;
        }
        Some(PlacementInfo {
            captured,
            liberties: liberties.count_ones(),
        })
    }
}

fn deserialize_cells<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<i8>, D::Error> {
    let cells = Vec::<i8>::deserialize(d)?;
    if cells.len() != CELL_COUNT || cells.iter().any(|&s| !(-1..=1).contains(&s)) {
        return Err(serde::de::Error::custom("需要合法的 19×19 盘面"));
    }
    Ok(cells)
}
fn deserialize_history<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Vec<i8>>, D::Error> {
    let history = Vec::<Vec<i8>>::deserialize(d)?;
    if history.is_empty()
        || history
            .iter()
            .any(|cells| cells.len() != CELL_COUNT || cells.iter().any(|&s| !(-1..=1).contains(&s)))
    {
        return Err(serde::de::Error::custom("需要合法的 19×19 历史盘面"));
    }
    Ok(history)
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
impl Board {
    pub fn new() -> Self {
        let cells = vec![0; CELL_COUNT];
        Self {
            history: vec![cells.clone()],
            cells: cells.clone(),
            to_move: Player::Black,
            moves: 0,
            passes: 0,
            komi_milli: 7500,
            previous: cells.clone(),
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
    pub fn consecutive_passes(&self) -> usize {
        self.passes
    }
    /// 摆局必须显式指定行棋方，不能用提子后的黑白子数推断。
    pub fn from_position(stones: &[(Move, Player)], to_move: Player) -> Option<Self> {
        let mut board = Self::new();
        for &(mv, player) in stones {
            if mv.0 >= CELL_COUNT || board.cells[mv.0] != 0 {
                return None;
            }
            board.cells[mv.0] = player.stone();
        }
        for index in 0..CELL_COUNT {
            if board.cells[index] != 0 && !group(&board.cells, index).1 {
                return None;
            }
        }
        board.to_move = to_move;
        board.previous = board.cells.clone();
        board.history = vec![board.cells.clone()];
        Some(board)
    }
    pub(crate) fn transformed(&self, symmetry: usize) -> Self {
        let transform = |cells: &[i8]| {
            let mut result = vec![0; CELL_COUNT];
            for (index, &stone) in cells.iter().enumerate() {
                result[transform_index(index, symmetry)] = stone;
            }
            result
        };
        Self {
            cells: transform(&self.cells),
            history: self.history.iter().map(|cells| transform(cells)).collect(),
            previous: transform(&self.previous),
            ..self.clone()
        }
    }
    pub(crate) fn placed(&self, mv: Move) -> Option<Vec<i8>> {
        self.placed_for(mv, self.to_move)
    }
    pub(crate) fn placed_for(&self, mv: Move, player: Player) -> Option<Vec<i8>> {
        if mv.0 >= CELL_COUNT || self.cells[mv.0] != 0 {
            return None;
        }
        let mut cells = self.cells.clone();
        cells[mv.0] = player.stone();
        for neighbor in neighbors(mv.0) {
            if cells[neighbor] == player.other().stone() {
                let (stones, has_liberty) = group(&cells, neighbor);
                if !has_liberty {
                    for stone in stones {
                        cells[stone] = 0;
                    }
                }
            }
        }
        if !group(&cells, mv.0).1 || self.history.contains(&cells) {
            return None;
        }
        Some(cells)
    }
    pub fn is_legal(&self, mv: Move) -> bool {
        !self.is_finished() && (mv == Move::PASS || self.placed(mv).is_some())
    }
    pub fn play(&mut self, mv: Move) -> bool {
        if self.is_finished() {
            return false;
        }
        let previous = self.cells.clone();
        if mv == Move::PASS {
            self.passes += 1;
        } else {
            let Some(cells) = self.placed(mv) else {
                return false;
            };
            self.cells = cells;
            self.history.push(self.cells.clone());
            self.passes = 0;
        }
        self.previous = previous;
        self.moves += 1;
        self.to_move = self.to_move.other();
        true
    }
    pub fn rule_legal_moves(&self) -> Vec<Move> {
        crate::scope_profile!("game.legal_moves");
        if self.is_finished() {
            return Vec::new();
        }
        let analysis = MoveAnalysis::new(&crate::scoring::chains(&self.cells));
        (0..ACTION_COUNT)
            .map(Move)
            .filter(|&mv| mv == Move::PASS || analysis.placement(self, mv, self.to_move).is_some())
            .collect()
    }
    pub fn search_candidates(&self) -> Vec<Move> {
        self.rule_legal_moves()
    }
    pub fn komi(&self) -> f32 {
        self.komi_milli as f32 / 1000.0
    }
    pub fn set_komi(&mut self, komi: f32) -> bool {
        if !komi.is_finite() || komi.abs() > 1000.0 {
            return false;
        }
        self.komi_milli = (komi * 1000.0).round() as i32;
        true
    }
    pub fn previous_cells(&self) -> &[i8] {
        &self.previous
    }
    pub fn is_finished(&self) -> bool {
        self.passes >= 2 || self.moves >= MAX_MOVES
    }
    /// GTP 允许指定任意行棋方，也允许在停一手后继续处理争议。
    pub fn for_turn(&self, player: Player) -> Self {
        crate::scope_profile!("game.clone_for_turn");
        let mut board = self.clone();
        board.to_move = player;
        board.passes = 0;
        board
    }
    pub(crate) fn for_reading(&self, player: Player) -> Self {
        let mut board = self.for_turn(player);
        board.moves = 0;
        board
    }
    pub fn score(&self) -> f32 {
        crate::scoring::analyze(self).score
    }
    pub fn raw_score(&self) -> f32 {
        crate::scoring::area_score(&self.cells) - self.komi()
    }
    pub fn outcome(&self) -> Option<Outcome> {
        if self.moves >= MAX_MOVES {
            return Some(Outcome::Aborted);
        }
        if self.passes < 2 {
            return None;
        }
        let score = self.score();
        Some(if score > 0.0 {
            Outcome::Win(Player::Black)
        } else if score < 0.0 {
            Outcome::Win(Player::White)
        } else {
            Outcome::Draw
        })
    }
}
pub(crate) fn neighbors(index: usize) -> impl Iterator<Item = usize> {
    let row = index / BOARD_SIZE;
    let col = index % BOARD_SIZE;
    [
        row.checked_sub(1).map(|r| r * BOARD_SIZE + col),
        (row + 1 < BOARD_SIZE).then_some(index + BOARD_SIZE),
        col.checked_sub(1).map(|c| row * BOARD_SIZE + c),
        (col + 1 < BOARD_SIZE).then_some(index + 1),
    ]
    .into_iter()
    .flatten()
}
pub(crate) fn group(cells: &[i8], start: usize) -> (Vec<usize>, bool) {
    let mut stones = vec![start];
    let mut visited = [false; CELL_COUNT];
    visited[start] = true;
    let mut has_liberty = false;
    let mut cursor = 0;
    while cursor < stones.len() {
        for neighbor in neighbors(stones[cursor]) {
            if cells[neighbor] == 0 {
                has_liberty = true;
            } else if cells[neighbor] == cells[start] && !visited[neighbor] {
                visited[neighbor] = true;
                stones.push(neighbor);
            }
        }
        cursor += 1;
    }
    (stones, has_liberty)
}
pub(crate) fn transform_index(index: usize, symmetry: usize) -> usize {
    if index == CELL_COUNT {
        return index;
    }
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
        for col in COORDINATES.chars() {
            write!(f, "{} ", col.to_ascii_uppercase())?;
        }
        writeln!(f)?;
        for row in (0..BOARD_SIZE).rev() {
            write!(f, "{:>2} ", row + 1)?;
            for col in 0..BOARD_SIZE {
                write!(
                    f,
                    "{} ",
                    match self.cells[row * BOARD_SIZE + col] {
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
    fn mv(text: &str) -> Move {
        Move::parse(text).unwrap()
    }
    fn position(black: &[&str], white: &[&str]) -> Board {
        let stones = black
            .iter()
            .map(|s| (mv(s), Player::Black))
            .chain(white.iter().map(|s| (mv(s), Player::White)))
            .collect::<Vec<_>>();
        Board::from_position(&stones, Player::Black).unwrap()
    }
    pub(super) fn ko_position() -> Board {
        position(&["a2", "b1", "c2"], &["b2", "a3", "c3", "b4"])
    }
    #[test]
    fn captures_top_right_corner_above_bit_128() {
        let mut board = position(&["s19"], &["t19"]);
        let analysis = MoveAnalysis::new(&crate::scoring::chains(board.cells()));
        let info = analysis
            .placement(&board, mv("t18"), Player::Black)
            .unwrap();
        assert_eq!(info.captured.count_ones(), 1);
        assert!(info.captured[360]);
        assert!(board.play(mv("t18")));
        assert_eq!(board.cells()[360], 0);
        assert_eq!(Move::PASS.0, 361);
    }

    #[test]
    fn rejects_nine_by_nine_serialized_board() {
        let mut data = serde_json::to_value(Board::new()).unwrap();
        data["cells"] = serde_json::json!(vec![0i8; 81]);
        assert!(serde_json::from_value::<Board>(data).is_err());
        let mut data = serde_json::to_value(Board::new()).unwrap();
        data["history"] = serde_json::json!([vec![0i8; 81]]);
        assert!(serde_json::from_value::<Board>(data).is_err());
    }

    #[test]
    fn bitset_candidates_match_reference_placement() {
        let mut rng = 719u64;
        for _ in 0..12 {
            let mut board = Board::new();
            for _ in 0..180 {
                let analysis = MoveAnalysis::new(&crate::scoring::chains(board.cells()));
                for player in [Player::Black, Player::White] {
                    for point in 0..CELL_COUNT {
                        let reference = board.placed_for(Move(point), player);
                        let actual = analysis.placement(&board, Move(point), player);
                        assert_eq!(
                            actual.is_some(),
                            reference.is_some(),
                            "point={point}, player={player:?}, board={board}"
                        );
                        if let Some(after) = reference {
                            let actual = actual.unwrap();
                            let mut captured = PointSet::ZERO;
                            for p in 0..CELL_COUNT {
                                if board.cells[p] != 0 && after[p] == 0 {
                                    captured.set(p, true);
                                }
                            }
                            assert_eq!(actual.captured, captured);
                            let mut liberties = [false; CELL_COUNT];
                            for p in group(&after, point).0 {
                                for n in neighbors(p) {
                                    if after[n] == 0 {
                                        liberties[n] = true;
                                    }
                                }
                            }
                            assert_eq!(
                                actual.liberties as usize,
                                liberties.iter().filter(|&&v| v).count()
                            );
                        }
                    }
                }
                let moves = board.rule_legal_moves();
                if moves.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                assert!(board.play(moves[rng as usize % moves.len()]));
            }
        }
    }

    #[test]
    fn captures_connected_group_and_rejects_suicide_without_mutation() {
        let mut board = position(&["a2", "b1", "c1", "d2", "b3"], &["b2", "c2"]);
        assert!(board.play(mv("c3")));
        assert_eq!(board.cells()[mv("b2").0], 0);
        assert_eq!(board.cells()[mv("c2").0], 0);
        let mut suicide = position(&[], &["b1", "a2"]);
        let before = suicide.clone();
        assert!(!suicide.play(mv("a1")));
        assert_eq!(suicide, before);
        assert!(!suicide.is_legal(Move(ACTION_COUNT)));
    }
    #[test]
    fn ko_history_survives_symmetry_and_serialization() {
        let mut board = ko_position();
        assert!(board.play(mv("b3")));
        assert!(!board.is_legal(mv("b2")));
        // 相同盘面但无历史的摆局允许回提：合法性确实来自历史。
        let stones = board
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                (s != 0).then_some((Move(i), if s == 1 { Player::Black } else { Player::White }))
            })
            .collect::<Vec<_>>();
        assert!(
            Board::from_position(&stones, Player::White)
                .unwrap()
                .is_legal(mv("b2"))
        );
        let restored: Board =
            serde_json::from_str(&serde_json::to_string(&board).unwrap()).unwrap();
        assert_eq!(restored, board);
        for symmetry in 0..8 {
            let transformed = restored.transformed(symmetry);
            assert!(!transformed.is_legal(Move(transform_index(mv("b2").0, symmetry))));
            assert_eq!(transform_index(Move::PASS.0, symmetry), Move::PASS.0);
        }
    }
    #[test]
    fn passes_end_game_and_placement_resets_pass_count() {
        let mut board = Board::new();
        assert_eq!(board.rule_legal_moves().len(), ACTION_COUNT);
        assert!(board.play(Move::PASS));
        assert!(board.play(mv("e5")));
        assert_eq!(board.consecutive_passes(), 0);
        assert!(board.play(Move::PASS));
        assert!(board.play(Move::PASS));
        assert_eq!(board.outcome(), Some(Outcome::Win(Player::White)));
        assert!(board.rule_legal_moves().is_empty());
        assert!(!board.play(Move::PASS));
    }
    #[test]
    fn area_counts_stones_territory_and_neutral_regions() {
        assert_eq!(Board::new().score(), -KOMI);
        assert_eq!(position(&["e5"], &[]).score(), CELL_COUNT as f32 - KOMI);
        assert_eq!(position(&["a1"], &["j9"]).score(), -KOMI);
        let mut board = position(&["a1", "b1", "c1", "d1", "e1"], &[]);
        assert_eq!(board.outcome(), None); // 五连不结束围棋。
        board.moves = MAX_MOVES - 1;
        assert!(board.play(Move::PASS));
        assert_eq!(board.outcome(), Some(Outcome::Aborted));
    }
    #[test]
    fn coordinates_skip_i_and_roundtrip_all_actions() {
        for index in 0..ACTION_COUNT {
            assert_eq!(Move::parse(&Move(index).notation()), Some(Move(index)));
        }
        for text in ["i1", "a0", "a20", "u1", "中1", "@1", ""] {
            assert!(Move::parse(text).is_none());
        }
    }
}
