use crate::{
    game::{BOARD_SIZE, Board, CELL_COUNT, Move, Outcome, Player, SEARCH_CANDIDATE_RADIUS},
    model::{EvalAccumulator, EvalScratch, PolicyValueModel},
};
use std::{collections::HashMap, time::Instant};

const MATE_SCORE: f32 = 10_000.0;
const PVS_EPSILON: f32 = 1.0e-4;

#[derive(Clone, Copy, Debug)]
pub struct AlphaBetaConfig {
    pub max_depth: u16,
    /// 0 表示不按节点数截断，完整完成迭代深度。
    pub max_nodes: u64,
    pub threat_extension_depth: u16,
}

impl Default for AlphaBetaConfig {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_nodes: 100_000,
            threat_extension_depth: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AlphaBetaResult {
    pub best_move: Option<Move>,
    pub value: f32,
    pub completed_depth: u16,
    pub nodes: u64,
    pub tt_hits: u64,
    pub beta_cutoffs: u64,
    pub threat_nodes: u64,
    pub elapsed_seconds: f64,
    pub principal_variation: Vec<Move>,
}

impl AlphaBetaResult {
    pub fn proven_win(&self) -> bool {
        self.value > MATE_SCORE - 512.0
    }

    pub fn proven_loss(&self) -> bool {
        self.value < -MATE_SCORE + 512.0
    }

    pub fn mate_distance(&self) -> Option<u16> {
        (self.proven_win() || self.proven_loss())
            .then(|| (MATE_SCORE - self.value.abs()).round().max(0.0) as u16)
    }
}

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    depth: u16,
    value: f32,
    bound: Bound,
    best_move: Option<Move>,
}

struct Searcher<'a> {
    model: &'a PolicyValueModel,
    cfg: AlphaBetaConfig,
    scratch: EvalScratch,
    table: HashMap<u64, TtEntry>,
    nodes: u64,
    tt_hits: u64,
    beta_cutoffs: u64,
    threat_nodes: u64,
}

struct SearchPosition {
    board: Board,
    accumulators: Vec<EvalAccumulator>,
    candidate_counts: [u8; CELL_COUNT],
    winning_black: [[bool; 4]; CELL_COUNT],
    winning_white: [[bool; 4]; CELL_COUNT],
    ply: usize,
    history: Vec<(Move, Player, Option<Move>)>,
}

impl SearchPosition {
    fn new(board: &Board, model: &PolicyValueModel, max_ply: usize) -> Self {
        let root = model.accumulator(board);
        let mut position = Self {
            board: board.clone(),
            accumulators: vec![root; max_ply.max(1) + 1],
            candidate_counts: [0; CELL_COUNT],
            winning_black: [[false; 4]; CELL_COUNT],
            winning_white: [[false; 4]; CELL_COUNT],
            ply: 0,
            history: Vec::new(),
        };
        for (index, &stone) in board.cells().iter().enumerate() {
            if stone != 0 {
                position.adjust_candidate_neighborhood(Move(index), 1);
            }
        }
        position.refresh_all_winning_moves();
        position
    }

    fn accumulator(&self) -> &EvalAccumulator {
        &self.accumulators[self.ply]
    }

    fn adjust_candidate_neighborhood(&mut self, mv: Move, delta: i8) {
        for dr in -SEARCH_CANDIDATE_RADIUS..=SEARCH_CANDIDATE_RADIUS {
            for dc in -SEARCH_CANDIDATE_RADIUS..=SEARCH_CANDIDATE_RADIUS {
                let row = mv.row() as i32 + dr;
                let col = mv.col() as i32 + dc;
                if row >= 0 && col >= 0 && row < BOARD_SIZE as i32 && col < BOARD_SIZE as i32 {
                    let count =
                        &mut self.candidate_counts[row as usize * BOARD_SIZE + col as usize];
                    *count = if delta > 0 { *count + 1 } else { *count - 1 };
                }
            }
        }
    }

    fn search_candidates(&self) -> Vec<Move> {
        if self.board.move_count() == 0 {
            return vec![Move::new(BOARD_SIZE / 2, BOARD_SIZE / 2).unwrap()];
        }
        self.candidate_counts
            .iter()
            .enumerate()
            .filter_map(|(index, &count)| {
                (count > 0 && self.board.cells()[index] == 0).then_some(Move(index))
            })
            .collect()
    }

    fn refresh_all_winning_moves(&mut self) {
        for index in 0..CELL_COUNT {
            for axis in 0..4 {
                self.refresh_winning_axis(Move(index), axis);
            }
        }
    }

    fn refresh_winning_axis(&mut self, mv: Move, axis: usize) {
        if self.board.cells()[mv.0] != 0 {
            self.winning_black[mv.0][axis] = false;
            self.winning_white[mv.0][axis] = false;
        } else {
            self.winning_black[mv.0][axis] =
                would_complete_axis(&self.board, mv, Player::Black, axis);
            self.winning_white[mv.0][axis] =
                would_complete_axis(&self.board, mv, Player::White, axis);
        }
    }

    fn refresh_winning_lines(&mut self, changed: Move) {
        for (axis, (dr, dc)) in [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)]
            .into_iter()
            .enumerate()
        {
            self.refresh_winning_axis(changed, axis);
            for sign in [-1_i32, 1] {
                for distance in 1..=4 {
                    let row = changed.row() as i32 + dr * sign * distance;
                    let col = changed.col() as i32 + dc * sign * distance;
                    if row >= 0 && col >= 0 && row < BOARD_SIZE as i32 && col < BOARD_SIZE as i32 {
                        self.refresh_winning_axis(
                            Move(row as usize * BOARD_SIZE + col as usize),
                            axis,
                        );
                    }
                }
            }
        }
    }

    fn winning_moves(&self, moves: &[Move], player: Player) -> Vec<Move> {
        moves
            .iter()
            .copied()
            .filter(|&mv| self.is_winning_move(mv, player))
            .collect()
    }

    fn is_winning_move(&self, mv: Move, player: Player) -> bool {
        match player {
            Player::Black => self.winning_black[mv.0].iter().any(|&value| value),
            Player::White => self.winning_white[mv.0].iter().any(|&value| value),
        }
    }

    fn forcing_moves(&self, moves: &[Move]) -> Option<ForcingMoves> {
        let winning = self.winning_moves(moves, self.board.to_move());
        if !winning.is_empty() {
            return Some(ForcingMoves::Win(winning));
        }
        let blocks = self.winning_moves(moves, self.board.to_move().other());
        (!blocks.is_empty()).then_some(ForcingMoves::Block(blocks))
    }

    fn created_immediate_threat(&self, last_move: Move, player: Player) -> bool {
        for (dr, dc) in [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)] {
            for sign in [-1_i32, 1] {
                for distance in 1..=4 {
                    let row = last_move.row() as i32 + dr * sign * distance;
                    let col = last_move.col() as i32 + dc * sign * distance;
                    if row >= 0
                        && col >= 0
                        && row < BOARD_SIZE as i32
                        && col < BOARD_SIZE as i32
                        && self
                            .is_winning_move(Move(row as usize * BOARD_SIZE + col as usize), player)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn make_move(&mut self, model: &PolicyValueModel, mv: Move) -> bool {
        if !self.board.is_legal(mv) {
            return false;
        }
        let player = self.board.to_move();
        let previous_last = self.board.last_move();
        let next_ply = self.ply + 1;
        let (parents, children) = self.accumulators.split_at_mut(next_ply);
        children[0].clone_from(&parents[self.ply]);
        model.accumulator_prepare_move(&mut children[0], &self.board, mv, player);
        let played = self.board.play(mv);
        debug_assert!(played);
        model.accumulator_finish_move(&mut children[0], &self.board, mv);
        self.adjust_candidate_neighborhood(mv, 1);
        self.refresh_winning_lines(mv);
        self.ply = next_ply;
        self.history.push((mv, player, previous_last));
        true
    }

    fn undo_move(&mut self, _model: &PolicyValueModel) {
        let (mv, player, previous_last) = self.history.pop().expect("撤销栈不能为空");
        self.adjust_candidate_neighborhood(mv, -1);
        self.board.undo(mv, player, previous_last);
        self.refresh_winning_lines(mv);
        self.ply -= 1;
    }
}

pub fn search(board: &Board, model: &PolicyValueModel, cfg: AlphaBetaConfig) -> AlphaBetaResult {
    let started = Instant::now();
    let mut position = SearchPosition::new(
        board,
        model,
        cfg.max_depth as usize + cfg.threat_extension_depth as usize + 2,
    );
    let mut searcher = Searcher {
        model,
        cfg,
        scratch: EvalScratch::new(model.hidden_size),
        table: HashMap::new(),
        nodes: 0,
        tt_hits: 0,
        beta_cutoffs: 0,
        threat_nodes: 0,
    };
    let root_moves = board.search_candidates();
    let mut best_move = root_moves.first().copied();
    let mut best_value = 0.0;
    let mut completed_depth = 0;

    for depth in 1..=cfg.max_depth {
        match searcher.negamax(&mut position, depth, 0, -MATE_SCORE, MATE_SCORE) {
            Ok((value, mv)) => {
                best_value = value;
                best_move = mv.or(best_move);
                completed_depth = depth;
                if value.abs() >= MATE_SCORE - depth as f32 - 1.0 {
                    break;
                }
            }
            Err(()) => break,
        }
    }

    let principal_variation =
        searcher.principal_variation(board, position.accumulator(), completed_depth);
    AlphaBetaResult {
        best_move,
        value: best_value,
        completed_depth,
        nodes: searcher.nodes,
        tt_hits: searcher.tt_hits,
        beta_cutoffs: searcher.beta_cutoffs,
        threat_nodes: searcher.threat_nodes,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        principal_variation,
    }
}

impl Searcher<'_> {
    fn node_budget_exhausted(&self) -> bool {
        self.cfg.max_nodes != 0 && self.nodes >= self.cfg.max_nodes
    }

    fn negamax(
        &mut self,
        position: &mut SearchPosition,
        depth: u16,
        ply: u16,
        mut alpha: f32,
        beta: f32,
    ) -> Result<(f32, Option<Move>), ()> {
        if self.node_budget_exhausted() {
            return Err(());
        }
        self.nodes += 1;

        if let Some(outcome) = position.board.outcome() {
            return Ok((terminal_value(outcome, position.board.to_move(), ply), None));
        }
        if depth == 0 {
            return self.threat_quiescence(
                position,
                ply,
                self.cfg.threat_extension_depth,
                alpha,
                beta,
                true,
            );
        }

        let original_alpha = alpha;
        let hash = position.accumulator().hash;
        let tt_move = if let Some(entry) = self.table.get(&hash).copied() {
            if entry.depth >= depth {
                self.tt_hits += 1;
                match entry.bound {
                    Bound::Exact => return Ok((entry.value, entry.best_move)),
                    Bound::Lower if entry.value >= beta => {
                        return Ok((entry.value, entry.best_move));
                    }
                    Bound::Upper if entry.value <= alpha => {
                        return Ok((entry.value, entry.best_move));
                    }
                    _ => {}
                }
            }
            entry.best_move
        } else {
            None
        };

        let candidates = position.search_candidates();
        let moves = match position.forcing_moves(&candidates) {
            Some(ForcingMoves::Win(moves) | ForcingMoves::Block(moves)) => moves,
            None => candidates,
        };
        if moves.is_empty() {
            return Ok((0.0, None));
        }
        let ordered = self.order_moves(position, &moves, tt_move);
        let mut best_value = -MATE_SCORE;
        let mut best_move = None;

        for (index, mv) in ordered.into_iter().enumerate() {
            if !position.make_move(self.model, mv) {
                continue;
            }
            let first_result = if index == 0 {
                self.negamax(position, depth - 1, ply + 1, -beta, -alpha)
            } else {
                self.negamax(position, depth - 1, ply + 1, -alpha - PVS_EPSILON, -alpha)
            };
            position.undo_move(self.model);
            let mut value = -first_result?.0;
            if index > 0 && value > alpha && value < beta {
                position.make_move(self.model, mv);
                let research = self.negamax(position, depth - 1, ply + 1, -beta, -alpha);
                position.undo_move(self.model);
                value = -research?.0;
            }
            if value > best_value {
                best_value = value;
                best_move = Some(mv);
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                self.beta_cutoffs += 1;
                break;
            }
        }

        let bound = if best_value <= original_alpha {
            Bound::Upper
        } else if best_value >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.insert(
            hash,
            TtEntry {
                depth,
                value: best_value,
                bound,
                best_move,
            },
        );
        Ok((best_value, best_move))
    }

    fn order_moves(
        &mut self,
        position: &SearchPosition,
        moves: &[Move],
        tt_move: Option<Move>,
    ) -> Vec<Move> {
        let board = &position.board;
        let accumulator = position.accumulator();
        let player = board.to_move();
        let opponent = player.other();
        let logits =
            self.model
                .move_logits_accumulator(board, accumulator, moves, &mut self.scratch);
        let mut scored = logits
            .into_iter()
            .map(|(mv, logit)| {
                let tactical = if Some(mv) == tt_move {
                    4.0e9
                } else if position.is_winning_move(mv, player) {
                    3.0e9
                } else if position.is_winning_move(mv, opponent) {
                    2.0e9
                } else {
                    0.0
                };
                (mv, tactical + logit)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.into_iter().map(|(mv, _)| mv).collect()
    }

    fn threat_quiescence(
        &mut self,
        position: &mut SearchPosition,
        ply: u16,
        remaining: u16,
        mut alpha: f32,
        beta: f32,
        check_trigger: bool,
    ) -> Result<(f32, Option<Move>), ()> {
        if let Some(outcome) = position.board.outcome() {
            return Ok((terminal_value(outcome, position.board.to_move(), ply), None));
        }
        if remaining == 0 {
            return Ok((
                self.model.evaluate_value_accumulator(
                    &position.board,
                    position.accumulator(),
                    &mut self.scratch,
                ),
                None,
            ));
        }
        if check_trigger {
            let Some(last_move) = position.board.last_move() else {
                return Ok((
                    self.model.evaluate_value_accumulator(
                        &position.board,
                        position.accumulator(),
                        &mut self.scratch,
                    ),
                    None,
                ));
            };
            if !position.created_immediate_threat(last_move, position.board.to_move().other()) {
                return Ok((
                    self.model.evaluate_value_accumulator(
                        &position.board,
                        position.accumulator(),
                        &mut self.scratch,
                    ),
                    None,
                ));
            }
        }
        let candidates = position.search_candidates();
        let Some(forcing) = position.forcing_moves(&candidates) else {
            return Ok((
                self.model.evaluate_value_accumulator(
                    &position.board,
                    position.accumulator(),
                    &mut self.scratch,
                ),
                None,
            ));
        };
        if let ForcingMoves::Win(moves) = &forcing {
            return Ok((MATE_SCORE - (ply + 1) as f32, moves.first().copied()));
        }

        let ForcingMoves::Block(moves) = forcing else {
            unreachable!();
        };
        let ordered = self.order_moves(position, &moves, None);
        let mut best_value = -MATE_SCORE;
        let mut best_move = None;
        for mv in ordered {
            if self.node_budget_exhausted() {
                return Err(());
            }
            self.nodes += 1;
            self.threat_nodes += 1;
            if !position.make_move(self.model, mv) {
                continue;
            }
            let result =
                self.threat_quiescence(position, ply + 1, remaining - 1, -beta, -alpha, false);
            position.undo_move(self.model);
            let value = -result?.0;
            if value > best_value {
                best_value = value;
                best_move = Some(mv);
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                self.beta_cutoffs += 1;
                break;
            }
        }
        Ok((best_value, best_move))
    }

    fn principal_variation(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        depth: u16,
    ) -> Vec<Move> {
        let mut board = board.clone();
        let mut accumulator = accumulator.clone();
        let mut pv = Vec::new();
        for _ in 0..depth {
            let Some(mv) = self
                .table
                .get(&accumulator.hash)
                .and_then(|entry| entry.best_move)
            else {
                break;
            };
            let player = board.to_move();
            let before = board.clone();
            if !board.play(mv) {
                break;
            }
            accumulator =
                self.model
                    .accumulator_after_move(&accumulator, &before, &board, mv, player);
            pv.push(mv);
        }
        pv
    }
}

fn terminal_value(outcome: Outcome, to_move: Player, ply: u16) -> f32 {
    match outcome {
        Outcome::Draw => 0.0,
        Outcome::Win(player) if player == to_move => MATE_SCORE - ply as f32,
        Outcome::Win(_) => -MATE_SCORE + ply as f32,
    }
}

#[cfg(test)]
fn would_win(board: &Board, mv: Move, player: Player) -> bool {
    if !board.is_legal(mv) {
        return false;
    }
    would_complete_five(board, mv, player)
}

#[cfg(test)]
fn would_complete_five(board: &Board, mv: Move, player: Player) -> bool {
    if mv.0 >= CELL_COUNT || board.cells()[mv.0] != 0 {
        return false;
    }
    (0..4).any(|axis| would_complete_axis(board, mv, player, axis))
}

fn would_complete_axis(board: &Board, mv: Move, player: Player, axis: usize) -> bool {
    let (dr, dc) = [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)][axis];
    let mut stones = 1;
    for sign in [-1_i32, 1] {
        let mut row = mv.row() as i32 + dr * sign;
        let mut col = mv.col() as i32 + dc * sign;
        while row >= 0
            && col >= 0
            && row < BOARD_SIZE as i32
            && col < BOARD_SIZE as i32
            && board.cells()[row as usize * BOARD_SIZE + col as usize] == player.stone()
        {
            stones += 1;
            row += dr * sign;
            col += dc * sign;
        }
    }
    stones >= 5
}

enum ForcingMoves {
    Win(Vec<Move>),
    Block(Vec<Move>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_immediate_win() {
        let board = Board::from_stones(&[
            (Move::new(7, 3).unwrap(), Player::Black),
            (Move::new(0, 0).unwrap(), Player::White),
            (Move::new(7, 4).unwrap(), Player::Black),
            (Move::new(0, 1).unwrap(), Player::White),
            (Move::new(7, 5).unwrap(), Player::Black),
            (Move::new(0, 2).unwrap(), Player::White),
            (Move::new(7, 6).unwrap(), Player::Black),
            (Move::new(1, 0).unwrap(), Player::White),
        ])
        .unwrap();
        let model = PolicyValueModel::random(16, 41);
        let result = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 2,
                max_nodes: 10_000,
                threat_extension_depth: 8,
            },
        );
        assert!(matches!(result.best_move, Some(mv) if would_win(&board, mv, Player::Black)));
        assert!(result.value > MATE_SCORE - 3.0);
        assert!(result.nodes < 10, "nodes={}", result.nodes);
    }

    #[test]
    fn obeys_node_budget() {
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let model = PolicyValueModel::random(16, 43);
        let result = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 8,
                max_nodes: 128,
                threat_extension_depth: 8,
            },
        );
        assert!(result.nodes <= 128);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn zero_node_budget_means_unlimited_depth_search() {
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let model = PolicyValueModel::random(8, 31);
        let result = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 2,
                max_nodes: 0,
                threat_extension_depth: 0,
            },
        );
        assert_eq!(result.completed_depth, 2);
        assert!(result.nodes > 0);
    }

    #[test]
    fn search_position_restores_after_multiple_moves() {
        let model = PolicyValueModel::random(16, 47);
        let board = Board::new();
        let mut position = SearchPosition::new(&board, &model, 16);
        let original_hash = position.accumulator().hash;
        for text in ["h8", "h9", "i8", "g8", "i9", "g9"] {
            assert!(position.make_move(&model, Move::parse(text).unwrap()));
            assert_eq!(
                position.search_candidates(),
                position.board.search_candidates()
            );
            for mv in position.search_candidates() {
                assert_eq!(
                    position.is_winning_move(mv, Player::Black),
                    would_win(&position.board, mv, Player::Black)
                );
                assert_eq!(
                    position.is_winning_move(mv, Player::White),
                    would_win(&position.board, mv, Player::White)
                );
            }
        }
        for _ in 0..6 {
            position.undo_move(&model);
            assert_eq!(
                position.search_candidates(),
                position.board.search_candidates()
            );
        }
        assert_eq!(position.board.cells(), board.cells());
        assert_eq!(position.board.to_move(), board.to_move());
        assert_eq!(position.board.move_count(), 0);
        assert_eq!(position.board.last_move(), None);
        assert_eq!(position.accumulator().hash, original_hash);
        let mut scratch = EvalScratch::new(model.hidden_size);
        let restored =
            model.evaluate_value_accumulator(&position.board, position.accumulator(), &mut scratch);
        assert!((restored - model.evaluate_value(&board)).abs() < 1e-6);
    }

    #[test]
    fn blocks_opponents_only_immediate_win() {
        let board = Board::from_stones(&[
            (Move::new(7, 2).unwrap(), Player::Black),
            (Move::new(7, 3).unwrap(), Player::White),
            (Move::new(0, 0).unwrap(), Player::Black),
            (Move::new(7, 4).unwrap(), Player::White),
            (Move::new(0, 1).unwrap(), Player::Black),
            (Move::new(7, 5).unwrap(), Player::White),
            (Move::new(0, 2).unwrap(), Player::Black),
            (Move::new(7, 6).unwrap(), Player::White),
        ])
        .unwrap();
        let model = PolicyValueModel::random(16, 47);
        let result = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 2,
                max_nodes: 20_000,
                threat_extension_depth: 8,
            },
        );
        assert_eq!(result.best_move, Move::new(7, 7));
    }

    #[test]
    fn threat_extension_finds_open_four_win_beyond_main_depth() {
        let board = Board::from_stones(&[
            (Move::new(7, 5).unwrap(), Player::Black),
            (Move::new(0, 0).unwrap(), Player::White),
            (Move::new(7, 6).unwrap(), Player::Black),
            (Move::new(0, 1).unwrap(), Player::White),
            (Move::new(7, 7).unwrap(), Player::Black),
            (Move::new(0, 2).unwrap(), Player::White),
        ])
        .unwrap();
        let model = PolicyValueModel::random(16, 53);
        let without_extension = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 1,
                max_nodes: 20_000,
                threat_extension_depth: 0,
            },
        );
        let with_extension = search(
            &board,
            &model,
            AlphaBetaConfig {
                max_depth: 1,
                max_nodes: 20_000,
                threat_extension_depth: 8,
            },
        );

        assert!(without_extension.value.abs() < MATE_SCORE - 100.0);
        assert!(with_extension.value > MATE_SCORE - 10.0);
        assert!(matches!(
            with_extension.best_move,
            Some(mv) if mv == Move::new(7, 4).unwrap() || mv == Move::new(7, 8).unwrap()
        ));
        assert!(with_extension.threat_nodes > 0);
    }
}
