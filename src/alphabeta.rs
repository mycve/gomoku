use crate::{
    game::{BOARD_SIZE, Board, Move, Outcome, Player},
    model::{EvalAccumulator, EvalScratch, PolicyValueModel},
};
use std::{collections::HashMap, time::Instant};

const MATE_SCORE: f32 = 10_000.0;
const PVS_EPSILON: f32 = 1.0e-4;

#[derive(Clone, Copy, Debug)]
pub struct AlphaBetaConfig {
    pub max_depth: u16,
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
        self.value >= MATE_SCORE - self.completed_depth as f32 - 1.0
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

pub fn search(board: &Board, model: &PolicyValueModel, cfg: AlphaBetaConfig) -> AlphaBetaResult {
    let started = Instant::now();
    let accumulator = model.accumulator(board);
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
        match searcher.negamax(board, &accumulator, depth, 0, -MATE_SCORE, MATE_SCORE) {
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

    let principal_variation = searcher.principal_variation(board, &accumulator, completed_depth);
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
    fn negamax(
        &mut self,
        board: &Board,
        accumulator: &EvalAccumulator,
        depth: u16,
        ply: u16,
        mut alpha: f32,
        beta: f32,
    ) -> Result<(f32, Option<Move>), ()> {
        if self.nodes >= self.cfg.max_nodes {
            return Err(());
        }
        self.nodes += 1;

        if let Some(outcome) = board.outcome() {
            return Ok((terminal_value(outcome, board.to_move(), ply), None));
        }
        if depth == 0 {
            return self.threat_quiescence(
                board,
                accumulator,
                ply,
                self.cfg.threat_extension_depth,
                alpha,
                beta,
                true,
            );
        }

        let original_alpha = alpha;
        let hash = accumulator.hash;
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

        let moves = match forcing_moves(board) {
            Some(ForcingMoves::Win(moves) | ForcingMoves::Block(moves)) => moves,
            None => board.search_candidates(),
        };
        if moves.is_empty() {
            return Ok((0.0, None));
        }
        let ordered = self.order_moves(board, accumulator, &moves, tt_move);
        let mut best_value = -MATE_SCORE;
        let mut best_move = None;

        for (index, mv) in ordered.into_iter().enumerate() {
            let player = board.to_move();
            let mut child_board = board.clone();
            if !child_board.play(mv) {
                continue;
            }
            let child_accumulator =
                self.model
                    .accumulator_after_move(accumulator, board, &child_board, mv, player);
            let mut value = if index == 0 {
                -self
                    .negamax(
                        &child_board,
                        &child_accumulator,
                        depth - 1,
                        ply + 1,
                        -beta,
                        -alpha,
                    )?
                    .0
            } else {
                -self
                    .negamax(
                        &child_board,
                        &child_accumulator,
                        depth - 1,
                        ply + 1,
                        -alpha - PVS_EPSILON,
                        -alpha,
                    )?
                    .0
            };
            if index > 0 && value > alpha && value < beta {
                value = -self
                    .negamax(
                        &child_board,
                        &child_accumulator,
                        depth - 1,
                        ply + 1,
                        -beta,
                        -alpha,
                    )?
                    .0;
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
        board: &Board,
        accumulator: &EvalAccumulator,
        moves: &[Move],
        tt_move: Option<Move>,
    ) -> Vec<Move> {
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
                } else if would_win(board, mv, player) {
                    3.0e9
                } else if would_win(board, mv, opponent) {
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
        board: &Board,
        accumulator: &EvalAccumulator,
        ply: u16,
        remaining: u16,
        mut alpha: f32,
        beta: f32,
        check_trigger: bool,
    ) -> Result<(f32, Option<Move>), ()> {
        if let Some(outcome) = board.outcome() {
            return Ok((terminal_value(outcome, board.to_move(), ply), None));
        }
        if remaining == 0 {
            return Ok((
                self.model
                    .evaluate_value_accumulator(board, accumulator, &mut self.scratch),
                None,
            ));
        }
        if check_trigger {
            let Some(last_move) = board.last_move() else {
                return Ok((
                    self.model
                        .evaluate_value_accumulator(board, accumulator, &mut self.scratch),
                    None,
                ));
            };
            if !created_immediate_threat(board, last_move, board.to_move().other()) {
                return Ok((
                    self.model
                        .evaluate_value_accumulator(board, accumulator, &mut self.scratch),
                    None,
                ));
            }
        }
        let Some(forcing) = forcing_moves(board) else {
            return Ok((
                self.model
                    .evaluate_value_accumulator(board, accumulator, &mut self.scratch),
                None,
            ));
        };
        if let ForcingMoves::Win(moves) = &forcing {
            return Ok((MATE_SCORE - (ply + 1) as f32, moves.first().copied()));
        }

        let ForcingMoves::Block(moves) = forcing else {
            unreachable!();
        };
        let ordered = self.order_moves(board, accumulator, &moves, None);
        let mut best_value = -MATE_SCORE;
        let mut best_move = None;
        for mv in ordered {
            if self.nodes >= self.cfg.max_nodes {
                return Err(());
            }
            self.nodes += 1;
            self.threat_nodes += 1;
            let player = board.to_move();
            let mut child_board = board.clone();
            if !child_board.play(mv) {
                continue;
            }
            let child_accumulator =
                self.model
                    .accumulator_after_move(accumulator, board, &child_board, mv, player);
            let value = -self
                .threat_quiescence(
                    &child_board,
                    &child_accumulator,
                    ply + 1,
                    remaining - 1,
                    -beta,
                    -alpha,
                    false,
                )?
                .0;
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

fn would_win(board: &Board, mv: Move, player: Player) -> bool {
    if !board.is_legal(mv) {
        return false;
    }
    for (dr, dc) in [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)] {
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
        if stones >= 5 {
            return true;
        }
    }
    false
}

fn created_immediate_threat(board: &Board, last_move: Move, player: Player) -> bool {
    for (dr, dc) in [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)] {
        for sign in [-1_i32, 1] {
            for distance in 1..=4 {
                let row = last_move.row() as i32 + dr * sign * distance;
                let col = last_move.col() as i32 + dc * sign * distance;
                if row < 0 || col < 0 || row >= BOARD_SIZE as i32 || col >= BOARD_SIZE as i32 {
                    continue;
                }
                let mv = Move::new(row as usize, col as usize).unwrap();
                if would_win(board, mv, player) {
                    return true;
                }
            }
        }
    }
    false
}

enum ForcingMoves {
    Win(Vec<Move>),
    Block(Vec<Move>),
}

fn forcing_moves(board: &Board) -> Option<ForcingMoves> {
    let moves = board.search_candidates();
    let player = board.to_move();
    let winning = moves
        .iter()
        .copied()
        .filter(|&mv| would_win(board, mv, player))
        .collect::<Vec<_>>();
    if !winning.is_empty() {
        return Some(ForcingMoves::Win(winning));
    }
    let blocks = moves
        .iter()
        .copied()
        .filter(|&mv| would_win(board, mv, player.other()))
        .collect::<Vec<_>>();
    if !blocks.is_empty() {
        return Some(ForcingMoves::Block(blocks));
    }
    None
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
