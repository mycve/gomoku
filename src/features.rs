//! 训练与推理共用的围棋输入，全部按指定行棋方的相对视角编码。
use crate::{
    game::{Board, CELL_COUNT, Move, Player, neighbors},
    scoring::{chains, pass_alive_with_chains},
};

pub const GO_PLANES: usize = 18;
pub const GO_FEATURE_SIZE: usize = GO_PLANES * CELL_COUNT + 1;

pub fn encode(board: &Board, perspective: Player) -> Vec<f32> {
    crate::scope_profile!("features.encode");
    let cells = board.cells();
    let mut data = vec![0.0; GO_FEATURE_SIZE];
    let groups = chains(cells);
    let safe = pass_alive_with_chains(cells, &groups);
    for chain in groups {
        let side = usize::from(chain.color != perspective.stone());
        let liberties = chain.liberties.len().clamp(1, 3) - 1;
        for &point in &chain.stones {
            data[(side * 3 + liberties) * CELL_COUNT + point] = 1.0;
            data[(6 + side) * CELL_COUNT + point] = chain.stones.len() as f32 / CELL_COUNT as f32;
            data[(13 + side) * CELL_COUNT + point] = f32::from(safe[point]);
        }
    }
    {
        crate::scope_profile!("features.candidate_analysis");
        for point in 0..CELL_COUNT {
            for side in 0..2 {
                let player = if side == 0 {
                    perspective
                } else {
                    perspective.other()
                };
                data[(8 + side) * CELL_COUNT + point] = f32::from(is_eye(cells, point, player));
                data[(15 + side) * CELL_COUNT + point] =
                    f32::from(board.previous_cells()[point] == player.stone());
            }
            data[17 * CELL_COUNT + point] =
                f32::from(board.previous_cells()[point] != cells[point]);
            if let Some(after) = board.placed_for(Move(point), perspective) {
                data[10 * CELL_COUNT + point] = 1.0;
                let captured = cells
                    .iter()
                    .zip(&after)
                    .filter(|(a, b)| **a == perspective.other().stone() && **b == 0)
                    .count();
                data[11 * CELL_COUNT + point] = captured as f32 / CELL_COUNT as f32;
                let chain = crate::game::group(&after, point).0;
                let mut liberties = [false; CELL_COUNT];
                for stone in chain {
                    for n in neighbors(stone) {
                        if after[n] == 0 {
                            liberties[n] = true;
                        }
                    }
                }
                data[12 * CELL_COUNT + point] =
                    f32::from(liberties.iter().filter(|&&x| x).count() == 1);
            }
        }
    }
    // 正值表示当前视角得到贴目优势。
    data[GO_PLANES * CELL_COUNT] = if perspective == Player::White {
        board.komi()
    } else {
        -board.komi()
    } / CELL_COUNT as f32;
    data
}

/// 局部真眼特征，不作为整个棋块活棋的证明。
pub fn is_eye(cells: &[i8], point: usize, player: Player) -> bool {
    if cells[point] != 0 || neighbors(point).any(|n| cells[n] != player.stone()) {
        return false;
    }
    let row = point / crate::game::BOARD_SIZE;
    let col = point % crate::game::BOARD_SIZE;
    let mut diagonals = 0;
    let mut hostile = 0;
    for dr in [-1, 1] {
        for dc in [-1, 1] {
            let r = row as i32 + dr;
            let c = col as i32 + dc;
            if r >= 0 && c >= 0 && r < 9 && c < 9 {
                diagonals += 1;
                hostile += usize::from(cells[r as usize * 9 + c as usize] != player.stone());
            }
        }
    }
    if diagonals < 4 {
        hostile == 0
    } else {
        hostile <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_legality_matches_rules_through_games() {
        for seed in 0..8 {
            let mut board = Board::new();
            for ply in 0..180 {
                for role in [Player::Black, Player::White] {
                    let position = board.for_turn(role);
                    let data = encode(&board, role);
                    assert_eq!(
                        moves_from_mask(&position, &data[10 * CELL_COUNT..11 * CELL_COUNT]),
                        position.search_candidates()
                    );
                }
                let moves = board.search_candidates();
                if moves.is_empty() {
                    break;
                }
                let mv = moves[(ply * 37 + seed * 13) % moves.len()];
                assert!(board.play(mv));
            }
            let mut terminal = Board::new();
            terminal.play(Move::PASS);
            terminal.play(Move::PASS);
            let data = encode(&terminal, terminal.to_move());
            assert!(moves_from_mask(&terminal, &data[10 * CELL_COUNT..11 * CELL_COUNT]).is_empty());
        }
    }
    #[test]
    fn encodes_liberties_captures_history_eyes_and_komi() {
        let board = crate::scoring::tests::enclosed_dead();
        let data = encode(&board, Player::Black);
        let victim = Move::parse("c4").unwrap().0;
        let capture = Move::parse("c5").unwrap().0;
        assert_eq!(data[3 * CELL_COUNT + victim], 1.0);
        assert_eq!(data[11 * CELL_COUNT + capture], 1.0 / CELL_COUNT as f32);
        assert_eq!(data[8 * CELL_COUNT + Move::parse("b2").unwrap().0], 1.0);
        assert_eq!(data[GO_PLANES * CELL_COUNT], -7.5 / CELL_COUNT as f32);
        let mut after = board.clone();
        assert!(after.play(Move(capture)));
        let features = encode(&after, Player::White);
        assert_eq!(features[15 * CELL_COUNT + victim], 1.0);
        assert_eq!(features[17 * CELL_COUNT + victim], 1.0);
        assert_eq!(features[17 * CELL_COUNT + capture], 1.0);
        assert!(after.play(Move::PASS));
        assert_eq!(after.previous_cells(), after.cells());
    }
    #[test]
    fn all_feature_planes_transform_with_board() {
        let board = crate::scoring::tests::enclosed_dead();
        let features = encode(&board, Player::White);
        for symmetry in 0..8 {
            let transformed = encode(&board.transformed(symmetry), Player::White);
            for plane in 0..GO_PLANES {
                for point in 0..CELL_COUNT {
                    assert_eq!(
                        features[plane * CELL_COUNT + point],
                        transformed
                            [plane * CELL_COUNT + crate::game::transform_index(point, symmetry)]
                    );
                }
            }
        }
    }
}

/// 复用合法落点平面，终局与停着遵循棋盘规则。
pub(crate) fn moves_from_mask(board: &Board, legal: &[f32]) -> Vec<Move> {
    if board.is_finished() {
        return Vec::new();
    }
    (0..CELL_COUNT)
        .filter(|&p| legal[p] != 0.0)
        .map(Move)
        .chain(std::iter::once(Move::PASS))
        .collect()
}
