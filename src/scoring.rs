//! 保守死活分析：Benson 活棋 + 在确定活棋围成的小区域内完整枚举防守着。
//! 搜索耗尽预算只返回未定，不能把“未证明活”直接当成死。
use crate::game::{Board, CELL_COUNT, Move, Player, neighbors};

#[derive(Clone, Debug)]
pub struct Chain {
    pub stones: Vec<usize>,
    pub liberties: Vec<usize>,
    pub color: i8,
}

pub fn chains(cells: &[i8]) -> Vec<Chain> {
    crate::scope_profile!("scoring.chains");
    let mut seen = [false; CELL_COUNT];
    let mut liberty_stamp = [0_usize; CELL_COUNT];
    let mut result = Vec::new();
    for start in 0..CELL_COUNT {
        if cells[start] == 0 || seen[start] {
            continue;
        }
        let mut stones = vec![start];
        seen[start] = true;
        let mut liberties = Vec::new();
        let stamp = result.len() + 1;
        let mut cursor = 0;
        while cursor < stones.len() {
            for n in neighbors(stones[cursor]) {
                if cells[n] == 0 && liberty_stamp[n] != stamp {
                    liberty_stamp[n] = stamp;
                    liberties.push(n);
                } else if cells[n] == cells[start] && !seen[n] {
                    seen[n] = true;
                    stones.push(n);
                }
            }
            cursor += 1;
        }
        result.push(Chain {
            stones,
            liberties,
            color: cells[start],
        });
    }
    result
}

/// 仅使用纯空区域证明活棋，比扩展 Benson 保守；绝不依赖对手配合。
pub fn pass_alive(cells: &[i8]) -> [bool; CELL_COUNT] {
    crate::scope_profile!("scoring.pass_alive");
    pass_alive_with_chains(cells, &chains(cells))
}

pub(crate) fn pass_alive_with_chains(cells: &[i8], groups: &[Chain]) -> [bool; CELL_COUNT] {
    crate::scope_profile!("scoring.benson");
    let mut at = [usize::MAX; CELL_COUNT];
    for (i, chain) in groups.iter().enumerate() {
        for &s in &chain.stones {
            at[s] = i;
        }
    }
    let mut regions = Vec::new();
    let mut seen = [false; CELL_COUNT];
    for start in 0..CELL_COUNT {
        if cells[start] != 0 || seen[start] {
            continue;
        }
        let mut points = vec![start];
        let mut boundary = Vec::new();
        seen[start] = true;
        let mut cursor = 0;
        while cursor < points.len() {
            for n in neighbors(points[cursor]) {
                if cells[n] == 0 {
                    if !seen[n] {
                        seen[n] = true;
                        points.push(n);
                    }
                } else if !boundary.contains(&at[n]) {
                    boundary.push(at[n]);
                }
            }
            cursor += 1;
        }
        if let Some(&first) = boundary.first() {
            if boundary
                .iter()
                .all(|&g| groups[g].color == groups[first].color)
            {
                regions.push((points, boundary));
            }
        }
    }
    let mut alive = vec![true; groups.len()];
    loop {
        let next = groups
            .iter()
            .enumerate()
            .map(|(i, chain)| {
                alive[i]
                    && regions
                        .iter()
                        .filter(|(points, boundary)| {
                            boundary.contains(&i)
                                && boundary.iter().all(|&g| alive[g])
                                && points.iter().all(|p| chain.liberties.contains(p))
                        })
                        .count()
                        >= 2
            })
            .collect::<Vec<_>>();
        if next == alive {
            break;
        }
        alive = next;
    }
    let mut result = [false; CELL_COUNT];
    for (i, chain) in groups.iter().enumerate() {
        for &s in &chain.stones {
            result[s] = alive[i];
        }
    }
    result
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub alive: Vec<Vec<Move>>,
    pub seki: Vec<Vec<Move>>,
    pub dead: Vec<Vec<Move>>,
    pub unsettled: Vec<Vec<Move>>,
    pub score: f32,
}

pub fn analyze(board: &Board) -> Analysis {
    crate::scope_profile!("scoring.analyze");
    let cells = board.cells();
    let safe = pass_alive(cells);
    let mut result = Analysis {
        alive: Vec::new(),
        seki: Vec::new(),
        dead: Vec::new(),
        unsettled: Vec::new(),
        score: 0.0,
    };
    let mut remaining = cells.to_vec();
    let groups = chains(cells);
    let mutual = simple_seki(board, &groups);
    for (index, chain) in groups.into_iter().enumerate() {
        let moves = chain.stones.iter().copied().map(Move).collect();
        if safe[chain.stones[0]] {
            result.alive.push(moves);
            continue;
        }
        if mutual[index] {
            result.seki.push(moves);
            continue;
        }
        let mut region = vec![chain.stones[0]];
        let mut seen = [false; CELL_COUNT];
        seen[region[0]] = true;
        let mut enclosed = true;
        let mut touches_boundary = false;
        let mut cursor = 0;
        while cursor < region.len() {
            for n in neighbors(region[cursor]) {
                if safe[n] {
                    touches_boundary = true;
                    if cells[n] == chain.color {
                        enclosed = false;
                    }
                } else if !seen[n] {
                    seen[n] = true;
                    region.push(n);
                }
            }
            cursor += 1;
        }
        // 不能跨越未证明活的进攻方棋块做局部证明，避免漏掉外部救援。
        enclosed &= touches_boundary
            && region
                .iter()
                .all(|&p| cells[p] == 0 || cells[p] == chain.color);
        let defender = if chain.color == 1 {
            Player::Black
        } else {
            Player::White
        };
        let dead = enclosed && region.len() <= 8 && {
            let mut budget = 4000;
            forced_capture(
                &board.for_reading(defender),
                &region,
                chain.stones[0],
                defender,
                region.len() * 2 + 4,
                &mut budget,
            ) == Some(true)
        };
        if dead {
            for &s in &chain.stones {
                remaining[s] = 0;
            }
            result.dead.push(moves);
        } else {
            result.unsettled.push(moves);
        }
    }
    result.score = area_score(&remaining) - board.komi();
    result
}

/// 两个棋块仅有同一对公气，任一方填气都会被立即整块提走，且不存在倒扑。
/// 更复杂的双活仍返回未定。
fn simple_seki(board: &Board, groups: &[Chain]) -> Vec<bool> {
    let mut seki = vec![false; groups.len()];
    for (i, a) in groups.iter().enumerate() {
        if a.liberties.len() != 2 {
            continue;
        }
        for (j, b) in groups.iter().enumerate().skip(i + 1) {
            if a.color == b.color
                || b.liberties.len() != 2
                || !a.liberties.iter().all(|p| b.liberties.contains(p))
            {
                continue;
            }
            // 不得存在第三块相邻敌棋，否则外部提子可能解除公气关系。
            if [(a, b), (b, a)].into_iter().any(|(own, other)| {
                own.stones.iter().any(|&s| {
                    neighbors(s)
                        .any(|n| board.cells()[n] == other.color && !other.stones.contains(&n))
                })
            }) {
                continue;
            }
            let safe = [a, b].into_iter().all(|chain| {
                let player = if chain.color == 1 {
                    Player::Black
                } else {
                    Player::White
                };
                chain.liberties.iter().all(|&point| {
                    let mut child = board.for_reading(player);
                    if !child.play(Move(point)) {
                        return true;
                    }
                    let reply = chain
                        .liberties
                        .iter()
                        .copied()
                        .find(|&p| p != point)
                        .unwrap();
                    if !child.play(Move(reply))
                        || chain
                            .stones
                            .iter()
                            .any(|&s| child.cells()[s] == chain.color)
                    {
                        return false;
                    }
                    // 提子方至少留两气，排除简单倒扑。
                    chains(child.cells())
                        .iter()
                        .find(|c| c.stones.contains(&reply))
                        .is_some_and(|c| c.liberties.len() >= 2)
                })
            });
            if safe {
                seki[i] = true;
                seki[j] = true;
            }
        }
    }
    seki
}

fn forced_capture(
    board: &Board,
    region: &[usize],
    target: usize,
    defender: Player,
    depth: usize,
    budget: &mut usize,
) -> Option<bool> {
    if board.cells()[target] != defender.stone() {
        return Some(true);
    }
    if board.consecutive_passes() >= 2 {
        return Some(false);
    }
    if depth == 0 || *budget == 0 {
        return None;
    }
    *budget -= 1;
    let attacking = board.to_move() != defender;
    let mut unknown = false;
    for mv in region
        .iter()
        .copied()
        .map(Move)
        .chain(std::iter::once(Move::PASS))
    {
        let mut child = board.clone();
        if !child.play(mv) {
            continue;
        }
        match forced_capture(&child, region, target, defender, depth - 1, budget) {
            Some(captured) if captured == attacking => return Some(captured),
            None => unknown = true,
            _ => {}
        }
    }
    if unknown { None } else { Some(!attacking) }
}

pub fn area_score(cells: &[i8]) -> f32 {
    let mut score = cells.iter().map(|&s| s as f32).sum::<f32>();
    let mut seen = [false; CELL_COUNT];
    for start in 0..CELL_COUNT {
        if cells[start] != 0 || seen[start] {
            continue;
        }
        let mut region = vec![start];
        seen[start] = true;
        let mut border = 0;
        let mut cursor = 0;
        while cursor < region.len() {
            for n in neighbors(region[cursor]) {
                match cells[n] {
                    1 => border |= 1,
                    -1 => border |= 2,
                    _ if !seen[n] => {
                        seen[n] = true;
                        region.push(n);
                    }
                    _ => {}
                }
            }
            cursor += 1;
        }
        score += match border {
            1 => region.len() as f32,
            2 => -(region.len() as f32),
            _ => 0.0,
        };
    }
    score
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn enclosed_dead() -> Board {
        let empty = ["b2", "d2", "c5"].map(|s| Move::parse(s).unwrap().0);
        let victim = Move::parse("c4").unwrap().0;
        let stones = (0..CELL_COUNT)
            .filter(|p| !empty.contains(p))
            .map(|p| {
                (
                    Move(p),
                    if p == victim {
                        Player::White
                    } else {
                        Player::Black
                    },
                )
            })
            .collect::<Vec<_>>();
        Board::from_position(&stones, Player::Black).unwrap()
    }
    #[test]
    fn proves_enclosed_dead_without_removing_live_chain() {
        let board = enclosed_dead();
        let result = analyze(&board);
        assert_eq!(result.dead, vec![vec![Move::parse("c4").unwrap()]]);
        assert_eq!(result.alive.len(), 1);
        assert!(result.unsettled.is_empty());
        assert_eq!(result.score, CELL_COUNT as f32 - board.komi());
        assert!(result.score > board.raw_score());
        assert_eq!(board.cells()[Move::parse("c4").unwrap().0], -1);
        for symmetry in 0..8 {
            let other = analyze(&board.transformed(symmetry));
            assert_eq!(other.score, result.score);
            assert_eq!(other.dead.len(), 1);
        }
    }
    #[test]
    fn one_eye_and_unsettled_groups_are_never_assumed_dead() {
        let eye = Move::parse("b2").unwrap().0;
        let stones = (0..CELL_COUNT)
            .filter(|&p| p != eye)
            .map(|p| (Move(p), Player::Black))
            .collect::<Vec<_>>();
        let board = Board::from_position(&stones, Player::Black).unwrap();
        assert!(pass_alive(board.cells()).iter().all(|&x| !x));
        let result = analyze(&board);
        assert!(result.dead.is_empty());
        assert_eq!(result.unsettled.len(), 1);
    }
    #[test]
    fn recognizes_two_shared_liberty_seki_without_deleting_either_side() {
        let holes = [Move::parse("a2").unwrap().0, Move::parse("b1").unwrap().0];
        let stones = (0..CELL_COUNT)
            .filter(|p| !holes.contains(p))
            .map(|p| (Move(p), if p == 0 { Player::Black } else { Player::White }))
            .collect::<Vec<_>>();
        let board = Board::from_position(&stones, Player::Black).unwrap();
        let result = analyze(&board);
        assert_eq!(result.seki.len(), 2);
        assert!(result.dead.is_empty());
        assert!(result.unsettled.is_empty());
        assert_eq!(result.score, board.raw_score());
    }
    #[test]
    fn reading_budget_exhaustion_is_unknown() {
        let board = enclosed_dead().for_reading(Player::White);
        let point = Move::parse("c4").unwrap().0;
        assert_eq!(
            forced_capture(&board, &[point], point, Player::White, 10, &mut 0),
            None
        );
    }
}
