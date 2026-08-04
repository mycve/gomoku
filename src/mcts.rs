use crate::{
    game::{Board, CELL_COUNT, Move, Outcome, transform_index},
    model::{EvalAccumulator, EvalScratch, PolicyValueModel},
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    pub cpuct: f32,
    pub cpuct_log: f32,
    pub cpuct_base: f32,
    pub root_desired_per_child_visits_coeff: f32,
    pub root_dirichlet_total_concentration: f32,
    pub root_exploration_fraction: f32,
    pub root_noise_seed: u64,
    pub policy_softmax_temp: f32,
    pub root_policy_temperature_early: f32,
    pub root_policy_temperature: f32,
    pub root_policy_temperature_halflife: f32,
    pub root_num_symmetries_to_sample: usize,
    pub use_graph_search: bool,
    pub graph_search_max_nodes: usize,
    pub use_lcb_for_selection: bool,
    pub lcb_stdevs: f32,
    pub min_visit_prop_for_lcb: f32,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub balanced_opening_probability: f32,
    pub policy_opening_probability: f32,
    pub policy_opening_avg_plies: usize,
    pub policy_opening_temperature: f32,
    pub early_fork_game_prob: f32,
    pub early_fork_max_ply: usize,
    pub early_fork_max_choices: usize,
    pub asymmetric_playout_prob: f32,
    pub max_asymmetric_ratio: f32,
    pub opening_random_plies: usize,
    pub opening_seed: u64,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 3000,
            cpuct: 1.5,
            cpuct_log: 0.45,
            cpuct_base: 500.0,
            root_desired_per_child_visits_coeff: 2.0,
            root_dirichlet_total_concentration: 0.0,
            root_exploration_fraction: 0.0,
            root_noise_seed: 0,
            policy_softmax_temp: 1.0,
            root_policy_temperature_early: 1.6,
            root_policy_temperature: 1.15,
            root_policy_temperature_halflife: 6.0,
            root_num_symmetries_to_sample: 4,
            use_graph_search: true,
            graph_search_max_nodes: 65_536,
            use_lcb_for_selection: true,
            lcb_stdevs: 3.0,
            min_visit_prop_for_lcb: 0.15,
            temperature_start: 0.0,
            temperature_endgame: 0.0,
            temperature_decay_delay_plies: 0,
            temperature_decay_plies: 0,
            temperature_value_cutoff: 0.0,
            temperature_visit_offset: 0.0,
            balanced_opening_probability: 0.0,
            policy_opening_probability: 0.0,
            policy_opening_avg_plies: 0,
            policy_opening_temperature: 1.0,
            early_fork_game_prob: 0.0,
            early_fork_max_ply: 0,
            early_fork_max_choices: 0,
            asymmetric_playout_prob: 0.0,
            max_asymmetric_ratio: 1.0,
            opening_random_plies: 0,
            opening_seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
    pub raw_prior: f32,
    pub value_std_error: f32,
}
pub struct SearchOutput {
    pub candidates: Vec<Candidate>,
    pub root_value: f32,
}
struct Node {
    board: Board,
    accumulator: EvalAccumulator,
    children: Vec<Edge>,
    expanded: bool,
    initial_value: f32,
}
struct Edge {
    mv: Move,
    prior: f32,
    visits: u32,
    value_sum: f32,
    value_square_sum: f32,
    raw_prior: f32,
    child: Option<usize>,
}

pub fn search(board: &Board, model: &PolicyValueModel, cfg: SearchConfig) -> Vec<Candidate> {
    search_with_info(board, model, cfg).candidates
}

pub fn search_with_info(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
) -> SearchOutput {
    search_until(board, model, cfg, None)
}

pub fn search_timed(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    time_limit: Duration,
) -> Vec<Candidate> {
    search_until(board, model, cfg, Some(Instant::now() + time_limit)).candidates
}

fn search_until(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    deadline: Option<Instant>,
) -> SearchOutput {
    crate::scope_profile!("mcts.search");
    let mut scratch = EvalScratch::new(model.hidden_size);
    let mut nodes = vec![Node {
        board: board.clone(),
        accumulator: model.accumulator(board),
        children: vec![],
        expanded: false,
        initial_value: 0.0,
    }];
    let mut transpositions = HashMap::new();
    transpositions.insert(board_hash(board), vec![0]);
    expand(&mut nodes, 0, model, cfg, &mut scratch);
    for simulation in 0..cfg.simulations {
        if simulation > 0 && deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        simulate(&mut nodes, &mut transpositions, 0, model, cfg, &mut scratch);
    }
    let mut out: Vec<_> = nodes[0]
        .children
        .iter()
        .map(|e| Candidate {
            mv: e.mv,
            visits: e.visits,
            q: if e.visits == 0 {
                0.0
            } else {
                e.value_sum / e.visits as f32
            },
            prior: e.prior,
            raw_prior: e.raw_prior,
            value_std_error: edge_std_error(e),
        })
        .collect();
    if cfg.use_lcb_for_selection {
        let max_visits = out.iter().map(|x| x.visits).max().unwrap_or(0) as f32;
        out.sort_by(|a, b| {
            lcb_selection_value(b, max_visits, cfg)
                .total_cmp(&lcb_selection_value(a, max_visits, cfg))
        });
    } else {
        out.sort_by_key(|x| std::cmp::Reverse(x.visits));
    }
    SearchOutput {
        candidates: out,
        root_value: nodes[0].initial_value,
    }
}
fn expand(
    nodes: &mut Vec<Node>,
    idx: usize,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    scratch: &mut EvalScratch,
) -> f32 {
    crate::scope_profile!("mcts.expand");
    if let Some(out) = nodes[idx].board.outcome() {
        return match out {
            Outcome::Draw => 0.0,
            Outcome::Win(p) => {
                if p == nodes[idx].board.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
    }
    let cached_tactics = !(idx == 0 && cfg.root_num_symmetries_to_sample > 1);
    let (mut policy, value) = {
        crate::scope_profile!("mcts.nn_eval");
        if idx == 0 && cfg.root_num_symmetries_to_sample > 1 {
            evaluate_root_symmetries(&nodes[idx].board, model, cfg)
        } else {
            model.evaluate_accumulator_with_scratch(
                &nodes[idx].board,
                &nodes[idx].accumulator,
                cfg.policy_softmax_temp,
                scratch,
            )
        }
    };
    nodes[idx].initial_value = value;
    let mut raw_priors = [0.0_f32; CELL_COUNT];
    for &(mv, prior) in &policy {
        raw_priors[mv.0] = prior;
    }
    if idx == 0 {
        apply_policy_temperature(
            &mut policy,
            root_policy_temperature(cfg, nodes[idx].board.move_count()),
        );
    }
    let us = nodes[idx].board.to_move();
    let winning = policy
        .iter()
        .filter_map(|&(mv, _)| {
            (if cached_tactics {
                scratch.is_winning_move(mv, false)
            } else {
                nodes[idx].board.is_winning_move(mv, us)
            })
            .then_some(mv)
        })
        .collect::<Vec<_>>();
    if !winning.is_empty() {
        let prior = 1.0 / winning.len() as f32;
        policy = winning.into_iter().map(|mv| (mv, prior)).collect();
    } else {
        let forced_blocks = policy
            .iter()
            .filter_map(|&(mv, _)| {
                (if cached_tactics {
                    scratch.is_winning_move(mv, true)
                } else {
                    nodes[idx].board.is_winning_move(mv, us.other())
                })
                .then_some(mv)
            })
            .collect::<Vec<_>>();
        if !forced_blocks.is_empty() {
            let prior = 1.0 / forced_blocks.len() as f32;
            policy = forced_blocks.into_iter().map(|mv| (mv, prior)).collect();
        } else if idx == 0 {
            let forcing_threshold = policy
                .iter()
                .map(|(_, prior)| *prior)
                .fold(0.0_f32, f32::max)
                * 0.1;
            let forcing = policy
                .iter()
                .filter_map(|&(mv, prior)| {
                    (prior >= forcing_threshold
                        && nodes[idx].board.winning_replies_after(mv, us) >= 2)
                        .then_some(mv)
                })
                .collect::<Vec<_>>();
            if !forcing.is_empty() {
                let prior = 1.0 / forcing.len() as f32;
                policy = forcing.into_iter().map(|mv| (mv, prior)).collect();
            }
        }
    }
    if idx == 0
        && cfg.root_dirichlet_total_concentration > 0.0
        && cfg.root_exploration_fraction > 0.0
    {
        let mut priors = policy.iter().map(|(_, prior)| *prior).collect::<Vec<_>>();
        let alpha = cfg.root_dirichlet_total_concentration / priors.len().max(1) as f32;
        apply_root_dirichlet_noise(
            &mut priors,
            alpha,
            cfg.root_exploration_fraction.clamp(0.0, 1.0),
            cfg.root_noise_seed,
        );
        for ((_, prior), noisy) in policy.iter_mut().zip(priors) {
            *prior = noisy;
        }
    }
    {
        crate::scope_profile!("mcts.children_build");
        nodes[idx].children = policy
            .into_iter()
            .map(|(mv, prior)| Edge {
                mv,
                prior,
                raw_prior: raw_priors[mv.0],
                visits: 0,
                value_sum: 0.0,
                value_square_sum: 0.0,
                child: None,
            })
            .collect();
    }
    nodes[idx].expanded = true;
    value
}
fn simulate(
    nodes: &mut Vec<Node>,
    transpositions: &mut HashMap<u64, Vec<usize>>,
    idx: usize,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    scratch: &mut EvalScratch,
) -> f32 {
    if let Some(out) = nodes[idx].board.outcome() {
        return match out {
            Outcome::Draw => 0.0,
            Outcome::Win(p) => {
                if p == nodes[idx].board.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
    }
    if !nodes[idx].expanded {
        return expand(nodes, idx, model, cfg, scratch);
    }
    let best = {
        crate::scope_profile!("mcts.select_child");
        let total = nodes[idx]
            .children
            .iter()
            .map(|e| e.visits)
            .sum::<u32>()
            .max(1) as f32;
        let mut best = 0;
        let mut score = f32::NEG_INFINITY;
        let cpuct = cfg.cpuct
            + cfg.cpuct_log.max(0.0)
                * ((total + cfg.cpuct_base.max(1.0) + 1.0) / cfg.cpuct_base.max(1.0)).ln();
        let exploration = cpuct * total.sqrt();
        for (i, e) in nodes[idx].children.iter().enumerate() {
            let q = if e.visits == 0 {
                0.0
            } else {
                e.value_sum / e.visits as f32
            };
            let desired = if idx == 0 {
                cfg.root_desired_per_child_visits_coeff.max(0.0) * e.prior * total.sqrt()
            } else {
                0.0
            };
            let effective_visits = (e.visits as f32 - desired).max(0.0);
            let s = q + exploration * e.prior / (1.0 + effective_visits);
            if s > score {
                score = s;
                best = i
            }
        }
        best
    };
    let child = if let Some(c) = nodes[idx].children[best].child {
        c
    } else {
        crate::scope_profile!("mcts.create_child");
        let mut b = nodes[idx].board.clone();
        let mv = nodes[idx].children[best].mv;
        let player = b.to_move();
        let accumulator = model.accumulator_after_move(&nodes[idx].accumulator, mv, player);
        b.play(mv);
        let key = board_hash(&b);
        let c = if cfg.use_graph_search {
            transpositions.get(&key).and_then(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .find(|&candidate| same_position(&nodes[candidate].board, &b))
            })
        } else {
            None
        }
        .unwrap_or_else(|| {
            let c = nodes.len();
            nodes.push(Node {
                board: b,
                accumulator,
                children: vec![],
                expanded: false,
                initial_value: 0.0,
            });
            if cfg.use_graph_search && nodes.len() <= cfg.graph_search_max_nodes.max(1) {
                transpositions.entry(key).or_insert_with(Vec::new).push(c);
            }
            c
        });
        nodes[idx].children[best].child = Some(c);
        c
    };
    let value = -simulate(nodes, transpositions, child, model, cfg, scratch);
    let e = &mut nodes[idx].children[best];
    e.visits += 1;
    e.value_sum += value;
    e.value_square_sum += value * value;
    value
}

fn board_hash(board: &Board) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64 ^ board.to_move().stone() as u64;
    for (index, &stone) in board.cells().iter().enumerate() {
        hash ^= ((stone as i64 + 2) as u64).wrapping_mul((index as u64 + 1) * 0x9E37_79B9);
        hash = hash.wrapping_mul(0x100_0000_01B3);
    }
    hash
}

fn same_position(a: &Board, b: &Board) -> bool {
    a.to_move() == b.to_move() && a.cells() == b.cells()
}

fn edge_std_error(edge: &Edge) -> f32 {
    if edge.visits <= 1 {
        return 1.0;
    }
    let n = edge.visits as f32;
    let mean = edge.value_sum / n;
    ((edge.value_square_sum / n - mean * mean).max(0.0) / n).sqrt()
}

fn lcb_selection_value(candidate: &Candidate, max_visits: f32, cfg: SearchConfig) -> f32 {
    if candidate.visits == 0
        || (candidate.visits as f32) < max_visits * cfg.min_visit_prop_for_lcb.max(0.0)
    {
        return f32::NEG_INFINITY;
    }
    candidate.q - cfg.lcb_stdevs.max(0.0) * candidate.value_std_error
}

fn root_policy_temperature(cfg: SearchConfig, ply: usize) -> f32 {
    let half_life = cfg.root_policy_temperature_halflife.max(1e-3);
    cfg.root_policy_temperature
        + (cfg.root_policy_temperature_early - cfg.root_policy_temperature)
            * 2.0_f32.powf(-(ply as f32) / half_life)
}

fn apply_policy_temperature(policy: &mut [(Move, f32)], temperature: f32) {
    let inverse = temperature.max(1e-3).recip();
    let sum = policy
        .iter_mut()
        .map(|(_, probability)| {
            *probability = probability.max(1e-12).powf(inverse);
            *probability
        })
        .sum::<f32>()
        .max(1e-12);
    for (_, probability) in policy {
        *probability /= sum;
    }
}

fn evaluate_root_symmetries(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
) -> (Vec<(Move, f32)>, f32) {
    let count = cfg.root_num_symmetries_to_sample.clamp(1, 8);
    let mut probabilities = vec![0.0f32; CELL_COUNT];
    let mut value = 0.0;
    for index in 0..count {
        let symmetry = (cfg.root_noise_seed as usize + index * 3) % 8;
        let mut inverse = [0usize; CELL_COUNT];
        for original in 0..CELL_COUNT {
            inverse[transform_index(original, symmetry)] = original;
        }
        let transformed = board.transformed(symmetry);
        let accumulator = model.accumulator(&transformed);
        let (policy, prediction) = model.evaluate_accumulator_with_temperature(
            &transformed,
            &accumulator,
            cfg.policy_softmax_temp,
        );
        value += prediction / count as f32;
        for (mv, probability) in policy {
            probabilities[inverse[mv.0]] += probability / count as f32;
        }
    }
    let policy = board
        .rule_legal_moves()
        .into_iter()
        .map(|mv| (mv, probabilities[mv.0]))
        .collect();
    (policy, value)
}

fn apply_root_dirichlet_noise(priors: &mut [f32], alpha: f32, fraction: f32, seed: u64) {
    let mut rng = SplitMix64(seed ^ 0xD1A1_71C7_0000_0000 ^ priors.len() as u64);
    let mut noise = Vec::with_capacity(priors.len());
    let mut sum = 0.0;
    for _ in 0..priors.len() {
        let value = sample_gamma(alpha.max(1e-3), &mut rng).max(1e-12);
        noise.push(value);
        sum += value;
    }
    let keep = 1.0 - fraction;
    for (prior, value) in priors.iter_mut().zip(noise) {
        *prior = keep * *prior + fraction * value / sum.max(1e-12);
    }
}

fn sample_gamma(alpha: f32, rng: &mut SplitMix64) -> f32 {
    if alpha < 1.0 {
        return sample_gamma(alpha + 1.0, rng) * rng.unit().max(1e-12).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x =
            (-2.0 * rng.unit().max(1e-12).ln()).sqrt() * (std::f32::consts::TAU * rng.unit()).cos();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.unit();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    #[test]
    fn immediate_win_replaces_network_policy() {
        let board = Board::from_stones(&[
            (Move::parse("d8").unwrap(), Player::Black),
            (Move::parse("d7").unwrap(), Player::White),
            (Move::parse("e8").unwrap(), Player::Black),
            (Move::parse("e7").unwrap(), Player::White),
            (Move::parse("f8").unwrap(), Player::Black),
            (Move::parse("f7").unwrap(), Player::White),
            (Move::parse("g8").unwrap(), Player::Black),
            (Move::parse("a1").unwrap(), Player::White),
        ])
        .unwrap();
        let result = search(
            &board,
            &PolicyValueModel::random(8, 23),
            SearchConfig {
                simulations: 2,
                ..Default::default()
            },
        );
        assert!(result.iter().all(|candidate| {
            candidate.mv == Move::parse("c8").unwrap() || candidate.mv == Move::parse("h8").unwrap()
        }));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn immediate_loss_restricts_search_to_blocks() {
        let board = Board::from_stones(&[
            (Move::parse("a1").unwrap(), Player::Black),
            (Move::parse("d8").unwrap(), Player::White),
            (Move::parse("a3").unwrap(), Player::Black),
            (Move::parse("e8").unwrap(), Player::White),
            (Move::parse("a5").unwrap(), Player::Black),
            (Move::parse("f8").unwrap(), Player::White),
            (Move::parse("a7").unwrap(), Player::Black),
            (Move::parse("g8").unwrap(), Player::White),
        ])
        .unwrap();
        let result = search(
            &board,
            &PolicyValueModel::random(8, 29),
            SearchConfig {
                simulations: 2,
                ..Default::default()
            },
        );
        assert!(result.iter().all(|candidate| {
            candidate.mv == Move::parse("c8").unwrap() || candidate.mv == Move::parse("h8").unwrap()
        }));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn open_four_replaces_network_policy() {
        let board = Board::from_stones(&[
            (Move::parse("d8").unwrap(), Player::Black),
            (Move::parse("a1").unwrap(), Player::White),
            (Move::parse("e8").unwrap(), Player::Black),
            (Move::parse("a3").unwrap(), Player::White),
            (Move::parse("g8").unwrap(), Player::Black),
            (Move::parse("a5").unwrap(), Player::White),
        ])
        .unwrap();
        let result = search(
            &board,
            &PolicyValueModel::random(8, 31),
            SearchConfig {
                simulations: 2,
                ..Default::default()
            },
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].mv, Move::parse("f8").unwrap());
    }

    #[test]
    fn root_noise_is_normalized_and_seeded() {
        let original = vec![0.1, 0.2, 0.3, 0.4];
        let mut first = original.clone();
        let mut same = original.clone();
        let mut other = original;
        apply_root_dirichlet_noise(&mut first, 0.12, 0.25, 7);
        apply_root_dirichlet_noise(&mut same, 0.12, 0.25, 7);
        apply_root_dirichlet_noise(&mut other, 0.12, 0.25, 8);
        assert_eq!(first, same);
        assert_ne!(first, other);
        assert!((first.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn standard_search_uses_exact_budget_without_forced_root_coverage() {
        let mut board = Board::new();
        assert!(board.play(Move::parse("h8").unwrap()));
        let model = PolicyValueModel::random(8, 17);

        let one = search(
            &board,
            &model,
            SearchConfig {
                simulations: 1,
                ..Default::default()
            },
        );
        assert_eq!(one.iter().map(|candidate| candidate.visits).sum::<u32>(), 1);
        assert_eq!(
            one.iter().filter(|candidate| candidate.visits > 0).count(),
            1
        );
        assert_eq!(one.len(), 224);

        let full = search(
            &board,
            &model,
            SearchConfig {
                simulations: 128,
                ..Default::default()
            },
        );
        assert_eq!(
            full.iter().map(|candidate| candidate.visits).sum::<u32>(),
            128
        );
    }
}
