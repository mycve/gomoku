use crate::{
    game::{Board, Move, Outcome},
    model::{EvalAccumulator, EvalScratch, PolicyValueModel},
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub root_noise_seed: u64,
    pub policy_softmax_temp: f32,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub opening_random_plies: usize,
    pub opening_seed: u64,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 3000,
            cpuct: 1.5,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            root_noise_seed: 0,
            policy_softmax_temp: 1.0,
            temperature_start: 0.0,
            temperature_endgame: 0.0,
            temperature_decay_delay_plies: 0,
            temperature_decay_plies: 0,
            temperature_value_cutoff: 0.0,
            temperature_visit_offset: 0.0,
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
}
struct Node {
    board: Board,
    accumulator: EvalAccumulator,
    children: Vec<Edge>,
    expanded: bool,
}
struct Edge {
    mv: Move,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

pub fn search(board: &Board, model: &PolicyValueModel, cfg: SearchConfig) -> Vec<Candidate> {
    search_until(board, model, cfg, None)
}

pub fn search_timed(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    time_limit: Duration,
) -> Vec<Candidate> {
    search_until(board, model, cfg, Some(Instant::now() + time_limit))
}

fn search_until(
    board: &Board,
    model: &PolicyValueModel,
    cfg: SearchConfig,
    deadline: Option<Instant>,
) -> Vec<Candidate> {
    crate::scope_profile!("mcts.search");
    let mut scratch = EvalScratch::new(model.hidden_size);
    let mut nodes = vec![Node {
        board: board.clone(),
        accumulator: model.accumulator(board),
        children: vec![],
        expanded: false,
    }];
    expand(&mut nodes, 0, model, cfg, &mut scratch);
    for simulation in 0..cfg.simulations {
        if simulation > 0 && deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        simulate(&mut nodes, 0, model, cfg, &mut scratch);
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
        })
        .collect();
    out.sort_by_key(|x| std::cmp::Reverse(x.visits));
    out
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
    let (mut policy, value) = {
        crate::scope_profile!("mcts.nn_eval");
        model.evaluate_accumulator_with_scratch(
            &nodes[idx].board,
            &nodes[idx].accumulator,
            cfg.policy_softmax_temp,
            scratch,
        )
    };
    if idx == 0 && cfg.root_dirichlet_alpha > 0.0 && cfg.root_exploration_fraction > 0.0 {
        let mut priors = policy.iter().map(|(_, prior)| *prior).collect::<Vec<_>>();
        apply_root_dirichlet_noise(
            &mut priors,
            cfg.root_dirichlet_alpha,
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
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
    }
    nodes[idx].expanded = true;
    value
}
fn simulate(
    nodes: &mut Vec<Node>,
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
        let exploration = cfg.cpuct * total.sqrt();
        for (i, e) in nodes[idx].children.iter().enumerate() {
            let q = if e.visits == 0 {
                0.0
            } else {
                e.value_sum / e.visits as f32
            };
            let s = q + exploration * e.prior / (1.0 + e.visits as f32);
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
        let c = nodes.len();
        nodes.push(Node {
            board: b,
            accumulator,
            children: vec![],
            expanded: false,
        });
        nodes[idx].children[best].child = Some(c);
        c
    };
    let value = -simulate(nodes, child, model, cfg, scratch);
    let e = &mut nodes[idx].children[best];
    e.visits += 1;
    e.value_sum += value;
    value
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
}
