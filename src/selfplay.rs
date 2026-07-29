use crate::{
    game::{Board, Outcome},
    mcts::{SearchConfig, search},
    model::PolicyValueModel,
    replay::Sample,
};
use rayon::prelude::*;

pub fn generate(model: &PolicyValueModel, games: usize, cfg: SearchConfig) -> Vec<Sample> {
    (0..games)
        .into_par_iter()
        .flat_map_iter(|game| generate_one_detailed(model, cfg, game as u64).samples)
        .collect()
}

pub fn generate_one(model: &PolicyValueModel, cfg: SearchConfig, seed: u64) -> Vec<Sample> {
    generate_one_detailed(model, cfg, seed).samples
}

#[derive(Clone, Debug, Default)]
pub struct SelfplayStats {
    pub games: usize,
    pub black_wins: usize,
    pub white_wins: usize,
    pub draws: usize,
    pub plies: usize,
    pub searches: usize,
    pub simulations: usize,
    pub entropy_sum: f32,
    pub visited_actions_sum: usize,
    pub policy_top1_sum: f32,
    pub policy_top2_sum: f32,
    pub sampled_moves: usize,
    pub sampled_best_moves: usize,
    pub sampled_q_gap_sum: f32,
}

impl SelfplayStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.games += other.games;
        self.black_wins += other.black_wins;
        self.white_wins += other.white_wins;
        self.draws += other.draws;
        self.plies += other.plies;
        self.searches += other.searches;
        self.simulations += other.simulations;
        self.entropy_sum += other.entropy_sum;
        self.visited_actions_sum += other.visited_actions_sum;
        self.policy_top1_sum += other.policy_top1_sum;
        self.policy_top2_sum += other.policy_top2_sum;
        self.sampled_moves += other.sampled_moves;
        self.sampled_best_moves += other.sampled_best_moves;
        self.sampled_q_gap_sum += other.sampled_q_gap_sum;
    }
}

pub struct GeneratedGame {
    pub samples: Vec<Sample>,
    pub stats: SelfplayStats,
}

pub fn generate_one_detailed(
    model: &PolicyValueModel,
    cfg: SearchConfig,
    mut seed: u64,
) -> GeneratedGame {
    let mut board = Board::new();
    let mut samples = vec![];
    let mut stats = SelfplayStats {
        games: 1,
        ..Default::default()
    };
    while board.outcome().is_none() {
        let mut ply_cfg = cfg;
        ply_cfg.root_noise_seed = seed ^ board.move_count() as u64;
        let c = search(&board, model, ply_cfg);
        if c.is_empty() {
            break;
        }
        let sum = c.iter().map(|x| x.visits).sum::<u32>().max(1) as f32;
        let policy: Vec<_> = c.iter().map(|x| (x.mv, x.visits as f32 / sum)).collect();
        let mut top = [0.0_f32; 2];
        for &(_, p) in &policy {
            if p > 0.0 {
                stats.entropy_sum -= p * p.ln();
            }
            if p > top[0] {
                top[1] = top[0];
                top[0] = p;
            } else if p > top[1] {
                top[1] = p;
            }
        }
        stats.searches += 1;
        stats.simulations += c.iter().map(|x| x.visits as usize).sum::<usize>();
        stats.visited_actions_sum += c.iter().filter(|x| x.visits > 0).count();
        stats.policy_top1_sum += top[0];
        stats.policy_top2_sum += top[0] + top[1];
        let temperature = temperature_for_ply(cfg, board.move_count());
        let mv = sample_with_temperature(
            &c,
            temperature,
            cfg.temperature_value_cutoff,
            cfg.temperature_visit_offset,
            &mut seed,
        );
        if temperature > 1e-6 {
            stats.sampled_moves += 1;
            stats.sampled_best_moves += usize::from(mv == c[0].mv);
            let played_q = c.iter().find(|x| x.mv == mv).map(|x| x.q).unwrap_or(0.0);
            stats.sampled_q_gap_sum += (c[0].q - played_q).max(0.0);
        }
        samples.push(Sample {
            board: board.clone(),
            policy,
            value: 0.0,
            moves_left: 0.0,
            generation: 0,
        });
        board.play(mv);
    }
    let out = board.outcome();
    stats.plies = samples.len();
    match out {
        Some(Outcome::Win(crate::game::Player::Black)) => stats.black_wins = 1,
        Some(Outcome::Win(crate::game::Player::White)) => stats.white_wins = 1,
        _ => stats.draws = 1,
    }
    let game_len = samples.len();
    for (index, s) in samples.iter_mut().enumerate() {
        s.value = match out {
            Some(Outcome::Draw) | None => 0.0,
            Some(Outcome::Win(p)) => {
                if p == s.board.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        s.moves_left = game_len.saturating_sub(index).max(1) as f32;
    }
    GeneratedGame { samples, stats }
}
fn temperature_for_ply(cfg: SearchConfig, ply: usize) -> f32 {
    if ply < cfg.temperature_decay_delay_plies {
        return cfg.temperature_start;
    }
    if cfg.temperature_decay_plies == 0 {
        return cfg.temperature_endgame;
    }
    let decay_ply = ply.saturating_sub(cfg.temperature_decay_delay_plies);
    if decay_ply >= cfg.temperature_decay_plies {
        return cfg.temperature_endgame;
    }
    let progress = decay_ply as f32 / cfg.temperature_decay_plies as f32;
    cfg.temperature_start + (cfg.temperature_endgame - cfg.temperature_start) * progress
}

fn sample_with_temperature(
    c: &[crate::mcts::Candidate],
    temperature: f32,
    value_cutoff: f32,
    visit_offset: f32,
    seed: &mut u64,
) -> crate::game::Move {
    if temperature <= 1e-6 {
        return c[0].mv;
    }
    let anchor_q = c
        .iter()
        .max_by(|a, b| {
            (a.visits as f32 + visit_offset).total_cmp(&(b.visits as f32 + visit_offset))
        })
        .map(|x| x.q)
        .unwrap_or(0.0);
    let min_q = anchor_q - 2.0 * value_cutoff;
    let weights = c
        .iter()
        .map(|x| {
            if value_cutoff > 0.0 && value_cutoff < 1.0 && x.q < min_q {
                0.0
            } else {
                (x.visits as f32 + visit_offset)
                    .max(1e-9)
                    .powf(temperature.max(1e-3).recip())
            }
        })
        .collect::<Vec<_>>();
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    let mut x = ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * weights.iter().sum::<f32>() as f64;
    for (a, weight) in c.iter().zip(weights) {
        x -= weight as f64;
        if x <= 0.0 {
            return a.mv;
        }
    }
    c[0].mv
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainStats {
    pub samples: usize,
    pub optimizer_steps: usize,
    pub loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub moves_left_loss: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaReport {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub wins_as_black: usize,
    pub losses_as_black: usize,
    pub draws_as_black: usize,
    pub wins_as_white: usize,
    pub losses_as_white: usize,
    pub draws_as_white: usize,
}

impl ArenaReport {
    pub fn games(self) -> usize {
        self.wins + self.losses + self.draws
    }
    pub fn score_rate(self) -> f32 {
        (self.wins as f32 + self.draws as f32 * 0.5) / self.games().max(1) as f32
    }
    pub fn score_rate_standard_error(self) -> f32 {
        let games = self.games();
        if games <= 1 {
            return 0.5;
        }
        let mean = self.score_rate();
        let mean_square = (self.wins as f32 + self.draws as f32 * 0.25) / games as f32;
        ((mean_square - mean * mean).max(0.0) / games as f32).sqrt()
    }
    pub fn score_rate_lower_bound(self, z: f32) -> f32 {
        self.score_rate() - z.max(0.0) * self.score_rate_standard_error()
    }
    pub fn elo_diff(self) -> f32 {
        let score = self.score_rate();
        if score <= 0.0 {
            -400.0
        } else if score >= 1.0 {
            400.0
        } else {
            400.0 * (score / (1.0 - score)).log10()
        }
    }
}

pub fn arena(
    candidate: &PolicyValueModel,
    baseline: &PolicyValueModel,
    games: usize,
    cfg: SearchConfig,
) -> ArenaReport {
    (0..games)
        .into_par_iter()
        .map(|game| {
            let candidate_black = game % 2 == 0;
            let mut board = Board::new();
            let opening_index = game / 2;
            let mut opening_seed =
                cfg.opening_seed ^ (opening_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..cfg.opening_random_plies {
                let legal = board.legal_moves();
                if legal.is_empty() {
                    break;
                }
                let index = random_index(&mut opening_seed, legal.len());
                board.play(legal[index]);
            }
            while board.outcome().is_none() {
                let candidate_turn =
                    (board.to_move() == crate::game::Player::Black) == candidate_black;
                let model = if candidate_turn { candidate } else { baseline };
                let result = search(&board, model, cfg);
                if result.is_empty() {
                    break;
                }
                board.play(result[0].mv);
            }
            match board.outcome() {
                Some(Outcome::Win(player)) => {
                    let candidate_won = (player == crate::game::Player::Black) == candidate_black;
                    if candidate_won {
                        ArenaReport {
                            wins: 1,
                            wins_as_black: usize::from(candidate_black),
                            wins_as_white: usize::from(!candidate_black),
                            ..Default::default()
                        }
                    } else {
                        ArenaReport {
                            losses: 1,
                            losses_as_black: usize::from(candidate_black),
                            losses_as_white: usize::from(!candidate_black),
                            ..Default::default()
                        }
                    }
                }
                _ => ArenaReport {
                    draws: 1,
                    draws_as_black: usize::from(candidate_black),
                    draws_as_white: usize::from(!candidate_black),
                    ..Default::default()
                },
            }
        })
        .reduce(ArenaReport::default, |a, b| ArenaReport {
            wins: a.wins + b.wins,
            losses: a.losses + b.losses,
            draws: a.draws + b.draws,
            wins_as_black: a.wins_as_black + b.wins_as_black,
            losses_as_black: a.losses_as_black + b.losses_as_black,
            draws_as_black: a.draws_as_black + b.draws_as_black,
            wins_as_white: a.wins_as_white + b.wins_as_white,
            losses_as_white: a.losses_as_white + b.losses_as_white,
            draws_as_white: a.draws_as_white + b.draws_as_white,
        })
}

fn random_index(seed: &mut u64, len: usize) -> usize {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as usize) % len.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_decays_linearly() {
        let cfg = SearchConfig {
            temperature_start: 0.9,
            temperature_endgame: 0.3,
            temperature_decay_delay_plies: 10,
            temperature_decay_plies: 20,
            ..Default::default()
        };
        assert_eq!(temperature_for_ply(cfg, 9), 0.9);
        assert!((temperature_for_ply(cfg, 20) - 0.6).abs() < 1e-6);
        assert_eq!(temperature_for_ply(cfg, 30), 0.3);
    }
}
