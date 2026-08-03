use crate::{
    game::{BOARD_SIZE, Board, Move, Outcome},
    mcts::{SearchConfig, search, search_with_info},
    model::PolicyValueModel,
    replay::Sample,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

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
    pub black_win_plies: usize,
    pub white_win_plies: usize,
    pub draw_plies: usize,
    pub random_opening_games: usize,
    pub random_opening_plies: usize,
    pub searches: usize,
    pub simulations: usize,
    pub entropy_sum: f32,
    pub visited_actions_sum: usize,
    pub policy_top1_sum: f32,
    pub policy_top2_sum: f32,
    pub sampled_moves: usize,
    pub sampled_best_moves: usize,
    pub sampled_q_gap_sum: f32,
    pub policy_surprise_sum: f32,
    pub value_surprise_sum: f32,
}

impl SelfplayStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.games += other.games;
        self.black_wins += other.black_wins;
        self.white_wins += other.white_wins;
        self.draws += other.draws;
        self.plies += other.plies;
        self.black_win_plies += other.black_win_plies;
        self.white_win_plies += other.white_win_plies;
        self.draw_plies += other.draw_plies;
        self.random_opening_games += other.random_opening_games;
        self.random_opening_plies += other.random_opening_plies;
        self.searches += other.searches;
        self.simulations += other.simulations;
        self.entropy_sum += other.entropy_sum;
        self.visited_actions_sum += other.visited_actions_sum;
        self.policy_top1_sum += other.policy_top1_sum;
        self.policy_top2_sum += other.policy_top2_sum;
        self.sampled_moves += other.sampled_moves;
        self.sampled_best_moves += other.sampled_best_moves;
        self.sampled_q_gap_sum += other.sampled_q_gap_sum;
        self.policy_surprise_sum += other.policy_surprise_sum;
        self.value_surprise_sum += other.value_surprise_sum;
    }
}

pub struct GeneratedGame {
    pub samples: Vec<Sample>,
    pub stats: SelfplayStats,
}

pub fn generate_one_detailed(
    model: &PolicyValueModel,
    cfg: SearchConfig,
    seed: u64,
) -> GeneratedGame {
    generate_one_detailed_controlled(model, cfg, seed, None)
}

pub fn generate_one_detailed_controlled(
    model: &PolicyValueModel,
    cfg: SearchConfig,
    mut seed: u64,
    stop: Option<&AtomicBool>,
) -> GeneratedGame {
    crate::scope_profile!("selfplay.game");
    let mut board = Board::new();
    let mut samples = vec![];
    let mut stats = SelfplayStats {
        games: 1,
        ..Default::default()
    };
    let opening_plies = apply_policy_opening(
        &mut board,
        model,
        cfg.random_opening_probability,
        cfg.policy_opening_avg_plies,
        cfg.policy_opening_temperature,
        &mut seed,
    );
    stats.random_opening_games = usize::from(opening_plies > 0);
    stats.random_opening_plies = opening_plies;
    let asymmetric = random_unit(&mut seed) < cfg.asymmetric_playout_prob.clamp(0.0, 1.0);
    let strong_player = if random_index(&mut seed, 2) == 0 {
        crate::game::Player::Black
    } else {
        crate::game::Player::White
    };
    let asymmetric_ratio = if asymmetric {
        1.0 + random_unit(&mut seed) * (cfg.max_asymmetric_ratio.max(1.0) - 1.0)
    } else {
        1.0
    };
    let fork_ply = (random_unit(&mut seed) < cfg.early_fork_game_prob.clamp(0.0, 1.0)
        && cfg.early_fork_max_ply > 0)
        .then(|| random_index(&mut seed, cfg.early_fork_max_ply));
    while board.outcome().is_none() && !stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        let mut ply_cfg = cfg;
        if asymmetric && board.to_move() != strong_player {
            ply_cfg.simulations = ((cfg.simulations as f32 / asymmetric_ratio).round() as usize)
                .clamp(1, cfg.simulations);
        }
        ply_cfg.root_noise_seed = seed ^ board.move_count() as u64;
        let result = {
            crate::scope_profile!("selfplay.search");
            search_with_info(&board, model, ply_cfg)
        };
        let c = result.candidates;
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
        let mv = if fork_ply == Some(board.move_count().saturating_sub(opening_plies))
            && cfg.early_fork_max_choices > 1
        {
            sample_fork_move(&c, cfg.early_fork_max_choices, &mut seed)
        } else {
            sample_with_temperature(
                &c,
                temperature,
                cfg.temperature_value_cutoff,
                cfg.temperature_visit_offset,
                &mut seed,
            )
        };
        if temperature > 1e-6 {
            stats.sampled_moves += 1;
            stats.sampled_best_moves += usize::from(mv == c[0].mv);
            let played_q = c.iter().find(|x| x.mv == mv).map(|x| x.q).unwrap_or(0.0);
            stats.sampled_q_gap_sum += (c[0].q - played_q).max(0.0);
        }
        let policy_surprise = policy
            .iter()
            .map(|&(mv, target)| {
                let raw = c
                    .iter()
                    .find(|candidate| candidate.mv == mv)
                    .map(|candidate| candidate.raw_prior)
                    .unwrap_or(0.0)
                    .max(1e-12);
                if target > 0.0 {
                    target * (target / raw).ln()
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            .max(0.0);
        stats.policy_surprise_sum += policy_surprise;
        samples.push(Sample {
            board: board.clone(),
            policy,
            value: 0.0,
            value_wdl: None,
            generation: 0,
            policy_weight: ply_cfg.simulations as f32 / cfg.simulations.max(1) as f32,
            value_weight: 1.0,
            policy_surprise,
            value_surprise: 0.0,
            predicted_value: result.root_value,
        });
        board.play(mv);
    }
    let out = board.outcome();
    stats.plies = board.move_count();
    match out {
        Some(Outcome::Win(crate::game::Player::Black)) => {
            stats.black_wins = 1;
            stats.black_win_plies = stats.plies;
        }
        Some(Outcome::Win(crate::game::Player::White)) => {
            stats.white_wins = 1;
            stats.white_win_plies = stats.plies;
        }
        _ => {
            stats.draws = 1;
            stats.draw_plies = stats.plies;
        }
    }
    for s in &mut samples {
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
        s.value_surprise = (s.value - s.predicted_value).abs();
        stats.value_surprise_sum += s.value_surprise;
    }
    GeneratedGame { samples, stats }
}

fn apply_policy_opening(
    board: &mut Board,
    model: &PolicyValueModel,
    probability: f32,
    avg_plies: usize,
    temperature: f32,
    seed: &mut u64,
) -> usize {
    if board.move_count() != 0
        || avg_plies == 0
        || probability <= 0.0
        || random_unit(seed) >= probability.min(1.0)
    {
        return 0;
    }
    let max_plies = avg_plies.saturating_mul(3).max(1);
    let mut played = 0;
    for _ in 0..max_plies {
        if board.outcome().is_some() {
            break;
        }
        let (policy, _) = model.evaluate(board);
        if policy.is_empty() {
            break;
        }
        let inverse_temperature = temperature.max(1e-3).recip();
        let weights = policy
            .iter()
            .map(|(_, probability)| probability.max(1e-12).powf(inverse_temperature))
            .collect::<Vec<_>>();
        let mv = sample_weighted_moves(&policy, &weights, seed);
        if !board.play(mv) {
            break;
        }
        played += 1;
        if random_unit(seed) < 1.0 / avg_plies as f32 {
            break;
        }
    }
    played
}

fn sample_fork_move(
    candidates: &[crate::mcts::Candidate],
    max_choices: usize,
    seed: &mut u64,
) -> Move {
    let mut choices = candidates
        .iter()
        .filter(|candidate| candidate.visits > 0)
        .take(max_choices.max(2))
        .collect::<Vec<_>>();
    if choices.len() <= 1 {
        return candidates[0].mv;
    }
    choices.remove(0);
    let weights = choices
        .iter()
        .map(|candidate| candidate.visits.max(1) as f32)
        .collect::<Vec<_>>();
    let mut draw = random_unit(seed) * weights.iter().sum::<f32>();
    for (candidate, weight) in choices.into_iter().zip(weights) {
        draw -= weight;
        if draw <= 0.0 {
            return candidate.mv;
        }
    }
    candidates[0].mv
}

fn sample_weighted_moves(policy: &[(Move, f32)], weights: &[f32], seed: &mut u64) -> Move {
    let mut draw = random_unit(seed) * weights.iter().sum::<f32>();
    for (&(mv, _), &weight) in policy.iter().zip(weights) {
        draw -= weight;
        if draw <= 0.0 {
            return mv;
        }
    }
    policy[0].0
}

fn apply_region_opening(board: &mut Board, plies: usize, seed: &mut u64) -> usize {
    const REGION_SIZES: [usize; 3] = [3, 4, 5];
    if board.move_count() != 0 || plies == 0 {
        return 0;
    }

    let size = REGION_SIZES[random_index(seed, REGION_SIZES.len())];
    let row_start = random_index(seed, BOARD_SIZE - size + 1);
    let col_start = random_index(seed, BOARD_SIZE - size + 1);
    let mut available = Vec::with_capacity(size * size);
    for row in row_start..row_start + size {
        for col in col_start..col_start + size {
            available.push(Move::new(row, col).expect("随机开局区域必须位于棋盘内"));
        }
    }

    let mut played = 0;
    for _ in 0..plies {
        if available.is_empty() {
            break;
        }
        let mv = available.swap_remove(random_index(seed, available.len()));
        if board.is_legal(mv) {
            board.play(mv);
            played += 1;
        }
    }
    played
}

fn random_unit(seed: &mut u64) -> f32 {
    const BUCKETS: usize = 1 << 24;
    random_index(seed, BUCKETS) as f32 / BUCKETS as f32
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
    pub policy_entropy: f32,
    pub value_entropy: f32,
    pub policy_kl: f32,
    pub value_kl: f32,
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
    pub paired_openings: usize,
    pub paired_score_square_sum: f32,
    pub plies: usize,
    pub win_plies: usize,
    pub loss_plies: usize,
    pub draw_plies: usize,
}

impl ArenaReport {
    pub fn games(self) -> usize {
        self.wins + self.losses + self.draws
    }
    pub fn score_rate(self) -> f32 {
        (self.wins as f32 + self.draws as f32 * 0.5) / self.games().max(1) as f32
    }
    pub fn score_rate_standard_error(self) -> f32 {
        if self.paired_openings > 1 && self.paired_openings * 2 == self.games() {
            let pairs = self.paired_openings as f32;
            let mean = self.score_rate();
            let sample_variance =
                ((self.paired_score_square_sum - pairs * mean * mean) / (pairs - 1.0)).max(0.0);
            return (sample_variance / pairs).sqrt();
        }
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
    arena_controlled(candidate, baseline, games, cfg, None)
}

pub fn arena_controlled(
    candidate: &PolicyValueModel,
    baseline: &PolicyValueModel,
    games: usize,
    cfg: SearchConfig,
    stop: Option<&AtomicBool>,
) -> ArenaReport {
    (0..games.div_ceil(2))
        .into_par_iter()
        .map(|opening_index| {
            if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return ArenaReport::default();
            }
            let mut board = Board::new();
            let mut opening_seed =
                cfg.opening_seed ^ (opening_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            apply_region_opening(&mut board, cfg.opening_random_plies, &mut opening_seed);
            let mut report = play_arena_game(board.clone(), true, candidate, baseline, cfg, stop);
            if opening_index * 2 + 1 < games {
                let second = play_arena_game(board, false, candidate, baseline, cfg, stop);
                report = merge_arena_reports(report, second);
                if report.games() == 2 {
                    let pair_score = report.score_rate();
                    report.paired_openings = 1;
                    report.paired_score_square_sum = pair_score * pair_score;
                }
            }
            report
        })
        .reduce(ArenaReport::default, merge_arena_reports)
}

fn play_arena_game(
    mut board: Board,
    candidate_black: bool,
    candidate: &PolicyValueModel,
    baseline: &PolicyValueModel,
    cfg: SearchConfig,
    stop: Option<&AtomicBool>,
) -> ArenaReport {
    while board.outcome().is_none() && !stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        let candidate_turn = (board.to_move() == crate::game::Player::Black) == candidate_black;
        let model = if candidate_turn { candidate } else { baseline };
        let result = search(&board, model, cfg);
        if result.is_empty() {
            break;
        }
        board.play(result[0].mv);
    }
    let plies = board.move_count();
    match board.outcome() {
        Some(Outcome::Win(player)) => {
            let candidate_won = (player == crate::game::Player::Black) == candidate_black;
            if candidate_won {
                ArenaReport {
                    wins: 1,
                    wins_as_black: usize::from(candidate_black),
                    wins_as_white: usize::from(!candidate_black),
                    plies,
                    win_plies: plies,
                    ..Default::default()
                }
            } else {
                ArenaReport {
                    losses: 1,
                    losses_as_black: usize::from(candidate_black),
                    losses_as_white: usize::from(!candidate_black),
                    plies,
                    loss_plies: plies,
                    ..Default::default()
                }
            }
        }
        _ => ArenaReport {
            draws: 1,
            draws_as_black: usize::from(candidate_black),
            draws_as_white: usize::from(!candidate_black),
            plies,
            draw_plies: plies,
            ..Default::default()
        },
    }
}

fn merge_arena_reports(a: ArenaReport, b: ArenaReport) -> ArenaReport {
    ArenaReport {
        wins: a.wins + b.wins,
        losses: a.losses + b.losses,
        draws: a.draws + b.draws,
        wins_as_black: a.wins_as_black + b.wins_as_black,
        losses_as_black: a.losses_as_black + b.losses_as_black,
        draws_as_black: a.draws_as_black + b.draws_as_black,
        wins_as_white: a.wins_as_white + b.wins_as_white,
        losses_as_white: a.losses_as_white + b.losses_as_white,
        draws_as_white: a.draws_as_white + b.draws_as_white,
        paired_openings: a.paired_openings + b.paired_openings,
        paired_score_square_sum: a.paired_score_square_sum + b.paired_score_square_sum,
        plies: a.plies + b.plies,
        win_plies: a.win_plies + b.win_plies,
        loss_plies: a.loss_plies + b.loss_plies,
        draw_plies: a.draw_plies + b.draw_plies,
    }
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

    #[test]
    fn arena_standard_error_uses_paired_openings() {
        let report = ArenaReport {
            wins: 3,
            losses: 3,
            paired_openings: 3,
            paired_score_square_sum: 1.25,
            ..Default::default()
        };
        let expected = (0.25_f32 / 3.0).sqrt();
        assert!((report.score_rate_standard_error() - expected).abs() < 1.0e-6);
    }

    #[test]
    fn policy_opening_can_be_disabled() {
        let mut board = Board::new();
        let mut seed = 7;
        let model = PolicyValueModel::random(8, 9);
        assert_eq!(
            apply_policy_opening(&mut board, &model, 0.0, 6, 1.6, &mut seed),
            0
        );
        assert_eq!(board.move_count(), 0);
    }

    #[test]
    fn policy_opening_is_deterministic_and_has_variable_length() {
        let model = PolicyValueModel::random(8, 9);
        let mut first = Board::new();
        let mut first_seed = 7;
        let plies = apply_policy_opening(&mut first, &model, 1.0, 6, 1.6, &mut first_seed);
        assert!((1..=18).contains(&plies));
        assert_eq!(first.move_count(), plies);

        let mut second = Board::new();
        let mut second_seed = 7;
        assert_eq!(
            apply_policy_opening(&mut second, &model, 1.0, 6, 1.6, &mut second_seed),
            plies
        );
        assert_eq!(first.cells(), second.cells());
    }

    #[test]
    fn selfplay_records_finite_surprise_and_weights() {
        let model = PolicyValueModel::random(8, 21);
        let game = generate_one_detailed(
            &model,
            SearchConfig {
                simulations: 4,
                random_opening_probability: 0.0,
                root_num_symmetries_to_sample: 2,
                use_graph_search: true,
                use_lcb_for_selection: true,
                ..Default::default()
            },
            31,
        );
        assert!(!game.samples.is_empty());
        assert!(game.samples.iter().all(|sample| {
            sample.policy_weight > 0.0
                && sample.value_weight > 0.0
                && sample.policy_surprise.is_finite()
                && sample.value_surprise.is_finite()
        }));
    }
}
