use crate::{
    async_selfplay::{AsyncSelfplay, SelfplayGame},
    az_loop_config::AzLoopConfig,
    candle_train,
    mcts::SearchConfig,
    model::PolicyValueModel,
    replay,
    selfplay::{SelfplayStats, arena_controlled},
};
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Progress {
    update: usize,
    total_games: usize,
    total_samples: usize,
    optimizer_steps: usize,
    learning_rate: f32,
}

#[derive(Default)]
struct PendingBatch {
    samples: Vec<crate::replay::Sample>,
    stats: SelfplayStats,
    games: usize,
    oldest_version: u64,
    newest_version: u64,
    workers: std::collections::HashSet<usize>,
    collect_seconds: f32,
}

struct TrainerEvent {
    model: PolicyValueModel,
    online_model: PolicyValueModel,
    batch: PendingBatch,
    train_stats: crate::selfplay::TrainStats,
    sampling_seconds: f32,
    train_seconds: f32,
    pool_samples: usize,
    learning_rate: f32,
    train_samples: usize,
    recent_quota_rate: f32,
    actual_recent_rate: f32,
}

pub fn run(config: AzLoopConfig, target_update: Option<usize>) -> io::Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    let interrupt_signal = Arc::clone(&interrupted);
    ctrlc::set_handler(move || {
        interrupt_signal.store(true, Ordering::SeqCst);
        signal.store(true, Ordering::SeqCst);
    })
    .map_err(io::Error::other)?;
    let mut progress = load_progress(&config.progress_path)?;
    let initial_model = load_or_init(&config.model_path)?;
    let ema_checkpoint_exists = Path::new(&config.ema_model_path).exists();
    let initial_ema = if ema_checkpoint_exists {
        PolicyValueModel::load(&config.ema_model_path)?
    } else {
        initial_model.clone()
    };
    let mut best = if Path::new(&config.best_model_path).exists() {
        PolicyValueModel::load(&config.best_model_path)?
    } else {
        initial_ema.save(&config.best_model_path)?;
        initial_ema.clone()
    };
    let initial_pool = if Path::new(&config.replay_path).exists() {
        let pool = replay::load(&config.replay_path)?;
        fs::remove_file(&config.replay_path)?;
        println!(
            "replay   : restored {}/{} samples from {} (file removed)",
            pool.len(),
            config.replay_capacity,
            config.replay_path
        );
        pool
    } else {
        Vec::new()
    };
    let initial_pool_samples = initial_pool.len();
    let max_workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let requested_workers = if config.selfplay_workers == 0 {
        max_workers
    } else {
        config.selfplay_workers.max(1)
    };
    let workers = requested_workers.min(max_workers);
    if requested_workers > max_workers {
        println!(
            "workers  : requested={} capped={} available_cores={}",
            requested_workers, workers, max_workers
        );
    }
    let queue_capacity = if config.selfplay_queue_capacity == 0 {
        workers.saturating_mul(8).max(32)
    } else {
        config.selfplay_queue_capacity.max(1)
    };
    println!(
        "explore  : temp={:.2}->{:.2} delay={} decay={} value_cutoff={:.2} visit_offset={:.2}",
        config.temperature_start,
        config.temperature_endgame,
        config.temperature_decay_delay_plies,
        config.temperature_decay_plies,
        config.temperature_value_cutoff,
        config.temperature_visit_offset
    );
    println!(
        "priors   : softmax_temp={:.2} root_noise(total_concentration={:.2}, fraction={:.2})",
        config.policy_softmax_temp,
        config.root_dirichlet_total_concentration,
        config.root_exploration_fraction
    );
    println!(
        "opening  : balanced={:.1}% policy={:.1}% avg_plies={} temperature={:.2} samples_after_opening=true",
        config.selfplay_balanced_opening_probability * 100.0,
        config.selfplay_policy_opening_probability * 100.0,
        config.selfplay_policy_opening_avg_plies,
        config.selfplay_policy_opening_temperature
    );
    println!(
        "search   : sims={} cpuct={:.2}+{:.2}log/base{:.0} desired={:.1} graph={} lcb={} symmetries={}",
        config.simulations,
        config.cpuct,
        config.cpuct_log,
        config.cpuct_base,
        config.root_desired_per_child_visits_coeff,
        config.use_graph_search,
        config.use_lcb_for_selection,
        config.root_num_symmetries_to_sample
    );
    println!(
        "targets  : policy=mcts_visits value=terminal_wdl ema_decay={:.6} ema_model={} ema_state={}",
        config.ema_decay,
        config.ema_model_path,
        if ema_checkpoint_exists {
            "restored"
        } else {
            "copy_first_update"
        }
    );
    println!(
        "lr       : warmup={} cosine={} min={:.6} peak={:.6} resumed_steps={}",
        config.learning_rate_warmup_steps,
        config.learning_rate_cosine_steps,
        config.learning_rate_min,
        config.learning_rate,
        progress.optimizer_steps
    );
    let published = Arc::new(RwLock::new(initial_ema.clone()));
    let version = Arc::new(AtomicU64::new(progress.update as u64));
    let mut actors = AsyncSelfplay::start(
        Arc::clone(&published),
        Arc::clone(&version),
        Arc::clone(&stop),
        workers,
        queue_capacity,
        SearchConfig {
            simulations: config.simulations,
            cpuct: config.cpuct,
            cpuct_log: config.cpuct_log,
            cpuct_base: config.cpuct_base,
            root_desired_per_child_visits_coeff: config.root_desired_per_child_visits_coeff,
            root_dirichlet_total_concentration: config.root_dirichlet_total_concentration,
            root_exploration_fraction: config.root_exploration_fraction,
            policy_softmax_temp: config.policy_softmax_temp,
            root_policy_temperature_early: config.root_policy_temperature_early,
            root_policy_temperature: config.root_policy_temperature,
            root_policy_temperature_halflife: config.root_policy_temperature_halflife,
            root_num_symmetries_to_sample: config.root_num_symmetries_to_sample,
            use_graph_search: config.use_graph_search,
            graph_search_max_nodes: config.graph_search_max_nodes,
            use_lcb_for_selection: config.use_lcb_for_selection,
            lcb_stdevs: config.lcb_stdevs,
            min_visit_prop_for_lcb: config.min_visit_prop_for_lcb,
            temperature_start: config.temperature_start,
            temperature_endgame: config.temperature_endgame,
            temperature_decay_delay_plies: config.temperature_decay_delay_plies,
            temperature_decay_plies: config.temperature_decay_plies,
            temperature_value_cutoff: config.temperature_value_cutoff,
            temperature_visit_offset: config.temperature_visit_offset,
            balanced_opening_probability: config.selfplay_balanced_opening_probability,
            policy_opening_probability: config.selfplay_policy_opening_probability,
            policy_opening_avg_plies: config.selfplay_policy_opening_avg_plies,
            policy_opening_temperature: config.selfplay_policy_opening_temperature,
            early_fork_game_prob: config.early_fork_game_prob,
            early_fork_max_ply: config.early_fork_max_ply,
            early_fork_max_choices: config.early_fork_max_choices,
            asymmetric_playout_prob: config.asymmetric_playout_prob,
            max_asymmetric_ratio: config.max_asymmetric_ratio,
            ..Default::default()
        },
        config.seed,
    );
    let actor_rx = actors.take_receiver();
    let backlog = actors.backlog_counter();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<PendingBatch>(1);
    let collector_stop = Arc::clone(&stop);
    let collector_version = Arc::clone(&version);
    let selfplay_samples_per_update = config.selfplay_samples_per_update.max(1);
    let mut collector_sample_target = config
        .replay_warmup_samples
        .saturating_sub(initial_pool_samples)
        .max(selfplay_samples_per_update);
    let collector = thread::spawn(move || {
        let mut pending = PendingBatch {
            oldest_version: u64::MAX,
            ..Default::default()
        };
        let mut started = Instant::now();
        while let Ok(game) = actor_rx.recv() {
            backlog.fetch_sub(1, Ordering::Relaxed);
            let current_version = collector_version.load(Ordering::Acquire);
            if game.model_version < current_version {
                continue;
            }
            if pending.games > 0 && pending.newest_version < current_version {
                pending = PendingBatch {
                    oldest_version: u64::MAX,
                    ..Default::default()
                };
                started = Instant::now();
            }
            merge_game(&mut pending, game);
            if pending.samples.len() < collector_sample_target {
                continue;
            }
            pending.collect_seconds = started.elapsed().as_secs_f32();
            let next = PendingBatch {
                oldest_version: u64::MAX,
                ..Default::default()
            };
            match ready_tx.try_send(std::mem::replace(&mut pending, next)) {
                Ok(()) => collector_sample_target = selfplay_samples_per_update,
                Err(mpsc::TrySendError::Full(_)) => {}
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
            started = Instant::now();
            if collector_stop.load(Ordering::SeqCst) {
                break;
            }
        }
    });
    let (event_tx, event_rx) = mpsc::sync_channel::<TrainerEvent>(0);
    let (trainer_error_tx, trainer_error_rx) = mpsc::sync_channel::<String>(1);
    let (trainer_ack_tx, trainer_ack_rx) = mpsc::sync_channel::<()>(0);
    let trainer_stop = Arc::clone(&stop);
    let trainer_interrupted = Arc::clone(&interrupted);
    let trainer_version = Arc::clone(&version);
    let trainer_config = config.clone();
    let start_update = progress.update;
    let start_optimizer_steps = progress.optimizer_steps;
    let trainer = thread::spawn(move || -> io::Result<()> {
        let result = (|| -> io::Result<()> {
            let mut model = initial_model;
            let mut ema_model = initial_ema;
            let mut ema_initialized = ema_checkpoint_exists;
            let mut training = candle_train::TrainingSession::new(
                &model,
                Some(&ema_model),
                current_lr(&trainer_config, start_optimizer_steps),
            )?;
            let mut pool = initial_pool;
            let mut index = 0usize;
            let mut optimizer_steps = start_optimizer_steps;
            while let Ok(batch) = ready_rx.recv() {
                if trainer_stop.load(Ordering::SeqCst) {
                    break;
                }
                if batch.newest_version < trainer_version.load(Ordering::Acquire) {
                    continue;
                }
                pool.extend(batch.samples.iter().cloned());
                if pool.len() > trainer_config.replay_capacity {
                    pool.drain(..pool.len() - trainer_config.replay_capacity);
                }
                let update = start_update + index + 1;
                let lr = current_lr(&trainer_config, optimizer_steps);
                let sampling_started = Instant::now();
                let sampled = replay::sample_mixed_recent(
                    &pool,
                    trainer_config.train_samples_per_update,
                    trainer_config.replay_recent_sample_fraction,
                    trainer_config.replay_recent_updates,
                    trainer_config.replay_policy_surprise_fraction,
                    trainer_config.replay_value_surprise_fraction,
                    trainer_config.seed ^ update as u64,
                );
                let sampling_seconds = sampling_started.elapsed().as_secs_f32();
                let train_samples = sampled.samples.len();
                let recent_quota_rate = sampled.recent_quota as f32 / train_samples.max(1) as f32;
                let actual_recent_rate = sampled.actual_recent as f32 / train_samples.max(1) as f32;
                let effective_ema_decay =
                    ema_decay_for_update(trainer_config.ema_decay, ema_initialized);
                let train_started = Instant::now();
                let train_stats = training.train_controlled(
                    &mut model,
                    Some(&mut ema_model),
                    &sampled.samples,
                    trainer_config.batch_epochs,
                    lr,
                    trainer_config.batch_size,
                    effective_ema_decay,
                    Some(&trainer_stop),
                )?;
                if trainer_stop.load(Ordering::SeqCst) {
                    break;
                }
                if train_stats.optimizer_steps > 0 {
                    ema_initialized = true;
                }
                optimizer_steps += train_stats.optimizer_steps;
                let train_seconds = train_started.elapsed().as_secs_f32();
                if event_tx
                    .send(TrainerEvent {
                        model: ema_model.clone(),
                        online_model: model.clone(),
                        batch,
                        train_stats,
                        sampling_seconds,
                        train_seconds,
                        pool_samples: pool.len(),
                        learning_rate: lr,
                        train_samples,
                        recent_quota_rate,
                        actual_recent_rate,
                    })
                    .is_err()
                {
                    break;
                }
                index += 1;
                if trainer_ack_rx.recv().is_err() {
                    break;
                }
            }
            if trainer_interrupted.load(Ordering::SeqCst) {
                replay::save(&trainer_config.replay_path, &pool)?;
                println!(
                    "replay   : interrupt snapshot {} ({}/{})",
                    trainer_config.replay_path,
                    pool.len(),
                    trainer_config.replay_capacity
                );
            }
            Ok(())
        })();
        if let Err(error) = &result {
            let _ = trainer_error_tx.try_send(format!("{error:#}"));
        }
        result
    });
    let train_device = candle_train::training_device_name()?;
    let end = target_update.unwrap_or(usize::MAX);
    println!(
        "loop     : mode=batch-async actors={} actor_queue={}(nonblocking-drop) collector_queue=1(nonblocking-drop) trainer_queue=rendezvous warmup={} samples/update>={} sims={} train_device={} batch={} arena_opening_plies={}",
        workers,
        queue_capacity,
        config.replay_warmup_samples,
        config.selfplay_samples_per_update,
        config.simulations,
        train_device,
        config.batch_size,
        config.arena_opening_plies
    );
    let mut tb = SummaryWriter::new(&config.tensorboard_logdir);
    'main: while progress.update < end && !stop.load(Ordering::SeqCst) {
        let event = loop {
            match event_rx.recv_timeout(Duration::from_millis(250)) {
                Ok(event) => break event,
                Err(mpsc::RecvTimeoutError::Timeout) if stop.load(Ordering::SeqCst) => break 'main,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) if stop.load(Ordering::SeqCst) => {
                    break 'main;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let detail = trainer_error_rx
                        .try_recv()
                        .unwrap_or_else(|_| "未知错误（训练线程未返回详细信息）".into());
                    return Err(io::Error::other(format!(
                        "Trainer 在线程完成更新前退出: {detail}"
                    )));
                }
            }
        };
        progress.update += 1;
        progress.total_games += event.batch.games;
        progress.total_samples += event.batch.samples.len();
        progress.optimizer_steps += event.train_stats.optimizer_steps;
        progress.learning_rate = event.learning_rate;
        *published.write().unwrap_or_else(|e| e.into_inner()) = event.model.clone();
        version.store(progress.update as u64, Ordering::Release);
        event.online_model.save(&config.model_path)?;
        event.model.save(&config.ema_model_path)?;
        save_progress(&config.progress_path, &progress)?;
        let checkpoint = if config.checkpoint_interval > 0
            && progress.update % config.checkpoint_interval == 0
        {
            let path = checkpoint_path(&config, progress.update);
            event.model.save(&path)?;
            prune_checkpoints(&config)?;
            Some(path)
        } else {
            None
        };
        print_event(
            &config,
            &progress,
            &event,
            workers,
            &train_device,
            checkpoint.as_deref(),
        );
        tb.add_scalar("train/loss", event.train_stats.loss, progress.update);
        tb.add_scalar(
            "train/policy_loss",
            event.train_stats.policy_loss,
            progress.update,
        );
        tb.add_scalar(
            "train/value_loss",
            event.train_stats.value_loss,
            progress.update,
        );
        tb.add_scalar("replay/samples", event.pool_samples as f32, progress.update);
        let selfplay_seconds = event.batch.collect_seconds.max(1.0e-6);
        tb.add_scalar(
            "selfplay/cycle_seconds",
            event.batch.collect_seconds,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/games_per_second",
            event.batch.games as f32 / selfplay_seconds,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/samples_per_second",
            event.batch.samples.len() as f32 / selfplay_seconds,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/simulations_per_second",
            event.batch.stats.simulations as f32 / selfplay_seconds,
            progress.update,
        );
        tb.add_scalar(
            "replay/train_recent_quota_rate",
            event.recent_quota_rate,
            progress.update,
        );
        tb.add_scalar(
            "replay/train_actual_recent_rate",
            event.actual_recent_rate,
            progress.update,
        );
        let games = event.batch.games.max(1) as f32;
        let searches = event.batch.stats.searches.max(1) as f32;
        let sampled_moves = event.batch.stats.sampled_moves.max(1) as f32;
        tb.add_scalar("train/learning_rate", event.learning_rate, progress.update);
        tb.add_scalar(
            "train/sampling_seconds",
            event.sampling_seconds,
            progress.update,
        );
        tb.add_scalar("train/seconds", event.train_seconds, progress.update);
        tb.add_scalar(
            "train/samples_per_second",
            event.train_stats.samples as f32 / event.train_seconds.max(1.0e-6),
            progress.update,
        );
        tb.add_scalar(
            "train/optimizer_steps",
            event.train_stats.optimizer_steps as f32,
            progress.update,
        );
        tb.add_scalar(
            "train/samples",
            event.train_stats.samples as f32,
            progress.update,
        );
        tb.add_scalar("selfplay/games", event.batch.games as f32, progress.update);
        tb.add_scalar(
            "selfplay/samples",
            event.batch.samples.len() as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/average_plies",
            event.batch.stats.plies as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/average_plies_black_win",
            event.batch.stats.black_win_plies as f32 / event.batch.stats.black_wins.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/average_plies_white_win",
            event.batch.stats.white_win_plies as f32 / event.batch.stats.white_wins.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/average_plies_draw",
            event.batch.stats.draw_plies as f32 / event.batch.stats.draws.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/random_opening_rate",
            event.batch.stats.random_opening_games as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/random_opening_plies",
            event.batch.stats.random_opening_plies as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/balanced_opening_rate",
            event.batch.stats.balanced_opening_games as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/balanced_opening_attempts",
            event.batch.stats.balanced_opening_attempts as f32
                / event.batch.stats.balanced_opening_games.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/balanced_opening_abs_value",
            event.batch.stats.balanced_opening_abs_value_sum
                / event.batch.stats.balanced_opening_games.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/black_win_rate",
            event.batch.stats.black_wins as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/white_win_rate",
            event.batch.stats.white_wins as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "selfplay/draw_rate",
            event.batch.stats.draws as f32 / games,
            progress.update,
        );
        tb.add_scalar(
            "search/average_simulations",
            event.batch.stats.simulations as f32 / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/policy_entropy",
            event.batch.stats.entropy_sum / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/policy_surprise",
            event.batch.stats.policy_surprise_sum / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/value_surprise",
            event.batch.stats.value_surprise_sum / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/visited_actions",
            event.batch.stats.visited_actions_sum as f32 / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/policy_top1",
            event.batch.stats.policy_top1_sum / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/policy_top2",
            event.batch.stats.policy_top2_sum / searches,
            progress.update,
        );
        tb.add_scalar(
            "search/temperature_best_move_rate",
            event.batch.stats.sampled_best_moves as f32 / sampled_moves,
            progress.update,
        );
        tb.add_scalar(
            "search/temperature_q_gap",
            event.batch.stats.sampled_q_gap_sum / sampled_moves,
            progress.update,
        );
        tb.add_scalar(
            "pipeline/active_workers",
            event.batch.workers.len() as f32,
            progress.update,
        );
        tb.add_scalar(
            "pipeline/active_worker_rate",
            event.batch.workers.len() as f32 / workers.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "pipeline/actor_backlog",
            actors.backlog() as f32,
            progress.update,
        );
        tb.add_scalar(
            "pipeline/dropped_games_total",
            actors.dropped() as f32,
            progress.update,
        );
        tb.add_scalar(
            "pipeline/model_version_lag",
            progress
                .update
                .saturating_sub(event.batch.oldest_version as usize) as f32,
            progress.update,
        );
        tb.add_scalar(
            "replay/fill_rate",
            event.pool_samples as f32 / config.replay_capacity.max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "replay/train_samples",
            event.train_samples as f32,
            progress.update,
        );
        tb.add_scalar(
            "replay/train_to_new_sample_ratio",
            event.train_samples as f32 / event.batch.samples.len().max(1) as f32,
            progress.update,
        );
        tb.add_scalar(
            "progress/total_games",
            progress.total_games as f32,
            progress.update,
        );
        tb.add_scalar(
            "progress/total_samples",
            progress.total_samples as f32,
            progress.update,
        );
        if config.arena_interval > 0 && progress.update % config.arena_interval == 0 {
            println!(
                "arena    : starting games={} simulations={} workers={}",
                config.arena_games,
                config.arena_simulations,
                rayon::current_num_threads().min(config.arena_games.max(1))
            );
            let arena_started = Instant::now();
            let report = arena_controlled(
                &event.model,
                &best,
                config.arena_games,
                SearchConfig {
                    simulations: config.arena_simulations,
                    cpuct: config.cpuct,
                    cpuct_log: config.cpuct_log,
                    cpuct_base: config.cpuct_base,
                    root_desired_per_child_visits_coeff: config.root_desired_per_child_visits_coeff,
                    policy_softmax_temp: config.policy_softmax_temp,
                    root_policy_temperature_early: config.root_policy_temperature_early,
                    root_policy_temperature: config.root_policy_temperature,
                    root_policy_temperature_halflife: config.root_policy_temperature_halflife,
                    root_num_symmetries_to_sample: config.root_num_symmetries_to_sample,
                    use_graph_search: config.use_graph_search,
                    graph_search_max_nodes: config.graph_search_max_nodes,
                    use_lcb_for_selection: config.use_lcb_for_selection,
                    lcb_stdevs: config.lcb_stdevs,
                    min_visit_prop_for_lcb: config.min_visit_prop_for_lcb,
                    opening_random_plies: config.arena_opening_plies,
                    opening_seed: config.seed ^ progress.update as u64,
                    ..Default::default()
                },
                Some(&stop),
            );
            let arena_seconds = arena_started.elapsed().as_secs_f32();
            let lower_bound = report.score_rate_lower_bound(config.arena_promotion_confidence_z);
            let promoted = report.score_rate() >= config.arena_promotion_rate && lower_bound > 0.5;
            let black_games =
                (report.wins_as_black + report.losses_as_black + report.draws_as_black).max(1)
                    as f32;
            let white_games =
                (report.wins_as_white + report.losses_as_white + report.draws_as_white).max(1)
                    as f32;
            let score_as_black =
                (report.wins_as_black as f32 + report.draws_as_black as f32 * 0.5) / black_games;
            let score_as_white =
                (report.wins_as_white as f32 + report.draws_as_white as f32 * 0.5) / white_games;
            println!(
                "arena    : W/L/D={}/{}/{} score={:.2}% lower={:.2}% elo={:+.1} promoted={}",
                report.wins,
                report.losses,
                report.draws,
                report.score_rate() * 100.0,
                lower_bound * 100.0,
                report.elo_diff(),
                promoted
            );
            println!(
                "arena    : paired_openings={} candidate_black={:.2}% candidate_white={:.2}% avg_plies={:.1}",
                report.paired_openings,
                score_as_black * 100.0,
                score_as_white * 100.0,
                report.plies as f32 / report.games().max(1) as f32
            );
            let arena_games = report.games().max(1) as f32;
            tb.add_scalar("arena/score_rate", report.score_rate(), progress.update);
            tb.add_scalar("arena/score_lower_bound_90", lower_bound, progress.update);
            tb.add_scalar(
                "arena/score_standard_error",
                report.score_rate_standard_error(),
                progress.update,
            );
            tb.add_scalar("arena/elo_diff", report.elo_diff(), progress.update);
            tb.add_scalar(
                "arena/average_plies",
                report.plies as f32 / arena_games,
                progress.update,
            );
            tb.add_scalar(
                "arena/average_plies_win",
                report.win_plies as f32 / report.wins.max(1) as f32,
                progress.update,
            );
            tb.add_scalar(
                "arena/average_plies_loss",
                report.loss_plies as f32 / report.losses.max(1) as f32,
                progress.update,
            );
            tb.add_scalar(
                "arena/average_plies_draw",
                report.draw_plies as f32 / report.draws.max(1) as f32,
                progress.update,
            );
            tb.add_scalar(
                "arena/win_rate",
                report.wins as f32 / arena_games,
                progress.update,
            );
            tb.add_scalar(
                "arena/loss_rate",
                report.losses as f32 / arena_games,
                progress.update,
            );
            tb.add_scalar(
                "arena/draw_rate",
                report.draws as f32 / arena_games,
                progress.update,
            );
            tb.add_scalar("arena/seconds", arena_seconds, progress.update);
            tb.add_scalar(
                "arena/games_per_second",
                report.games() as f32 / arena_seconds.max(1.0e-6),
                progress.update,
            );
            tb.add_scalar("arena/promoted", f32::from(promoted), progress.update);
            tb.add_scalar("arena/score_as_black", score_as_black, progress.update);
            tb.add_scalar("arena/score_as_white", score_as_white, progress.update);
            if promoted {
                best = event.model.clone();
                best.save(&config.best_model_path)?;
                println!(
                    "promote  : best={} model_version={}",
                    config.best_model_path, progress.update
                );
            }
        }
        tb.flush();
        if progress.update >= end {
            stop.store(true, Ordering::SeqCst);
        }
        trainer_ack_tx
            .send(())
            .map_err(|_| io::Error::other("Trainer 确认通道提前关闭"))?;
    }
    stop.store(true, Ordering::SeqCst);
    drop(trainer_ack_tx);
    drop(event_rx);
    actors.shutdown()?;
    collector
        .join()
        .map_err(|_| io::Error::other("Collector 线程异常退出"))?;
    trainer
        .join()
        .map_err(|_| io::Error::other("Trainer 线程异常退出"))??;
    if !interrupted.load(Ordering::SeqCst) && Path::new(&config.replay_path).exists() {
        fs::remove_file(&config.replay_path)?;
    }
    tb.flush();
    println!(
        "stopped  : update={} total_games={} total_samples={}",
        progress.update, progress.total_games, progress.total_samples
    );
    Ok(())
}

fn merge_game(p: &mut PendingBatch, mut game: SelfplayGame) {
    p.oldest_version = p.oldest_version.min(game.model_version);
    p.newest_version = p.newest_version.max(game.model_version);
    p.workers.insert(game.worker);
    p.stats.add_assign(&game.stats);
    for sample in &mut game.samples {
        sample.generation = game.model_version;
    }
    p.samples.extend(game.samples);
    p.games += 1;
}

fn print_event(
    config: &AzLoopConfig,
    progress: &Progress,
    event: &TrainerEvent,
    workers: usize,
    device: &str,
    checkpoint: Option<&Path>,
) {
    let searches = event.batch.stats.searches.max(1) as f32;
    println!(
        "update   : {:06} games={} samples={} total_samples={}{}",
        progress.update,
        event.batch.games,
        event.batch.samples.len(),
        progress.total_samples,
        checkpoint
            .map(|x| format!(" checkpoint={}", x.display()))
            .unwrap_or_default()
    );
    println!(
        "result   : B/W/D={}/{}/{} avg_plies={:.1}",
        event.batch.stats.black_wins,
        event.batch.stats.white_wins,
        event.batch.stats.draws,
        event.batch.stats.plies as f32 / event.batch.games.max(1) as f32
    );
    println!(
        "opening  : randomized={}/{} rate={:.1}% plies={} balanced={}/{} attempts={:.2} abs_v={:.3}",
        event.batch.stats.random_opening_games,
        event.batch.games,
        event.batch.stats.random_opening_games as f32 * 100.0 / event.batch.games.max(1) as f32,
        event.batch.stats.random_opening_plies,
        event.batch.stats.balanced_opening_games,
        event.batch.games,
        event.batch.stats.balanced_opening_attempts as f32
            / event.batch.stats.balanced_opening_games.max(1) as f32,
        event.batch.stats.balanced_opening_abs_value_sum
            / event.batch.stats.balanced_opening_games.max(1) as f32
    );
    println!(
        "search   : avg_sims={:.1} entropy={:.3} visited={:.1} surprise={:.3}/{:.3}",
        event.batch.stats.simulations as f32 / searches,
        event.batch.stats.entropy_sum / searches,
        event.batch.stats.visited_actions_sum as f32 / searches,
        event.batch.stats.policy_surprise_sum / searches,
        event.batch.stats.value_surprise_sum / searches
    );
    println!(
        "pipeline : actors={}/{} versions={}-{} collect={:.2}s",
        event.batch.workers.len(),
        workers,
        event.batch.oldest_version,
        event.batch.newest_version,
        event.batch.collect_seconds
    );
    let selfplay_seconds = event.batch.collect_seconds.max(1.0e-6);
    println!(
        "selfplay : cycle={:.2}s gps={:.2} samples/s={:.0} simulations/s={:.0}",
        event.batch.collect_seconds,
        event.batch.games as f32 / selfplay_seconds,
        event.batch.samples.len() as f32 / selfplay_seconds,
        event.batch.stats.simulations as f32 / selfplay_seconds
    );
    println!(
        "replay   : samples={}/{} fill={:.1}%",
        event.pool_samples,
        config.replay_capacity,
        event.pool_samples as f32 * 100.0 / config.replay_capacity.max(1) as f32
    );
    println!(
        "sampling : train_samples={} recent_quota={:.1}% actual_recent={:.1}% window={} updates",
        event.train_samples,
        event.recent_quota_rate * 100.0,
        event.actual_recent_rate * 100.0,
        config.replay_recent_updates
    );
    println!(
        "balance  : new_samples={} train/new={:.2}",
        event.batch.samples.len(),
        event.train_samples as f32 / event.batch.samples.len().max(1) as f32
    );
    println!(
        "train    : device={} samples={} steps={} total_steps={} lr={:.6} loss={:.4} policy={:.4} value={:.4} sample={:.3}s time={:.2}s sps={:.1}",
        device,
        event.train_stats.samples,
        event.train_stats.optimizer_steps,
        progress.optimizer_steps,
        event.learning_rate,
        event.train_stats.loss,
        event.train_stats.policy_loss,
        event.train_stats.value_loss,
        event.sampling_seconds,
        event.train_seconds,
        event.train_stats.samples as f32 / event.train_seconds.max(1e-6)
    );
}

fn current_lr(c: &AzLoopConfig, optimizer_step: usize) -> f32 {
    if optimizer_step < c.learning_rate_warmup_steps {
        let progress = optimizer_step as f32 / c.learning_rate_warmup_steps as f32;
        return c.learning_rate_min + (c.learning_rate - c.learning_rate_min) * progress;
    }
    let decay_step = optimizer_step - c.learning_rate_warmup_steps;
    if decay_step >= c.learning_rate_cosine_steps {
        return c.learning_rate_min;
    }
    let progress = decay_step as f32 / c.learning_rate_cosine_steps as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    c.learning_rate_min + (c.learning_rate - c.learning_rate_min) * cosine
}

fn ema_decay_for_update(configured_decay: f32, initialized: bool) -> f32 {
    if initialized { configured_decay } else { 0.0 }
}

fn load_or_init(path: &str) -> io::Result<PolicyValueModel> {
    if Path::new(path).exists() {
        PolicyValueModel::load(path)
    } else {
        let m = PolicyValueModel::default();
        m.save(path)?;
        Ok(m)
    }
}
fn load_progress(path: &str) -> io::Result<Progress> {
    if !Path::new(path).exists() {
        return Ok(Progress::default());
    }
    serde_json::from_slice(&fs::read(path)?).map_err(io::Error::other)
}
fn save_progress(path: &str, p: &Progress) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(p).map_err(io::Error::other)?,
    )
}
fn checkpoint_path(c: &AzLoopConfig, update: usize) -> PathBuf {
    Path::new(&c.checkpoint_dir).join(format!("update-{update:06}-model.safetensors"))
}
fn prune_checkpoints(c: &AzLoopConfig) -> io::Result<()> {
    if c.max_checkpoints == 0 {
        return Ok(());
    }
    let mut files = match fs::read_dir(&c.checkpoint_dir) {
        Ok(x) => x
            .filter_map(Result::ok)
            .map(|x| x.path())
            .filter(|x| {
                x.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("update-"))
            })
            .collect::<Vec<_>>(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    files.sort();
    let remove = files.len().saturating_sub(c.max_checkpoints);
    for path in files.into_iter().take(remove) {
        fs::remove_file(path)?
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learning_rate_warms_up_then_cosine_decays() {
        let config = AzLoopConfig::default();
        assert!((current_lr(&config, 0) - 0.0001).abs() < 1e-8);
        assert!((current_lr(&config, 100) - 0.00045).abs() < 1e-8);
        assert!((current_lr(&config, 200) - 0.0008).abs() < 1e-8);
        assert!((current_lr(&config, 5_200) - 0.00045).abs() < 1e-8);
        assert!((current_lr(&config, 10_200) - 0.0001).abs() < 1e-8);
        assert!((current_lr(&config, 20_000) - 0.0001).abs() < 1e-8);
    }

    #[test]
    fn ema_first_update_copies_online_model() {
        assert_eq!(ema_decay_for_update(0.999, false), 0.0);
        assert_eq!(ema_decay_for_update(0.999, true), 0.999);
    }
}
