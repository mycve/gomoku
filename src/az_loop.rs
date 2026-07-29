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
    train_seconds: f32,
    pool_samples: usize,
    learning_rate: f32,
    train_samples: usize,
    recent_quota_rate: f32,
    actual_recent_rate: f32,
    value_search_target_weight: f32,
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
    let initial_ema = if Path::new(&config.ema_model_path).exists() {
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
        "priors   : softmax_temp={:.2} root_noise(alpha={:.2}, fraction={:.2})",
        config.policy_softmax_temp, config.root_dirichlet_alpha, config.root_exploration_fraction
    );
    println!(
        "aux      : moves_left_weight={:.3} search_value_weight={:.3}@update{} ema_decay={:.6} ema_model={}",
        config.moves_left_loss_weight,
        config.value_search_target_weight,
        config.value_search_target_start_update,
        config.ema_decay,
        config.ema_model_path
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
            root_dirichlet_alpha: config.root_dirichlet_alpha,
            root_exploration_fraction: config.root_exploration_fraction,
            policy_softmax_temp: config.policy_softmax_temp,
            temperature_start: config.temperature_start,
            temperature_endgame: config.temperature_endgame,
            temperature_decay_delay_plies: config.temperature_decay_delay_plies,
            temperature_decay_plies: config.temperature_decay_plies,
            temperature_value_cutoff: config.temperature_value_cutoff,
            temperature_visit_offset: config.temperature_visit_offset,
            ..Default::default()
        },
        config.seed,
    );
    let actor_rx = actors.take_receiver();
    let backlog = actors.backlog_counter();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<PendingBatch>(1);
    let collector_stop = Arc::clone(&stop);
    let collector_version = Arc::clone(&version);
    let games_per_update = config.games_per_update.max(1);
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
            if pending.games < games_per_update {
                continue;
            }
            pending.collect_seconds = started.elapsed().as_secs_f32();
            let next = PendingBatch {
                oldest_version: u64::MAX,
                ..Default::default()
            };
            match ready_tx.try_send(std::mem::replace(&mut pending, next)) {
                Ok(()) | Err(mpsc::TrySendError::Full(_)) => {}
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
            started = Instant::now();
            if collector_stop.load(Ordering::SeqCst) {
                break;
            }
        }
    });
    let (event_tx, event_rx) = mpsc::sync_channel::<TrainerEvent>(0);
    let (trainer_ack_tx, trainer_ack_rx) = mpsc::sync_channel::<()>(0);
    let trainer_stop = Arc::clone(&stop);
    let trainer_interrupted = Arc::clone(&interrupted);
    let trainer_version = Arc::clone(&version);
    let trainer_config = config.clone();
    let start_update = progress.update;
    let trainer = thread::spawn(move || -> io::Result<()> {
        let mut model = initial_model;
        let mut ema_model = initial_ema;
        let mut pool = initial_pool;
        let mut index = 0usize;
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
            let lr = current_lr(&trainer_config, update - 1);
            let started = Instant::now();
            let sampled = replay::sample_mixed_recent(
                &pool,
                trainer_config.train_samples_per_update,
                trainer_config.replay_recent_sample_fraction,
                trainer_config.replay_recent_updates,
                trainer_config.seed ^ update as u64,
            );
            let train_samples = sampled.samples.len();
            let recent_quota_rate = sampled.recent_quota as f32 / train_samples.max(1) as f32;
            let actual_recent_rate = sampled.actual_recent as f32 / train_samples.max(1) as f32;
            let value_search_target_weight =
                if update >= trainer_config.value_search_target_start_update {
                    trainer_config.value_search_target_weight
                } else {
                    0.0
                };
            let train_stats = candle_train::train_controlled(
                &mut model,
                &sampled.samples,
                trainer_config.batch_epochs,
                lr,
                trainer_config.batch_size,
                &trainer_config.gpu_devices,
                trainer_config.moves_left_loss_weight,
                value_search_target_weight,
                Some(&trainer_stop),
            )?;
            if trainer_stop.load(Ordering::SeqCst) {
                break;
            }
            let effective_decay = trainer_config
                .ema_decay
                .clamp(0.0, 1.0)
                .powi(train_stats.optimizer_steps.min(i32::MAX as usize) as i32);
            ema_model.update_ema(&model, effective_decay);
            let train_seconds = started.elapsed().as_secs_f32();
            if event_tx
                .send(TrainerEvent {
                    model: ema_model.clone(),
                    online_model: model.clone(),
                    batch,
                    train_stats,
                    train_seconds,
                    pool_samples: pool.len(),
                    learning_rate: lr,
                    train_samples,
                    recent_quota_rate,
                    actual_recent_rate,
                    value_search_target_weight,
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
    });
    let train_devices =
        compact_device_names(&candle_train::training_device_names(&config.gpu_devices)?);
    let end = target_update.unwrap_or(usize::MAX);
    println!(
        "loop     : mode=batch-async actors={} actor_queue={}(nonblocking-drop) collector_queue=1(nonblocking-drop) trainer_queue=rendezvous games/update={} sims={} train_device={} batch={} arena_opening_plies={}",
        workers,
        queue_capacity,
        config.games_per_update,
        config.simulations,
        train_devices,
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
                    return Err(io::Error::other("Trainer 在线程完成更新前退出"));
                }
            }
        };
        progress.update += 1;
        progress.total_games += event.batch.games;
        progress.total_samples += event.batch.samples.len();
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
            &train_devices,
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
        tb.add_scalar(
            "train/moves_left_loss",
            event.train_stats.moves_left_loss,
            progress.update,
        );
        tb.add_scalar(
            "train/search_value_target_weight",
            event.value_search_target_weight,
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
                    opening_random_plies: config.arena_opening_plies,
                    opening_seed: config.seed ^ progress.update as u64,
                    ..Default::default()
                },
                Some(&stop),
            );
            let arena_seconds = arena_started.elapsed().as_secs_f32();
            let promoted = report.score_rate() >= config.arena_promotion_rate;
            println!(
                "arena    : W/L/D={}/{}/{} score={:.2}% elo={:+.1} promoted={}",
                report.wins,
                report.losses,
                report.draws,
                report.score_rate() * 100.0,
                report.elo_diff(),
                promoted
            );
            let arena_games = report.games().max(1) as f32;
            tb.add_scalar("arena/score_rate", report.score_rate(), progress.update);
            tb.add_scalar(
                "arena/score_lower_bound_90",
                report.score_rate_lower_bound(1.28),
                progress.update,
            );
            tb.add_scalar(
                "arena/score_standard_error",
                report.score_rate_standard_error(),
                progress.update,
            );
            tb.add_scalar("arena/elo_diff", report.elo_diff(), progress.update);
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
            let black_games =
                (report.wins_as_black + report.losses_as_black + report.draws_as_black).max(1)
                    as f32;
            let white_games =
                (report.wins_as_white + report.losses_as_white + report.draws_as_white).max(1)
                    as f32;
            tb.add_scalar(
                "arena/score_as_black",
                (report.wins_as_black as f32 + report.draws_as_black as f32 * 0.5) / black_games,
                progress.update,
            );
            tb.add_scalar(
                "arena/score_as_white",
                (report.wins_as_white as f32 + report.draws_as_white as f32 * 0.5) / white_games,
                progress.update,
            );
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
        "search   : avg_sims={:.1} entropy={:.3} visited={:.1}",
        event.batch.stats.simulations as f32 / searches,
        event.batch.stats.entropy_sum / searches,
        event.batch.stats.visited_actions_sum as f32 / searches
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
        "train    : device={} samples={} steps={} lr={:.6} search_value_weight={:.3} loss={:.4} policy={:.4} value={:.4} moves_left={:.4} time={:.2}s sps={:.1}",
        device,
        event.train_stats.samples,
        event.train_stats.optimizer_steps,
        event.learning_rate,
        event.value_search_target_weight,
        event.train_stats.loss,
        event.train_stats.policy_loss,
        event.train_stats.value_loss,
        event.train_stats.moves_left_loss,
        event.train_seconds,
        event.train_stats.samples as f32 / event.train_seconds.max(1e-6)
    );
}

fn current_lr(c: &AzLoopConfig, update: usize) -> f32 {
    let exponent = update.min(i32::MAX as usize) as i32;
    (c.learning_rate * c.learning_rate_decay.powi(exponent)).max(c.learning_rate_min)
}

fn compact_device_names(devices: &[String]) -> String {
    if devices.is_empty() {
        return "none".into();
    }
    let cuda_ids = devices
        .iter()
        .map(|name| {
            name.strip_prefix("cuda:")
                .and_then(|id| id.parse::<usize>().ok())
        })
        .collect::<Option<Vec<_>>>();
    if let Some(ids) = cuda_ids {
        if ids.len() == 1 {
            return format!("cuda:{}", ids[0]);
        }
        if ids.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            return format!("cuda:{}-{}({}卡)", ids[0], ids[ids.len() - 1], ids.len());
        }
        return format!("cuda:{}卡", ids.len());
    }
    if devices.len() <= 2 {
        devices.join(",")
    } else {
        format!("{}等{}设备", devices[0], devices.len())
    }
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
    fn compacts_sequential_cuda_devices() {
        let devices = (0..8).map(|id| format!("cuda:{id}")).collect::<Vec<_>>();
        assert_eq!(compact_device_names(&devices), "cuda:0-7(8卡)");
        assert_eq!(compact_device_names(&["cuda:3".into()]), "cuda:3");
    }
}
