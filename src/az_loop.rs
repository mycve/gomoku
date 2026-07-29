use crate::{
    async_selfplay::{AsyncSelfplay, SelfplayGame},
    az_loop_config::AzLoopConfig,
    candle_train,
    mcts::SearchConfig,
    model::PolicyValueModel,
    replay,
    selfplay::{SelfplayStats, arena},
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
    batch: PendingBatch,
    train_stats: crate::selfplay::TrainStats,
    train_seconds: f32,
    pool_samples: usize,
    learning_rate: f32,
}

pub fn run(config: AzLoopConfig, target_update: Option<usize>) -> io::Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::SeqCst)).map_err(io::Error::other)?;
    let mut progress = load_progress(&config.progress_path)?;
    let initial_model = load_or_init(&config.model_path)?;
    let mut best = if Path::new(&config.best_model_path).exists() {
        PolicyValueModel::load(&config.best_model_path)?
    } else {
        initial_model.save(&config.best_model_path)?;
        initial_model.clone()
    };
    let workers = if config.selfplay_workers == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        config.selfplay_workers.max(1)
    };
    let queue_capacity = if config.selfplay_queue_capacity == 0 {
        workers.saturating_mul(8).max(32)
    } else {
        config.selfplay_queue_capacity.max(1)
    };
    let published = Arc::new(RwLock::new(initial_model.clone()));
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
        },
        config.seed,
    );
    let actor_rx = actors.take_receiver();
    let backlog = actors.backlog_counter();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<PendingBatch>(1);
    let collector_stop = Arc::clone(&stop);
    let games_per_update = config.games_per_update.max(1);
    let collector = thread::spawn(move || {
        let mut pending = PendingBatch {
            oldest_version: u64::MAX,
            ..Default::default()
        };
        let mut started = Instant::now();
        while let Ok(game) = actor_rx.recv() {
            backlog.fetch_sub(1, Ordering::Relaxed);
            merge_game(&mut pending, game);
            if pending.games < games_per_update {
                continue;
            }
            pending.collect_seconds = started.elapsed().as_secs_f32();
            let next = PendingBatch {
                oldest_version: u64::MAX,
                ..Default::default()
            };
            if ready_tx
                .send(std::mem::replace(&mut pending, next))
                .is_err()
            {
                break;
            }
            started = Instant::now();
            if collector_stop.load(Ordering::SeqCst) {
                break;
            }
        }
    });
    let (event_tx, event_rx) = mpsc::sync_channel::<TrainerEvent>(2);
    let trainer_stop = Arc::clone(&stop);
    let trainer_config = config.clone();
    let start_update = progress.update;
    let trainer = thread::spawn(move || -> io::Result<()> {
        let mut model = initial_model;
        let mut pool = replay::load(&trainer_config.replay_path)?;
        let mut index = 0usize;
        while let Ok(batch) = ready_rx.recv() {
            if trainer_stop.load(Ordering::SeqCst) {
                break;
            }
            pool.extend(batch.samples.iter().cloned());
            if pool.len() > trainer_config.replay_capacity {
                pool.drain(..pool.len() - trainer_config.replay_capacity);
            }
            let update = start_update + index + 1;
            let lr = current_lr(&trainer_config, update - 1);
            let started = Instant::now();
            let train_stats = candle_train::train(
                &mut model,
                &pool,
                trainer_config.batch_epochs,
                lr,
                trainer_config.batch_size,
                &trainer_config.gpu_devices,
            )?;
            let train_seconds = started.elapsed().as_secs_f32();
            replay::save(&trainer_config.replay_path, &pool)?;
            if event_tx
                .send(TrainerEvent {
                    model: model.clone(),
                    batch,
                    train_stats,
                    train_seconds,
                    pool_samples: pool.len(),
                    learning_rate: lr,
                })
                .is_err()
            {
                break;
            }
            index += 1;
        }
        Ok(())
    });
    let train_devices = candle_train::training_device_names(&config.gpu_devices)?.join(",");
    let end = target_update.unwrap_or(usize::MAX);
    println!(
        "loop     : mode=batch-async actors={} actor_queue={} collector_queue=1 trainer_queue=2 games/update={} sims={} train_device={} batch={}",
        workers,
        queue_capacity,
        config.games_per_update,
        config.simulations,
        train_devices,
        config.batch_size
    );
    let mut tb = SummaryWriter::new(&config.tensorboard_logdir);
    'main: while progress.update < end && !stop.load(Ordering::SeqCst) {
        let event = loop {
            match event_rx.recv_timeout(Duration::from_millis(250)) {
                Ok(event) => break event,
                Err(mpsc::RecvTimeoutError::Timeout) if stop.load(Ordering::SeqCst) => break 'main,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
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
        event.model.save(&config.model_path)?;
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
        tb.add_scalar("replay/samples", event.pool_samples as f32, progress.update);
        if config.arena_interval > 0 && progress.update % config.arena_interval == 0 {
            let report = arena(
                &event.model,
                &best,
                config.arena_games,
                SearchConfig {
                    simulations: config.arena_simulations,
                    cpuct: config.cpuct,
                },
            );
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
            if promoted {
                best = event.model.clone();
                best.save(&config.best_model_path)?;
            }
        }
    }
    stop.store(true, Ordering::SeqCst);
    drop(event_rx);
    actors.shutdown()?;
    collector
        .join()
        .map_err(|_| io::Error::other("Collector 线程异常退出"))?;
    trainer
        .join()
        .map_err(|_| io::Error::other("Trainer 线程异常退出"))??;
    tb.flush();
    println!(
        "stopped  : update={} total_games={} total_samples={}",
        progress.update, progress.total_games, progress.total_samples
    );
    Ok(())
}

fn merge_game(p: &mut PendingBatch, game: SelfplayGame) {
    p.oldest_version = p.oldest_version.min(game.model_version);
    p.newest_version = p.newest_version.max(game.model_version);
    p.workers.insert(game.worker);
    p.stats.add_assign(&game.stats);
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
    println!(
        "replay   : samples={}/{} fill={:.1}%",
        event.pool_samples,
        config.replay_capacity,
        event.pool_samples as f32 * 100.0 / config.replay_capacity.max(1) as f32
    );
    println!(
        "train    : device={} samples={} lr={:.6} loss={:.4} policy={:.4} value={:.4} time={:.2}s sps={:.1}",
        device,
        event.train_stats.samples,
        event.learning_rate,
        event.train_stats.loss,
        event.train_stats.policy_loss,
        event.train_stats.value_loss,
        event.train_seconds,
        event.train_stats.samples as f32 / event.train_seconds.max(1e-6)
    );
}

fn current_lr(c: &AzLoopConfig, update: usize) -> f32 {
    (c.learning_rate * c.learning_rate_decay.powi(update as i32)).max(c.learning_rate_min)
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
