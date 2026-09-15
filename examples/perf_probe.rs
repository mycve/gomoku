//! 性能诊断，不覆盖输入模型或回放。每次进程只测一个场景。
use go19::{
    candle_train::TrainingSession,
    features,
    game::Board,
    mcts::{SearchConfig, search},
    model::PolicyValueModel,
    profile, replay, scoring,
};
use rayon::prelude::*;
use std::{hint::black_box, io, time::Instant};

fn main() -> io::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let mode = args.get(1).map(String::as_str).unwrap_or("search");
    let model_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("go19-v32-model.safetensors");
    let replay_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("data/go19-v32/replay.jsonl");
    if mode == "prepare" {
        if std::path::Path::new(model_path).exists() || std::path::Path::new(replay_path).exists() {
            return Err(io::Error::other("测试输入已存在，不覆盖"));
        }
        let model = PolicyValueModel::random(128, 719);
        let samples = go19::selfplay::generate(
            &model,
            1,
            SearchConfig {
                simulations: 4,
                ..Default::default()
            },
        );
        model.save(model_path)?;
        replay::save(replay_path, &samples)?;
        println!("prepared {} fresh MC samples", samples.len());
        return Ok(());
    }
    let mut model = if model_path == "random" {
        PolicyValueModel::random(128, 719)
    } else {
        PolicyValueModel::load(model_path)?
    };
    let samples = replay::load(replay_path)?;
    let samples = if mode == "sampling" {
        let pool_size = std::env::var("GO19_PROBE_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(samples.len());
        if pool_size == samples.len() {
            samples
        } else {
            samples.iter().cycle().take(pool_size).cloned().collect()
        }
    } else {
        samples
    };
    if samples.is_empty() {
        return Err(io::Error::other("性能诊断需要非空回放"));
    }
    let cfg = SearchConfig {
        simulations: std::env::var("GO19_PROBE_SIMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(800),
        cpuct: 2.0,
        cpuct_log: 1.5,
        ..Default::default()
    };
    println!(
        "mode={mode} hidden={} replay_samples={}",
        model.hidden_size,
        samples.len()
    );
    let mut boards = vec![Board::new()];
    for plies in [100, 250, 400, 500] {
        if let Some(sample) = samples
            .iter()
            .min_by_key(|s| s.board.move_count().abs_diff(plies))
        {
            boards.push(sample.board.clone());
        }
    }
    match mode {
        "sampling" => {
            let count = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(81920);
            let start = Instant::now();
            let batch = replay::sample_mixed_recent(&samples, count, 0.4, 5, 0.4, 0.1, 719);
            println!(
                "samples={} sampling_seconds={:.6} recent_quota={} actual_recent={}",
                batch.samples.len(),
                start.elapsed().as_secs_f64(),
                batch.recent_quota,
                batch.actual_recent
            );
            if args.get(5).is_some_and(|arg| arg == "digest") {
                let mut digest = 0xcbf29ce484222325u64;
                for sample in &batch.samples {
                    for byte in serde_json::to_vec(sample).map_err(io::Error::other)? {
                        digest = (digest ^ byte as u64).wrapping_mul(0x100000001b3);
                    }
                }
                println!("serialized_digest={digest:016x}");
            }
        }
        "fingerprint" => {
            for board in &boards {
                let evaluation = model.evaluate(board);
                let candidates = search(board, &model, cfg)
                    .into_iter()
                    .map(|c| (c.mv.0, c.visits, c.q, c.prior))
                    .collect::<Vec<_>>();
                println!(
                    "{}",
                    serde_json::to_string(&(
                        board.move_count(),
                        features::encode(board, board.to_move()),
                        evaluation,
                        candidates
                    ))
                    .map_err(io::Error::other)?
                );
            }
        }
        "search" | "graph-off" | "symmetry-one" => {
            let cfg = SearchConfig {
                use_graph_search: mode != "graph-off",
                root_num_symmetries_to_sample: if mode == "symmetry-one" { 1 } else { 4 },
                ..cfg
            };
            for board in &boards {
                black_box(search(board, &model, cfg));
            }
            profile::reset();
            for board in &boards {
                profile::reset();
                let start = Instant::now();
                let mut visits = 0;
                for _ in 0..20 {
                    visits += search(board, &model, cfg)
                        .iter()
                        .map(|c| c.visits as usize)
                        .sum::<usize>();
                }
                println!(
                    "plies={} stones={} total_ms={:.3} simulations_per_s={:.0} actual_visits={visits}",
                    board.move_count(),
                    board.cells().iter().filter(|&&s| s != 0).count(),
                    start.elapsed().as_secs_f64() * 1000.0,
                    visits as f64 / start.elapsed().as_secs_f64()
                );
                profile::print_report();
            }
        }
        "parallel" => {
            let workers = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(io::Error::other)?;
            pool.install(|| {
                boards.par_iter().for_each(|board| {
                    black_box(search(board, &model, cfg));
                })
            });
            let start = Instant::now();
            let visits: usize = pool.install(|| {
                (0..200usize)
                    .into_par_iter()
                    .map(|i| {
                        search(&boards[i % boards.len()], &model, cfg)
                            .iter()
                            .map(|c| c.visits as usize)
                            .sum::<usize>()
                    })
                    .sum()
            });
            println!(
                "workers={workers} searches=200 elapsed_ms={:.3} simulations_per_s={:.0}",
                start.elapsed().as_secs_f64() * 1000.0,
                visits as f64 / start.elapsed().as_secs_f64()
            );
        }
        "micro" => {
            for board in &boards {
                // 仅用于隔离历史长度的成本；不能用于实际搜索或训练。
                let mut position = serde_json::to_value(board).map_err(io::Error::other)?;
                position["history"] = serde_json::json!([board.cells()]);
                let short_history: Board =
                    serde_json::from_value(position).map_err(io::Error::other)?;
                let runs = 1000;
                let start = Instant::now();
                for _ in 0..runs {
                    black_box(board.clone());
                }
                let clone = start.elapsed().as_secs_f64() * 1e6 / runs as f64;
                let start = Instant::now();
                for _ in 0..runs {
                    black_box(features::encode(board, board.to_move()));
                }
                let encode = start.elapsed().as_secs_f64() * 1e6 / runs as f64;
                let start = Instant::now();
                for _ in 0..runs {
                    black_box(short_history.clone());
                }
                let short_clone = start.elapsed().as_secs_f64() * 1e6 / runs as f64;
                let start = Instant::now();
                for _ in 0..runs {
                    black_box(features::encode(&short_history, short_history.to_move()));
                }
                let short_encode = start.elapsed().as_secs_f64() * 1e6 / runs as f64;
                let start = Instant::now();
                for _ in 0..runs {
                    black_box(scoring::analyze(board));
                }
                println!(
                    "plies={} clone_us={clone:.3} encode_us={encode:.3} short_history_clone_us={short_clone:.3} short_history_encode_us={short_encode:.3} scoring_us={:.3}",
                    board.move_count(),
                    start.elapsed().as_secs_f64() * 1e6 / runs as f64
                );
            }
        }
        "train" | "train-loaded" => {
            let batch_size = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
            let sample_count = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2048);
            let batch = samples
                .iter()
                .step_by((samples.len() / sample_count.max(1)).max(1))
                .cycle()
                .take(sample_count)
                .cloned()
                .collect::<Vec<_>>();
            let mut session = TrainingSession::new(&model, 0.0001)?;
            session.train_controlled(
                &mut model,
                &batch[..batch_size.min(batch.len())],
                1,
                0.0001,
                batch_size,
                None,
            )?;
            profile::reset();
            let start = Instant::now();
            let epochs = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
            let stats = if mode == "train-loaded" {
                let search_model = model.clone();
                let stop = std::sync::atomic::AtomicBool::new(false);
                let visits = std::sync::atomic::AtomicUsize::new(0);
                let result = std::thread::scope(|scope| {
                    for worker in 0..128 {
                        let board = &boards[worker % boards.len()];
                        let search_model = &search_model;
                        let stop = &stop;
                        let visits = &visits;
                        scope.spawn(move || {
                            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                                let count = search(board, search_model, cfg)
                                    .iter()
                                    .map(|c| c.visits as usize)
                                    .sum::<usize>();
                                visits.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
                            }
                        });
                    }
                    let result = session
                        .train_controlled(&mut model, &batch, epochs, 0.0001, batch_size, None);
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    result
                });
                println!(
                    "search_load_workers=128 simulations_per_s={:.0}",
                    visits.load(std::sync::atomic::Ordering::Relaxed) as f64
                        / start.elapsed().as_secs_f64()
                );
                result?
            } else {
                session.train_controlled(&mut model, &batch, epochs, 0.0001, batch_size, None)?
            };
            println!(
                "batch={batch_size} samples={} elapsed_ms={:.3} samples_per_s={:.0} loss={:.5}",
                stats.samples,
                start.elapsed().as_secs_f64() * 1000.0,
                stats.samples as f64 / start.elapsed().as_secs_f64(),
                stats.loss
            );
        }
        _ => {
            return Err(io::Error::other(
                "mode: search | graph-off | symmetry-one | parallel | micro | train",
            ));
        }
    }
    if !matches!(mode, "search" | "graph-off" | "symmetry-one") {
        profile::print_report();
    }
    Ok(())
}
