//! 性能诊断，不覆盖输入模型或回放。每次进程只测一个场景。
use go9::{
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
        .unwrap_or("go9-v31-model.safetensors");
    let replay_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("data/go9-v31/replay.jsonl");
    let mut model = PolicyValueModel::load(model_path)?;
    let samples = replay::load(replay_path)?;
    if samples.is_empty() {
        return Err(io::Error::other("性能诊断需要非空回放"));
    }
    let cfg = SearchConfig {
        simulations: 400,
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
    for plies in [20, 50, 80, 110] {
        if let Some(sample) = samples
            .iter()
            .min_by_key(|s| s.board.move_count().abs_diff(plies))
        {
            boards.push(sample.board.clone());
        }
    }
    match mode {
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
                    black_box(scoring::analyze(board));
                }
                println!(
                    "plies={} clone_us={clone:.3} encode_us={encode:.3} scoring_us={:.3}",
                    board.move_count(),
                    start.elapsed().as_secs_f64() * 1e6 / runs as f64
                );
            }
        }
        "train" => {
            let batch_size = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
            let batch = samples
                .iter()
                .step_by((samples.len() / 2048).max(1))
                .take(2048)
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
            let stats =
                session.train_controlled(&mut model, &batch, 5, 0.0001, batch_size, None)?;
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
    profile::print_report();
    Ok(())
}
