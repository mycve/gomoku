use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use gomoku::{
    alphabeta::{AlphaBetaConfig, search as alpha_beta_search},
    az_loop,
    az_loop_config::{DEFAULT_CONFIG_PATH, load_or_create},
    candle_train,
    game::{Board, Move, Outcome, Player},
    mcts::{SearchConfig, search},
    model::PolicyValueModel,
    replay,
    selfplay::arena,
};
use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

#[derive(Parser)]
#[command(
    name = "gomoku",
    version,
    about = "Gomoku AZ policy/value search and training tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// 创建初始策略价值模型。
    AzInit(AzInitArgs),
    /// 搜索局面并显示候选着。
    AzSearch(AzSearchArgs),
    /// 测试固定局面的搜索速度。
    AzBench(AzBenchArgs),
    /// 测试回放样本训练速度。
    AzTrainBench(AzTrainBenchArgs),
    /// 比较完整 Policy/Value 与候选无关 Value 路径。
    AzValueBench(AzValueBenchArgs),
    /// 使用保守 PVS/Alpha-Beta 搜索局面。
    AzAlphaBeta(AzAlphaBetaArgs),
    /// 在回放样本上评估候选无关 WDL 的误差与入口一致性。
    AzValueEval(AzValueEvalArgs),
    /// 按 TOML 配置持续执行自博弈训练。
    AzLoop(AzLoopArgs),
    /// 人工在控制台挑战 Best 模型。
    AzEvalBest(AzEvalBestArgs),
    /// 自动评估候选模型相对 Best 模型的表现。
    AzArenaBest(AzArenaBestArgs),
    /// 终端人机对战（玩家执黑）。
    Play(PlayArgs),
}

#[derive(Args)]
struct AzInitArgs {
    #[arg(default_value = "model.safetensors")]
    output: String,
    #[arg(default_value_t = 192)]
    hidden: usize,
    #[arg(default_value_t = 20260730)]
    seed: u64,
}

#[derive(Args)]
struct AzSearchArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    /// 已落子坐标序列，例如 h8 h9 i8。
    moves: Vec<String>,
}

#[derive(Args)]
struct AzBenchArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 20)]
    repeat: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    moves: Vec<String>,
}

#[derive(Args)]
struct AzTrainBenchArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(default_value = "data/replay.jsonl")]
    replay: String,
    #[arg(default_value_t = 2)]
    epochs: usize,
    #[arg(default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,
}

#[derive(Args)]
struct AzValueBenchArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(long, default_value_t = 100_000)]
    iterations: usize,
    /// 已落子坐标序列，例如 h8 h9 i8。
    moves: Vec<String>,
}

#[derive(Args)]
struct AzAlphaBetaArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(long, default_value_t = 8)]
    depth: u16,
    #[arg(long, default_value_t = 100_000)]
    nodes: u64,
    #[arg(long, default_value_t = 8)]
    threat_depth: u16,
    /// 已落子坐标序列，例如 h8 h9 i8。
    moves: Vec<String>,
}

#[derive(Args)]
struct AzValueEvalArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(default_value = "data/replay.jsonl")]
    replay: String,
    #[arg(long, default_value_t = 10_000)]
    limit: usize,
}

#[derive(Args)]
struct AzLoopArgs {
    #[arg(default_value = DEFAULT_CONFIG_PATH)]
    config: String,
    /// 在完成该绝对更新编号后停止。
    #[arg(long)]
    target_update: Option<usize>,
}

#[derive(Args)]
struct AzArenaBestArgs {
    #[arg(default_value = "model.safetensors")]
    candidate: String,
    #[arg(default_value = "best.safetensors")]
    best: String,
    #[arg(default_value_t = 100)]
    games: usize,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    /// 单侧置信下界的 Z 值；1.28 约为 90%。
    #[arg(long, default_value_t = 1.28)]
    confidence_z: f32,
}

#[derive(Clone, Copy, ValueEnum)]
enum HumanSide {
    Black,
    White,
}

#[derive(Args)]
struct AzEvalBestArgs {
    #[arg(default_value = "best.safetensors")]
    best: String,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    #[arg(long, value_enum, default_value_t = HumanSide::Black)]
    human_side: HumanSide,
    /// 禁用 ANSI 清屏，适合不支持终端控制符的日志窗口。
    #[arg(long)]
    no_clear: bool,
}

#[derive(Args)]
struct PlayArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
}

fn main() -> io::Result<()> {
    match Cli::parse().command {
        None => {
            Cli::command().print_help()?;
            println!();
        }
        Some(Command::AzInit(args)) => {
            PolicyValueModel::random(args.hidden, args.seed).save(&args.output)?;
            println!("model    : initialized {}", args.output);
            println!(
                "arch     : input=451 hidden={} rmsnorm local=4axesx8cells-pattern2 incremental-axis-mean move-logit=lazy value=96x96xWDL3(candidate-independent)",
                args.hidden
            );
            println!("board    : 15x15 freestyle gomoku");
        }
        Some(Command::AzSearch(args)) => {
            let model = load_model(&args.model)?;
            let board = board_from_moves(&args.moves)?;
            println!("{board}");
            print_search(&board, &model, args.simulations, args.cpuct);
        }
        Some(Command::AzBench(args)) => {
            let model = load_model(&args.model)?;
            let board = board_from_moves(&args.moves)?;
            let started = Instant::now();
            for _ in 0..args.repeat {
                let _ = search(
                    &board,
                    &model,
                    SearchConfig {
                        simulations: args.simulations,
                        cpuct: args.cpuct,
                        ..Default::default()
                    },
                );
            }
            let seconds = started.elapsed().as_secs_f64();
            let total = args.repeat * args.simulations;
            println!(
                "bench    : repeats={} simulations/search={} elapsed={seconds:.3}s",
                args.repeat, args.simulations
            );
            println!(
                "speed    : {:.0} simulations/s",
                total as f64 / seconds.max(1e-9)
            );
        }
        Some(Command::AzTrainBench(args)) => {
            let mut model = load_model(&args.model)?;
            let samples = replay::load(&args.replay)?;
            if samples.is_empty() {
                return Err(io::Error::other(format!(
                    "回放池 `{}` 为空，请先运行 az-loop",
                    args.replay
                )));
            }
            let started = Instant::now();
            let device = candle_train::training_device_name(args.gpu_device)?;
            let stats = candle_train::train(
                &mut model,
                &samples,
                args.epochs,
                args.learning_rate,
                args.batch_size,
                args.gpu_device,
            )?;
            let seconds = started.elapsed().as_secs_f64();
            println!(
                "train    : samples={} epochs={} batch_size={} device={} elapsed={seconds:.3}s",
                samples.len(),
                args.epochs,
                args.batch_size,
                device
            );
            println!(
                "speed    : {:.0} samples/s",
                (samples.len() * args.epochs) as f64 / seconds.max(1e-9)
            );
            println!(
                "loss     : total={:.4} policy={:.4} value={:.4}",
                stats.loss, stats.policy_loss, stats.value_loss
            );
        }
        Some(Command::AzValueBench(args)) => {
            let model = load_model(&args.model)?;
            let board = board_from_moves(&args.moves)?;
            let result = model.benchmark_value_paths(&board, args.iterations);
            let full_eps = result.iterations as f64 / result.policy_value_seconds.max(1e-9);
            let value_eps = result.iterations as f64 / result.value_seconds.max(1e-9);
            let update_eps = result.iterations as f64 / result.update_seconds.max(1e-9);
            println!("position : moves={}", board.move_count());
            println!(
                "full     : value={:.6} elapsed={:.3}s eval/s={full_eps:.0}",
                result.value, result.policy_value_seconds
            );
            println!(
                "value    : value={:.6} elapsed={:.3}s eval/s={value_eps:.0}",
                result.value, result.value_seconds
            );
            println!("compare  : speedup={:.2}x", value_eps / full_eps.max(1e-9));
            println!(
                "update   : elapsed={:.3}s updates/s={update_eps:.0}",
                result.update_seconds
            );
        }
        Some(Command::AzAlphaBeta(args)) => {
            let model = load_model(&args.model)?;
            let board = board_from_moves(&args.moves)?;
            let result = alpha_beta_search(
                &board,
                &model,
                AlphaBetaConfig {
                    max_depth: args.depth,
                    max_nodes: args.nodes,
                    threat_extension_depth: args.threat_depth,
                },
            );
            println!("{board}");
            println!(
                "pvs      : best={} value={:.4} depth={} nodes={} elapsed={:.3}s nodes/s={:.0}",
                result
                    .best_move
                    .map(|mv| mv.notation())
                    .unwrap_or_else(|| "none".into()),
                result.value,
                result.completed_depth,
                result.nodes,
                result.elapsed_seconds,
                result.nodes as f64 / result.elapsed_seconds.max(1e-9)
            );
            println!(
                "search   : tt_hits={} beta_cutoffs={} threat_nodes={} pv={}",
                result.tt_hits,
                result.beta_cutoffs,
                result.threat_nodes,
                result
                    .principal_variation
                    .iter()
                    .map(|mv| mv.notation())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        Some(Command::AzValueEval(args)) => {
            let model = load_model(&args.model)?;
            let samples = replay::load(&args.replay)?;
            let samples = &samples[..samples.len().min(args.limit)];
            if samples.is_empty() {
                return Err(io::Error::other("没有可评估的回放样本"));
            }
            let mut entrypoint_abs = 0.0_f64;
            let mut outcome_abs = 0.0_f64;
            let mut full_outcome_abs = 0.0_f64;
            let mut strong = 0usize;
            let mut signs = 0usize;
            for sample in samples {
                let (_, full) = model.evaluate(&sample.board);
                let value = model.evaluate_value(&sample.board);
                entrypoint_abs += (value - full).abs() as f64;
                outcome_abs += (value - sample.value).abs() as f64;
                full_outcome_abs += (full - sample.value).abs() as f64;
                if full.abs() >= 0.1 {
                    strong += 1;
                    signs += usize::from(value.signum() == full.signum());
                }
            }
            let count = samples.len() as f64;
            println!("samples  : {}", samples.len());
            println!("entrypoint: mae={:.6}", entrypoint_abs / count);
            println!(
                "outcome   : value_mae={:.6} combined_mae={:.6}",
                outcome_abs / count,
                full_outcome_abs / count
            );
            println!(
                "sign     : strong={} agreement={:.2}% threshold=0.1",
                strong,
                signs as f64 * 100.0 / strong.max(1) as f64
            );
        }
        Some(Command::AzLoop(args)) => {
            let (config, created) = load_or_create(&args.config)?;
            if created {
                println!("config   : 已生成 {}，请检查参数后再次运行", args.config);
            } else {
                az_loop::run(config, args.target_update)?;
            }
        }
        Some(Command::AzArenaBest(args)) => {
            let candidate = load_model(&args.candidate)?;
            let best = load_model(&args.best)?;
            println!("best-eval: candidate={} best={}", args.candidate, args.best);
            println!(
                "settings : games={} simulations={} cpuct={} opening_random_plies=2 workers={} confidence_z={}",
                args.games,
                args.simulations,
                args.cpuct,
                rayon::current_num_threads().min(args.games.max(1)),
                args.confidence_z
            );
            let started = Instant::now();
            let report = arena(
                &candidate,
                &best,
                args.games,
                SearchConfig {
                    simulations: args.simulations,
                    cpuct: args.cpuct,
                    opening_random_plies: 2,
                    opening_seed: 20260730,
                    ..Default::default()
                },
            );
            let seconds = started.elapsed().as_secs_f32();
            println!(
                "result   : W/L/D={}/{}/{} score={:.2}% stderr={:.2}% lower={:.2}% elo={:+.1} avg_plies={:.1}",
                report.wins,
                report.losses,
                report.draws,
                report.score_rate() * 100.0,
                report.score_rate_standard_error() * 100.0,
                report.score_rate_lower_bound(args.confidence_z) * 100.0,
                report.elo_diff(),
                report.plies as f32 / report.games().max(1) as f32
            );
            println!(
                "as-black : W/L/D={}/{}/{}",
                report.wins_as_black, report.losses_as_black, report.draws_as_black
            );
            println!(
                "as-white : W/L/D={}/{}/{}",
                report.wins_as_white, report.losses_as_white, report.draws_as_white
            );
            println!(
                "speed    : elapsed={:.2}s games/s={:.2}",
                seconds,
                report.games() as f32 / seconds.max(1e-6)
            );
        }
        Some(Command::AzEvalBest(args)) => {
            human_evaluate_best(
                &load_model(&args.best)?,
                &args.best,
                args.simulations,
                args.cpuct,
                args.human_side,
                !args.no_clear,
            )?;
        }
        Some(Command::Play(args)) => play(&load_model(&args.model)?)?,
    }
    gomoku::profile::print_report();
    Ok(())
}

fn load_model(path: &str) -> io::Result<PolicyValueModel> {
    if Path::new(path).exists() {
        PolicyValueModel::load(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("模型 `{path}` 不存在，请先运行 az-init"),
        ))
    }
}

fn board_from_moves(moves: &[String]) -> io::Result<Board> {
    let mut board = Board::new();
    for text in moves {
        let mv = Move::parse(text).ok_or_else(|| io::Error::other(format!("无效坐标 `{text}`")))?;
        if !board.play(mv) {
            return Err(io::Error::other(format!("非法落子 `{text}`")));
        }
    }
    Ok(board)
}

fn print_search(board: &Board, model: &PolicyValueModel, simulations: usize, cpuct: f32) {
    let started = Instant::now();
    let result = search(
        board,
        model,
        SearchConfig {
            simulations,
            cpuct,
            ..Default::default()
        },
    );
    println!(
        "search   : simulations={} elapsed={:.3}s",
        simulations,
        started.elapsed().as_secs_f32()
    );
    for c in result.into_iter().take(10) {
        println!(
            "candidate: {} visits={} q={:.3} prior={:.3}",
            c.mv.notation(),
            c.visits,
            c.q,
            c.prior
        );
    }
}

fn play(model: &PolicyValueModel) -> io::Result<()> {
    let mut board = Board::new();
    loop {
        println!("{board}");
        if let Some(outcome) = board.outcome() {
            println!("result   : {outcome:?}");
            return Ok(());
        }
        if board.to_move() == Player::Black {
            print!("你的落子（如 h8）：");
            io::stdout().flush()?;
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            if !Move::parse(&line).is_some_and(|mv| board.play(mv)) {
                println!("非法落子");
            }
        } else {
            let result = search(&board, model, SearchConfig::default());
            let Some(best) = result.first() else {
                return Ok(());
            };
            println!("AI 落子：{}", best.mv.notation());
            board.play(best.mv);
        }
    }
}

fn human_evaluate_best(
    model: &PolicyValueModel,
    model_path: &str,
    simulations: usize,
    cpuct: f32,
    human_side: HumanSide,
    clear_screen: bool,
) -> io::Result<()> {
    let human = match human_side {
        HumanSide::Black => Player::Black,
        HumanSide::White => Player::White,
    };
    let mut board = Board::new();
    let mut last_search: Option<(Move, u32, f32, f32, Vec<gomoku::mcts::Candidate>)> = None;
    loop {
        if clear_screen {
            print!("\x1b[2J\x1b[H");
        }
        println!("GomokuAI — 人工评估 Best");
        println!("model    : {model_path}");
        println!("players  : human={human:?}  best={:?}", human.other());
        println!("search   : simulations={simulations}  cpuct={cpuct}");
        if let Some((mv, visits, q, seconds, _)) = &last_search {
            println!(
                "lastmove : {}  visits={}  q={:.3}  time={:.3}s",
                mv.notation(),
                visits,
                q,
                seconds
            );
        } else {
            println!("lastmove : -");
        }
        println!();
        println!("{board}");
        if let Some(outcome) = board.outcome() {
            match outcome {
                Outcome::Draw => println!("evaluation: DRAW"),
                Outcome::Win(player) if player == human => println!("evaluation: HUMAN WIN"),
                Outcome::Win(_) => println!("evaluation: BEST WIN"),
            }
            return Ok(());
        }
        if board.to_move() == human {
            loop {
                print!("move> ");
                io::stdout().flush()?;
                let mut line = String::new();
                io::stdin().read_line(&mut line)?;
                let command = line.trim();
                if command.eq_ignore_ascii_case("quit") {
                    println!("evaluation: ABORTED");
                    return Ok(());
                }
                if command.eq_ignore_ascii_case("help") {
                    println!("commands : <坐标> 落子 | info 候选详情 | quit 退出");
                    continue;
                }
                if command.eq_ignore_ascii_case("info") {
                    if let Some((_, _, _, _, candidates)) = &last_search {
                        println!("rank move visits      q   prior");
                        for (rank, candidate) in candidates.iter().take(10).enumerate() {
                            println!(
                                "{:>4} {:>4} {:>6} {:+.3}  {:.4}",
                                rank + 1,
                                candidate.mv.notation(),
                                candidate.visits,
                                candidate.q,
                                candidate.prior
                            );
                        }
                    } else {
                        println!("info     : Best 尚未搜索");
                    }
                    continue;
                }
                if Move::parse(command).is_some_and(|mv| board.play(mv)) {
                    break;
                }
                println!("error    : 非法输入；输入 help 查看命令");
            }
            continue;
        }
        println!("best     : thinking...");
        io::stdout().flush()?;
        let started = Instant::now();
        let result = search(
            &board,
            model,
            SearchConfig {
                simulations,
                cpuct,
                ..Default::default()
            },
        );
        let Some(best) = result.first() else {
            println!("evaluation: DRAW");
            return Ok(());
        };
        let elapsed = started.elapsed().as_secs_f32();
        last_search = Some((best.mv, best.visits, best.q, elapsed, result.clone()));
        board.play(best.mv);
    }
}
