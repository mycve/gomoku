use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use gomoku::{
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
    #[arg(default_value_t = 400)]
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
    #[arg(default_value_t = 400)]
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
    #[arg(long, value_delimiter = ',')]
    gpu_devices: Vec<usize>,
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
    #[arg(default_value_t = 40)]
    games: usize,
    #[arg(default_value_t = 400)]
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
    #[arg(default_value_t = 800)]
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
                "arch     : input=451 hidden={} rmsnorm policy=225 value=96x96xWDL3 moves-left=96x1",
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
            let devices = candle_train::training_device_names(&args.gpu_devices)?;
            let stats = candle_train::train(
                &mut model,
                &samples,
                args.epochs,
                args.learning_rate,
                args.batch_size,
                &args.gpu_devices,
                0.1,
            )?;
            let seconds = started.elapsed().as_secs_f64();
            println!(
                "train    : samples={} epochs={} batch_size={} device={} elapsed={seconds:.3}s",
                samples.len(),
                args.epochs,
                args.batch_size,
                devices.join(",")
            );
            println!(
                "speed    : {:.0} samples/s",
                (samples.len() * args.epochs) as f64 / seconds.max(1e-9)
            );
            println!(
                "loss     : total={:.4} policy={:.4} value={:.4} moves_left={:.4}",
                stats.loss, stats.policy_loss, stats.value_loss, stats.moves_left_loss
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
                "settings : games={} simulations={} cpuct={} opening_random_plies=2 confidence_z={}",
                args.games, args.simulations, args.cpuct, args.confidence_z
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
                "result   : W/L/D={}/{}/{} score={:.2}% stderr={:.2}% lower={:.2}% elo={:+.1}",
                report.wins,
                report.losses,
                report.draws,
                report.score_rate() * 100.0,
                report.score_rate_standard_error() * 100.0,
                report.score_rate_lower_bound(args.confidence_z) * 100.0,
                report.elo_diff()
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
