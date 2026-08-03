use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use gomoku::{
    az_loop,
    az_loop_config::{DEFAULT_CONFIG_PATH, load_or_create},
    candle_train, distill,
    game::{Board, Move, Outcome, Player},
    mcts::{SearchConfig, search},
    model::PolicyValueModel,
    replay,
    selfplay::arena,
};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
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
    /// 使用 KataGo 标注的 NPZ 数据蒸馏模型。
    AzDistill(AzDistillArgs),
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
}

#[derive(Args)]
struct AzDistillArgs {
    /// fs15x_label28b/train 目录。
    #[arg(default_value = "katago-gomoku-distill-2025.5/fs15x_label28b/train")]
    data: String,
    /// 起始模型；文件不存在时随机初始化。
    #[arg(long, default_value = "model.safetensors")]
    model: String,
    #[arg(long, default_value = "distilled.safetensors")]
    output: String,
    /// 验证集目录；空字符串表示禁用验证。
    #[arg(
        long,
        default_value = "katago-gomoku-distill-2025.5/fs15x_label28b/val"
    )]
    validation: String,
    /// 验证损失最低时保存到这里。
    #[arg(long, default_value = "distilled-best.safetensors")]
    best_output: String,
    /// 每训练多少个分片验证一次。
    #[arg(long, default_value_t = 25)]
    validate_every: usize,
    #[arg(long, default_value_t = 192)]
    hidden: usize,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,
    /// 余弦衰减的最终学习率。
    #[arg(long, default_value_t = 0.00001)]
    min_learning_rate: f32,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    /// 最多处理多少个 NPZ；0 表示全部已下载分片。
    #[arg(long, default_value_t = 0)]
    max_files: usize,
    /// 跳过排序后的前 N 个已下载分片，便于分阶段续训。
    #[arg(long, default_value_t = 0)]
    skip_files: usize,
    /// 每个 NPZ 最多读取多少个样本；0 表示全部。
    #[arg(long, default_value_t = 0)]
    max_samples_per_file: usize,
    /// 最多加载的验证样本数；0 表示全部。按分片顺序均匀截取。
    #[arg(long, default_value_t = 200000)]
    validation_samples: usize,
    /// 已完成分片数的续跑状态文件；空字符串表示禁用。
    #[arg(long, default_value = "data/distill-progress.txt")]
    progress: String,
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
                "arch     : input=451 hidden={} rmsnorm local=4axesx8cells-pattern{} policy=225 value=96x96xWDL3",
                args.hidden,
                gomoku::model::LOCAL_AXIS_FEATURE_SIZE,
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
            let device = candle_train::training_device_name()?;
            let stats = candle_train::train(
                &mut model,
                &samples,
                args.epochs,
                args.learning_rate,
                args.batch_size,
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
        Some(Command::AzDistill(args)) => {
            if args.epochs == 0 {
                return Err(io::Error::other(
                    "epochs 必须大于 0，避免未训练却推进续跑游标",
                ));
            }
            if !args.learning_rate.is_finite()
                || !args.min_learning_rate.is_finite()
                || args.learning_rate <= 0.0
                || args.min_learning_rate <= 0.0
                || args.min_learning_rate > args.learning_rate
            {
                return Err(io::Error::other(
                    "学习率必须有限且满足 0 < min-learning-rate <= learning-rate",
                ));
            }
            let saved_progress = if args.progress.is_empty() {
                0
            } else {
                fs::read_to_string(&args.progress)
                    .ok()
                    .and_then(|text| text.trim().parse::<usize>().ok())
                    .unwrap_or(0)
            };
            let skip_files = args.skip_files.max(saved_progress);
            let resume_output = skip_files > 0 && Path::new(&args.output).exists();
            let mut model = if resume_output {
                PolicyValueModel::load(&args.output)?
            } else if Path::new(&args.model).exists() {
                PolicyValueModel::load(&args.model)?
            } else {
                PolicyValueModel::random(args.hidden, 20260801)
            };
            let all_files = distill::npz_files(&args.data)?
                .into_iter()
                .filter(|path| !distill::is_lfs_pointer(path).unwrap_or(false))
                .collect::<Vec<_>>();
            let total_files = all_files.len();
            let files = all_files
                .into_iter()
                .skip(skip_files)
                .take(if args.max_files == 0 {
                    usize::MAX
                } else {
                    args.max_files
                })
                .collect::<Vec<_>>();
            if files.is_empty() {
                return Err(io::Error::other(format!(
                    "{} 中没有已下载的 NPZ 实体（当前文件可能都是 Git LFS 占位符）",
                    args.data
                )));
            }
            let device = candle_train::training_device_name()?;
            let mut session = candle_train::TrainingSession::new(&model, None, args.learning_rate)?;
            println!(
                "distill  : files={} skip={} total={} device={} output={}",
                files.len(),
                skip_files,
                total_files,
                device,
                args.output
            );
            let validation = if args.validation.is_empty() {
                Vec::new()
            } else {
                let mut samples = Vec::new();
                let validation_files = distill::npz_files(&args.validation)?
                    .into_iter()
                    .filter(|path| !distill::is_lfs_pointer(path).unwrap_or(false))
                    .collect::<Vec<_>>();
                let per_file = if args.validation_samples == 0 {
                    usize::MAX
                } else {
                    args.validation_samples
                        .div_ceil(validation_files.len().max(1))
                };
                for path in validation_files {
                    samples.extend(distill::load_npz(path, Some(per_file))?);
                }
                if args.validation_samples > 0 {
                    samples.truncate(args.validation_samples);
                }
                samples
            };
            let mut best_validation_loss = f32::INFINITY;
            if !validation.is_empty() {
                if skip_files > 0 && Path::new(&args.best_output).exists() {
                    let best_model = PolicyValueModel::load(&args.best_output)?;
                    let best_session =
                        candle_train::TrainingSession::new(&best_model, None, args.learning_rate)?;
                    let best_stats = best_session.evaluate(&validation, args.batch_size)?;
                    best_validation_loss = best_stats.loss;
                    println!(
                        "best     : samples={} loss={:.4} policy={:.4} value={:.4} kl={:.4}/{:.4}",
                        validation.len(),
                        best_stats.loss,
                        best_stats.policy_loss,
                        best_stats.value_loss,
                        best_stats.policy_kl,
                        best_stats.value_kl,
                    );
                }
                let stats = session.evaluate(&validation, args.batch_size)?;
                let improved = stats.loss < best_validation_loss;
                if improved {
                    best_validation_loss = stats.loss;
                    save_model_retry(&model, &args.best_output)?;
                }
                println!(
                    "validate : samples={} loss={:.4} policy={:.4} value={:.4} kl={:.4}/{:.4}{}",
                    validation.len(),
                    stats.loss,
                    stats.policy_loss,
                    stats.value_loss,
                    stats.policy_kl,
                    stats.value_kl,
                    if improved { " [best]" } else { "" },
                );
            }
            let started = Instant::now();
            let mut total_samples = 0usize;
            let stop = Arc::new(AtomicBool::new(false));
            let stop_handler = Arc::clone(&stop);
            ctrlc::set_handler(move || stop_handler.store(true, Ordering::Relaxed))
                .map_err(|error| io::Error::other(error.to_string()))?;
            for (index, path) in files.iter().enumerate() {
                let limit = (args.max_samples_per_file > 0).then_some(args.max_samples_per_file);
                let (mut samples, load_stats) = distill::load_npz_with_stats(path, limit)?;
                distill::augment_and_shuffle(
                    &mut samples,
                    20260801_u64.wrapping_add((skip_files + index) as u64),
                );
                let progress = (skip_files + index) as f32 + 0.5;
                let progress = progress / total_files.max(1) as f32;
                let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                let learning_rate =
                    args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine;
                let stats = session.train_controlled(
                    &mut model,
                    None,
                    &samples,
                    args.epochs,
                    learning_rate,
                    args.batch_size,
                    1.0,
                    Some(&stop),
                )?;
                total_samples += samples.len();
                save_model_retry(&model, &args.output)?;
                let complete = stats.samples == samples.len().saturating_mul(args.epochs);
                if complete && !args.progress.is_empty() {
                    save_distill_progress(&args.progress, skip_files + index + 1)?;
                }
                println!(
                    "file     : {}/{} {} samples={}/{} lr={:.2e} loss={:.4} policy={:.4} value={:.4} mass={:.1}% top1={:.1}%",
                    index + 1,
                    files.len(),
                    path.display(),
                    samples.len(),
                    load_stats.rows,
                    learning_rate,
                    stats.loss,
                    stats.policy_loss,
                    stats.value_loss,
                    100.0 * load_stats.policy_mass_retention(),
                    100.0 * load_stats.top1_retention(),
                );
                if !complete || stop.load(Ordering::Relaxed) {
                    println!(
                        "stopped  : 当前分片完成 {}/{} 个样本轮次，模型已保存，续跑游标未越过未完成分片",
                        stats.samples,
                        samples.len().saturating_mul(args.epochs),
                    );
                    break;
                }
                let should_validate = !validation.is_empty()
                    && ((index + 1) % args.validate_every.max(1) == 0 || index + 1 == files.len());
                if should_validate {
                    let stats = session.evaluate(&validation, args.batch_size)?;
                    let improved = stats.loss < best_validation_loss;
                    if improved {
                        best_validation_loss = stats.loss;
                        save_model_retry(&model, &args.best_output)?;
                    }
                    println!(
                        "validate : samples={} loss={:.4} policy={:.4} value={:.4} kl={:.4}/{:.4}{}",
                        validation.len(),
                        stats.loss,
                        stats.policy_loss,
                        stats.value_loss,
                        stats.policy_kl,
                        stats.value_kl,
                        if improved { " [best]" } else { "" }
                    );
                }
            }
            println!(
                "complete : samples={} elapsed={:.1}s model={}",
                total_samples,
                started.elapsed().as_secs_f64(),
                args.output
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

fn save_distill_progress(path: impl AsRef<Path>, completed: usize) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&temporary, completed.to_string())?;
    let mut last_error = None;
    for attempt in 0..20 {
        match fs::rename(&temporary, path) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt < 19 {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            }
        }
    }
    Err(last_error.expect("替换进度文件至少尝试一次"))
}

fn save_model_retry(model: &PolicyValueModel, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    let mut last_error = None;
    for attempt in 0..20 {
        match model.save(path) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt < 19 {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            }
        }
    }
    Err(last_error.expect("保存模型至少尝试一次"))
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
