use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use crossterm::{
    cursor::MoveTo,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};
use gomoku::{
    az_loop,
    az_loop_config::{DEFAULT_CONFIG_PATH, load_or_create},
    candle_train, distill,
    game::{Board, Move, Outcome, Player},
    mcts::{Candidate, SearchConfig, search},
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
    /// 方向键交互摆局并显示候选着概率。
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
    /// 方向键终端人机对战。
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
}

#[derive(Args)]
struct PlayArgs {
    #[arg(default_value = "model.safetensors")]
    model: String,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    #[arg(long, value_enum, default_value_t = HumanSide::Black)]
    human_side: HumanSide,
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
                "arch     : input=451 hidden={} rmsnorm local=4axesx8cells-pattern{} policy=global+local-gate->relu64->225 value=96x96xWDL3",
                args.hidden,
                gomoku::model::LOCAL_AXIS_FEATURE_SIZE,
            );
            println!("board    : 15x15 freestyle gomoku");
        }
        Some(Command::AzSearch(args)) => {
            let model = load_model(&args.model)?;
            interactive_search(&model, &args.moves, args.simulations, args.cpuct)?;
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
            )?;
        }
        Some(Command::Play(args)) => human_evaluate_best(
            &load_model(&args.model)?,
            &args.model,
            args.simulations,
            args.cpuct,
            args.human_side,
        )?,
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

struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

enum BoardAction {
    Place(Move),
    Undo,
    Reset,
    Quit,
}

fn render_interactive_board(
    board: &Board,
    cursor: (usize, usize),
    title: &str,
    details: &[String],
    candidates: &[Candidate],
    editable: bool,
) -> io::Result<()> {
    let mut output = io::stdout();
    execute!(output, MoveTo(0, 0), Clear(ClearType::All))?;
    writeln!(output, "{title}")?;
    for line in details {
        writeln!(output, "{line}")?;
    }
    writeln!(output, "turn     : {:?}", board.to_move())?;
    write!(output, "          ")?;
    for col in 0..gomoku::game::BOARD_SIZE {
        write!(output, " {} ", (b'a' + col as u8) as char)?;
    }
    writeln!(output)?;
    for row in 0..gomoku::game::BOARD_SIZE {
        write!(output, "{:>3}       ", row + 1)?;
        for col in 0..gomoku::game::BOARD_SIZE {
            let stone = match board.cells()[row * gomoku::game::BOARD_SIZE + col] {
                1 => '●',
                -1 => '○',
                _ => '·',
            };
            if cursor == (row, col) {
                write!(output, "[{stone}]")?;
            } else {
                write!(output, " {stone} ")?;
            }
        }
        writeln!(output)?;
    }
    writeln!(output)?;
    print_candidates(&mut output, candidates)?;
    writeln!(output)?;
    if editable {
        writeln!(
            output,
            "keys     : 方向键移动 Enter落子 Backspace撤销 R清盘 Q退出"
        )?;
    } else {
        writeln!(output, "keys     : 方向键移动 Enter落子 Q退出")?;
    }
    output.flush()
}

fn print_candidates(output: &mut impl Write, candidates: &[Candidate]) -> io::Result<()> {
    if candidates.is_empty() {
        return writeln!(output, "candidates: -");
    }
    let total = candidates
        .iter()
        .map(|candidate| candidate.visits)
        .sum::<u32>()
        .max(1) as f32;
    writeln!(output, "rank move   mcts%  prior%       q visits")?;
    for (rank, candidate) in candidates.iter().take(12).enumerate() {
        writeln!(
            output,
            "{:>4} {:>4} {:>7.2} {:>7.2} {:+.4} {:>6}",
            rank + 1,
            candidate.mv.notation(),
            candidate.visits as f32 * 100.0 / total,
            candidate.prior * 100.0,
            candidate.q,
            candidate.visits,
        )?;
    }
    Ok(())
}

fn read_board_action(
    board: &Board,
    cursor: &mut (usize, usize),
    title: &str,
    details: &[String],
    candidates: &[Candidate],
    editable: bool,
) -> io::Result<BoardAction> {
    loop {
        render_interactive_board(board, *cursor, title, details, candidates, editable)?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        match key.code {
            KeyCode::Up => cursor.0 = cursor.0.saturating_sub(1),
            KeyCode::Down => cursor.0 = (cursor.0 + 1).min(gomoku::game::BOARD_SIZE - 1),
            KeyCode::Left => cursor.1 = cursor.1.saturating_sub(1),
            KeyCode::Right => cursor.1 = (cursor.1 + 1).min(gomoku::game::BOARD_SIZE - 1),
            KeyCode::Enter => {
                let mv = Move::new(cursor.0, cursor.1).expect("光标始终位于棋盘内");
                if board.is_legal(mv) {
                    return Ok(BoardAction::Place(mv));
                }
            }
            KeyCode::Backspace if editable => return Ok(BoardAction::Undo),
            KeyCode::Char('r' | 'R') if editable => return Ok(BoardAction::Reset),
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => return Ok(BoardAction::Quit),
            _ => {}
        }
    }
}

fn run_search(
    board: &Board,
    model: &PolicyValueModel,
    simulations: usize,
    cpuct: f32,
) -> (Vec<Candidate>, f32) {
    let started = Instant::now();
    let candidates = search(
        board,
        model,
        SearchConfig {
            simulations,
            cpuct,
            ..Default::default()
        },
    );
    (candidates, started.elapsed().as_secs_f32())
}

fn interactive_search(
    model: &PolicyValueModel,
    initial_moves: &[String],
    simulations: usize,
    cpuct: f32,
) -> io::Result<()> {
    let mut history = initial_moves
        .iter()
        .map(|text| Move::parse(text).ok_or_else(|| io::Error::other(format!("无效坐标 `{text}`"))))
        .collect::<io::Result<Vec<_>>>()?;
    let mut board = board_from_move_values(&history)?;
    let mut cursor = (7, 7);
    let _raw = RawModeGuard::enter()?;
    loop {
        let (candidates, seconds) = if board.outcome().is_none() {
            run_search(&board, model, simulations, cpuct)
        } else {
            (Vec::new(), 0.0)
        };
        let details = vec![
            format!("search   : simulations={simulations} cpuct={cpuct:.2} time={seconds:.3}s"),
            format!("result   : {:?}", board.outcome()),
        ];
        match read_board_action(
            &board,
            &mut cursor,
            "GomokuAI — 交互式局面搜索",
            &details,
            &candidates,
            true,
        )? {
            BoardAction::Place(mv) => {
                history.push(mv);
                board.play(mv);
            }
            BoardAction::Undo => {
                history.pop();
                board = board_from_move_values(&history)?;
            }
            BoardAction::Reset => {
                history.clear();
                board = Board::new();
            }
            BoardAction::Quit => return Ok(()),
        }
    }
}

fn board_from_move_values(moves: &[Move]) -> io::Result<Board> {
    let mut board = Board::new();
    for &mv in moves {
        if !board.play(mv) {
            return Err(io::Error::other(format!("非法落子 `{}`", mv.notation())));
        }
    }
    Ok(board)
}

fn human_evaluate_best(
    model: &PolicyValueModel,
    model_path: &str,
    simulations: usize,
    cpuct: f32,
    human_side: HumanSide,
) -> io::Result<()> {
    let human = match human_side {
        HumanSide::Black => Player::Black,
        HumanSide::White => Player::White,
    };
    let mut board = Board::new();
    let mut cursor = (7, 7);
    let mut last_search = Vec::new();
    let mut last_seconds = 0.0;
    let _raw = RawModeGuard::enter()?;
    loop {
        if let Some(outcome) = board.outcome() {
            let result = match outcome {
                Outcome::Draw => "DRAW",
                Outcome::Win(player) if player == human => "HUMAN WIN",
                Outcome::Win(_) => "MODEL WIN",
            };
            render_interactive_board(
                &board,
                cursor,
                "GomokuAI — 方向键人机对弈",
                &[format!("result   : {result}")],
                &last_search,
                false,
            )?;
            return Ok(());
        }
        if board.to_move() == human {
            let details = vec![
                format!("model    : {model_path}"),
                format!("players  : human={human:?} model={:?}", human.other()),
                format!(
                    "search   : simulations={simulations} cpuct={cpuct:.2} last={last_seconds:.3}s"
                ),
            ];
            match read_board_action(
                &board,
                &mut cursor,
                "GomokuAI — 方向键人机对弈",
                &details,
                &last_search,
                false,
            )? {
                BoardAction::Place(mv) => {
                    board.play(mv);
                }
                BoardAction::Quit => return Ok(()),
                BoardAction::Undo | BoardAction::Reset => unreachable!(),
            }
            continue;
        }
        let (result, seconds) = run_search(&board, model, simulations, cpuct);
        let Some(best) = result.first() else {
            return Ok(());
        };
        let mv = best.mv;
        last_seconds = seconds;
        last_search = result;
        board.play(mv);
    }
}
