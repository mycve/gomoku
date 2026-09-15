use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use crossterm::{
    cursor::MoveTo,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};
use go19::{
    az_loop,
    az_loop_config::{DEFAULT_CONFIG_PATH, load_or_create},
    candle_train,
    game::{BOARD_SIZE, Board, Move, Outcome, Player},
    mcts::{Candidate, SearchConfig, search},
    model::{
        INPUT_SIZE, LOCAL_AXIS_FEATURE_SIZE, POLICY_HEAD_SIZE, PolicyValueModel,
        REGION_FEATURE_SIZE, ROLE_ADAPTER_RANK, VALUE_HEAD_SIZE,
    },
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
    name = "go19",
    version,
    about = "19x19 Go policy/value search and training tools"
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
    /// 按 TOML 配置持续执行自博弈训练。
    AzLoop(AzLoopArgs),
    /// 人工在控制台挑战 Best 模型。
    AzEvalBest(AzEvalBestArgs),
    /// 自动评估候选模型相对 Best 模型的表现。
    AzArenaBest(AzArenaBestArgs),
    /// 方向键终端人机对战。
    Play(PlayArgs),
    /// 通过标准输入输出运行 GTP 2 围棋引擎。
    Gtp(GtpArgs),
}

#[derive(Args)]
struct GtpArgs {
    #[arg(long, default_value = "go19-v32-model.safetensors")]
    model: String,
    #[arg(long, default_value_t = 256)]
    simulations: usize,
}

#[derive(Args)]
struct AzInitArgs {
    #[arg(default_value = "go19-v32-model.safetensors")]
    output: String,
    #[arg(default_value_t = 128)]
    hidden: usize,
    #[arg(default_value_t = 20260730)]
    seed: u64,
}

#[derive(Args)]
struct AzSearchArgs {
    #[arg(default_value = "go19-v32-model.safetensors")]
    model: String,
    #[arg(default_value_t = 3000)]
    simulations: usize,
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    /// 已落子坐标序列，例如 e5 e6 f5。
    moves: Vec<String>,
}

#[derive(Args)]
struct AzBenchArgs {
    #[arg(default_value = "go19-v32-model.safetensors")]
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
    #[arg(default_value = "go19-v32-model.safetensors")]
    model: String,
    #[arg(default_value = "data/go19-v32/replay.jsonl")]
    replay: String,
    #[arg(default_value_t = 2)]
    epochs: usize,
    #[arg(default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(default_value_t = 256)]
    batch_size: usize,
}

#[derive(Args)]
struct AzLoopArgs {
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: String,
    /// 在完成该绝对更新编号后停止。
    #[arg(long)]
    target_update: Option<usize>,
}

#[derive(Args)]
struct AzArenaBestArgs {
    #[arg(default_value = "go19-v32-model.safetensors")]
    candidate: String,
    #[arg(default_value = "go19-v32-best.safetensors")]
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
    #[arg(default_value = "go19-v32-best.safetensors")]
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
    #[arg(default_value = "go19-v32-model.safetensors")]
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
                "arch     : input={} hidden={} rmsnorm role=rank{} region=3x3x{} local=4axesx8cells-pattern{} policy=dynamic{} value={}x{}x1(sigmoid)",
                INPUT_SIZE,
                args.hidden,
                ROLE_ADAPTER_RANK,
                REGION_FEATURE_SIZE,
                LOCAL_AXIS_FEATURE_SIZE,
                POLICY_HEAD_SIZE,
                VALUE_HEAD_SIZE,
                VALUE_HEAD_SIZE,
            );
            println!("board    : 19x19 Go, area scoring, komi 7.5");
        }
        Some(Command::Gtp(args)) => {
            if args.simulations == 0 {
                return Err(io::Error::other("simulations 必须大于 0"));
            }
            let model = load_model(&args.model)?;
            go19::gtp::Engine::new(
                &model,
                SearchConfig {
                    simulations: args.simulations,
                    ..Default::default()
                },
            )
            .run(io::stdin().lock(), io::stdout().lock())?;
            return Ok(());
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
                "result   : W/L/D={}/{}/{} aborted={} aborted_as_loss=true score={:.2}% stderr={:.2}% lower={:.2}% elo={:+.1} avg_plies={:.1}",
                report.wins,
                report.losses,
                report.draws,
                report.aborted,
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
    go19::profile::print_report();
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
    write!(output, "{title}\r\n")?;
    for line in details {
        write!(output, "{line}\r\n")?;
    }
    write!(
        output,
        "turn     : {:?} | pass={} | 黑方净面积={:+.1}（贴目7.5）\r\n",
        board.to_move(),
        board.consecutive_passes(),
        board.raw_score()
    )?;
    write!(output, "          ")?;
    for col in 0..go19::game::BOARD_SIZE {
        write!(
            output,
            " {} ",
            go19::game::COORDINATES.as_bytes()[col] as char
        )?;
    }
    write!(output, "\r\n")?;
    for row in (0..go19::game::BOARD_SIZE).rev() {
        write!(output, "{:>3}       ", row + 1)?;
        for col in 0..go19::game::BOARD_SIZE {
            let stone = match board.cells()[row * go19::game::BOARD_SIZE + col] {
                1 => 'X',
                -1 => 'O',
                _ => '.',
            };
            if cursor == (row, col) {
                write!(output, "[{stone}]")?;
            } else {
                write!(output, " {stone} ")?;
            }
        }
        write!(output, "\r\n")?;
    }
    write!(output, "\r\n")?;
    print_candidates(&mut output, candidates)?;
    write!(output, "\r\n")?;
    if editable {
        write!(
            output,
            "keys     : 方向键移动 Enter落子 P停一手 Backspace撤销 R清盘 Q退出\r\n"
        )?;
    } else {
        write!(output, "keys     : 方向键移动 Enter落子 P停一手 Q退出\r\n")?;
    }
    output.flush()
}

fn print_candidates(output: &mut impl Write, candidates: &[Candidate]) -> io::Result<()> {
    if candidates.is_empty() {
        return write!(output, "candidates: -\r\n");
    }
    let total = candidates
        .iter()
        .map(|candidate| candidate.visits)
        .sum::<u32>()
        .max(1) as f32;
    write!(output, "rank move   mcts%  prior%       q visits\r\n")?;
    for (rank, candidate) in candidates.iter().take(12).enumerate() {
        write!(
            output,
            "{:>4} {:>4} {:>7.2} {:>7.2} {:+.4} {:>6}\r\n",
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
            KeyCode::Up => cursor.0 = (cursor.0 + 1).min(go19::game::BOARD_SIZE - 1),
            KeyCode::Down => cursor.0 = cursor.0.saturating_sub(1),
            KeyCode::Left => cursor.1 = cursor.1.saturating_sub(1),
            KeyCode::Right => cursor.1 = (cursor.1 + 1).min(go19::game::BOARD_SIZE - 1),
            KeyCode::Enter => {
                let mv = Move::new(cursor.0, cursor.1).expect("光标始终位于棋盘内");
                if board.is_legal(mv) {
                    return Ok(BoardAction::Place(mv));
                }
            }
            KeyCode::Char('p' | 'P') if board.is_legal(Move::PASS) => {
                return Ok(BoardAction::Place(Move::PASS));
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
    let mut cursor = (BOARD_SIZE / 2, BOARD_SIZE / 2);
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
            "Go19 — 交互式局面搜索",
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
    let mut cursor = (BOARD_SIZE / 2, BOARD_SIZE / 2);
    let mut last_search = Vec::new();
    let mut last_seconds = 0.0;
    let _raw = RawModeGuard::enter()?;
    loop {
        if let Some(outcome) = board.outcome() {
            let result = match outcome {
                Outcome::Draw => "DRAW",
                Outcome::Aborted => "ABORTED (no score)",
                Outcome::Win(player) if player == human => "HUMAN WIN",
                Outcome::Win(_) => "MODEL WIN",
            };
            let analysis = go19::scoring::analyze(&board);
            render_interactive_board(
                &board,
                cursor,
                "Go19 — 方向键人机对弈",
                &[
                    format!("result   : {result}"),
                    format!(
                        "score    : {:+.1} | dead={} seki={} unsettled={}（未定棋保留，面积估分）",
                        analysis.score,
                        analysis.dead.len(),
                        analysis.seki.len(),
                        analysis.unsettled.len()
                    ),
                ],
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
                "Go19 — 方向键人机对弈",
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
