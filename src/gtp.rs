//! GTP 2：stdout 仅包含协议响应；诊断信息由调用者写入 stderr。
use crate::{
    game::{Board, Move, Player},
    mcts::{SearchConfig, search, search_timed},
    model::PolicyValueModel,
    scoring,
};
use std::{
    io::{self, BufRead, Write},
    time::{Duration, Instant},
};

const COMMANDS: &[&str] = &[
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "reg_genmove",
    "undo",
    "showboard",
    "final_score",
    "final_status_list",
    "estimate_score",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "set_free_handicap",
];

#[derive(Clone, Copy)]
struct Clock {
    remaining: f64,
    stones: usize,
}

pub struct Engine<'a> {
    board: Board,
    undo: Vec<Board>,
    model: &'a PolicyValueModel,
    config: SearchConfig,
    clocks: Option<[Clock; 2]>,
    time_settings: [usize; 3],
}

impl<'a> Engine<'a> {
    pub fn new(model: &'a PolicyValueModel, config: SearchConfig) -> Self {
        Self {
            board: Board::new(),
            undo: Vec::new(),
            model,
            config,
            clocks: None,
            time_settings: [0; 3],
        }
    }
    fn command(&mut self, command: &str, args: &[&str]) -> Result<String, String> {
        let empty = || Ok(String::new());
        match command {
            "protocol_version" => {
                arity(args, 0)?;
                Ok("2".into())
            }
            "name" => {
                arity(args, 0)?;
                Ok("Go19".into())
            }
            "version" => {
                arity(args, 0)?;
                Ok(env!("CARGO_PKG_VERSION").into())
            }
            "known_command" => {
                arity(args, 1)?;
                Ok(COMMANDS.contains(&args[0]).to_string())
            }
            "list_commands" => {
                arity(args, 0)?;
                Ok(COMMANDS.join("\n"))
            }
            "quit" => {
                arity(args, 0)?;
                empty()
            }
            "boardsize" | "clear_board" => {
                if command == "boardsize" {
                    arity(args, 1)?;
                    if integer(args[0])? != crate::game::BOARD_SIZE {
                        return Err("unacceptable size".into());
                    }
                } else {
                    arity(args, 0)?;
                }
                let komi = self.board.komi();
                self.board = Board::new();
                self.board.set_komi(komi);
                self.undo.clear();
                self.reset_clocks();
                empty()
            }
            "komi" => {
                arity(args, 1)?;
                let value = args[0].parse::<f32>().map_err(|_| "invalid komi")?;
                if !self.board.set_komi(value) {
                    return Err("invalid komi".into());
                }
                empty()
            }
            "play" => {
                arity(args, 2)?;
                let color = color(args[0])?;
                let mv = Move::parse(args[1]).ok_or("invalid vertex")?;
                let mut board = if self.board.to_move() == color && !self.board.is_finished() {
                    self.board.clone()
                } else {
                    self.board.for_turn(color)
                };
                if !board.play(mv) {
                    return Err("illegal move".into());
                }
                self.undo.push(self.board.clone());
                self.board = board;
                empty()
            }
            "genmove" | "reg_genmove" => {
                arity(args, 1)?;
                let color = color(args[0])?;
                let mut board = if color == self.board.to_move() {
                    self.board.clone()
                } else {
                    self.board.for_turn(color)
                };
                if board.is_finished() {
                    return Ok("pass".into());
                }
                let started = Instant::now();
                let candidates = if let Some(clocks) = self.clocks {
                    let clock = clocks[side(color)];
                    let divisor = if clock.stones > 0 {
                        clock.stones as f64
                    } else {
                        30.0
                    };
                    let seconds = (clock.remaining / divisor * 0.9).clamp(0.001, 60.0);
                    search_timed(
                        &board,
                        self.model,
                        self.config,
                        Duration::from_secs_f64(seconds),
                    )
                } else {
                    search(&board, self.model, self.config)
                };
                let mv = candidates.first().map_or(Move::PASS, |c| c.mv);
                if command == "genmove" {
                    if !board.play(mv) {
                        return Err("cannot generate move".into());
                    }
                    self.undo.push(self.board.clone());
                    self.board = board;
                    self.consume_time(color, started.elapsed().as_secs_f64());
                }
                Ok(if mv == Move::PASS {
                    "pass".into()
                } else {
                    mv.notation().to_ascii_uppercase()
                })
            }
            "undo" => {
                arity(args, 0)?;
                let mut board = self.undo.pop().ok_or("cannot undo")?;
                board.set_komi(self.board.komi());
                self.board = board;
                empty()
            }
            "showboard" => {
                arity(args, 0)?;
                Ok(format!("\n{}", self.board).trim_end().into())
            }
            "estimate_score" | "final_score" => {
                arity(args, 0)?;
                let analysis = scoring::analyze(&self.board);
                if command == "final_score"
                    && (!self.board.is_finished()
                        || !analysis.unsettled.is_empty()
                        || self.board.move_count() >= crate::game::MAX_MOVES)
                {
                    return Err("cannot score".into());
                }
                Ok(score_text(analysis.score))
            }
            "final_status_list" => {
                arity(args, 1)?;
                let analysis = scoring::analyze(&self.board);
                let groups = match args[0] {
                    "dead" => analysis.dead,
                    "unsettled" => analysis.unsettled,
                    "alive" if analysis.unsettled.is_empty() => analysis.alive,
                    "seki" if analysis.unsettled.is_empty() => analysis.seki,
                    "alive" | "seki" => return Err("cannot determine status".into()),
                    _ => return Err("invalid status".into()),
                };
                Ok(groups
                    .iter()
                    .map(|group| {
                        group
                            .iter()
                            .map(|mv| mv.notation().to_ascii_uppercase())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "time_settings" => {
                arity(args, 3)?;
                self.time_settings = [integer(args[0])?, integer(args[1])?, integer(args[2])?];
                self.reset_clocks();
                empty()
            }
            "time_left" => {
                arity(args, 3)?;
                let player = color(args[0])?;
                let clock = Clock {
                    remaining: integer(args[1])? as f64,
                    stones: integer(args[2])?,
                };
                let clocks = self.clocks.get_or_insert(
                    [Clock {
                        remaining: 0.0,
                        stones: 0,
                    }; 2],
                );
                clocks[side(player)] = clock;
                empty()
            }
            "fixed_handicap" | "set_free_handicap" => {
                if self.board.cells().iter().any(|&s| s != 0) || self.board.move_count() != 0 {
                    return Err("board not empty".into());
                }
                let points = if command == "fixed_handicap" {
                    arity(args, 1)?;
                    let count = integer(args[0])?;
                    if !(2..=5).contains(&count) {
                        return Err("invalid handicap".into());
                    }
                    ["c3", "g7", "c7", "g3", "e5"][..count]
                        .iter()
                        .map(|s| Move::parse(s).unwrap())
                        .collect::<Vec<_>>()
                } else {
                    if args.len() < 2 || args.len() > 40 {
                        return Err("invalid handicap".into());
                    }
                    args.iter()
                        .map(|s| {
                            Move::parse(s)
                                .filter(|&m| m != Move::PASS)
                                .ok_or("invalid vertex".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let stones = points
                    .iter()
                    .map(|&mv| (mv, Player::Black))
                    .collect::<Vec<_>>();
                let mut board =
                    Board::from_position(&stones, Player::White).ok_or("invalid handicap")?;
                board.set_komi(self.board.komi());
                self.board = board;
                self.undo.clear();
                if command == "fixed_handicap" {
                    Ok(points
                        .iter()
                        .map(|m| m.notation().to_ascii_uppercase())
                        .collect::<Vec<_>>()
                        .join(" "))
                } else {
                    empty()
                }
            }
            _ => Err("unknown command".into()),
        }
    }
    fn reset_clocks(&mut self) {
        let [main, byo, stones] = self.time_settings;
        self.clocks = if (byo > 0 && stones == 0) || (main == 0 && byo == 0 && stones == 0) {
            None
        } else {
            Some(
                [if main > 0 {
                    Clock {
                        remaining: main as f64,
                        stones: 0,
                    }
                } else {
                    Clock {
                        remaining: byo as f64,
                        stones,
                    }
                }; 2],
            )
        };
    }
    fn consume_time(&mut self, player: Player, elapsed: f64) {
        if let Some(clocks) = &mut self.clocks {
            let clock = &mut clocks[side(player)];
            let overtime = clock.stones > 0;
            clock.remaining = (clock.remaining - elapsed).max(0.0);
            if overtime {
                clock.stones -= 1;
            }
            if ((overtime && clock.stones == 0) || (!overtime && clock.remaining == 0.0))
                && self.time_settings[1] > 0
            {
                *clock = Clock {
                    remaining: self.time_settings[1] as f64,
                    stones: self.time_settings[2],
                };
            }
        }
    }
    pub fn run(&mut self, reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
        for line in reader.lines() {
            let line = line?;
            let clean = line
                .chars()
                .filter(|&c| c == '\t' || !c.is_control())
                .collect::<String>();
            let mut parts = clean.split('#').next().unwrap_or("").split_whitespace();
            let Some(first) = parts.next() else {
                continue;
            };
            let (id, command) = if first.bytes().all(|b| b.is_ascii_digit()) {
                (first, parts.next().unwrap_or(""))
            } else {
                ("", first)
            };
            let args = parts.collect::<Vec<_>>();
            let result = self.command(command, &args);
            let quit = command == "quit" && result.is_ok();
            match result {
                Ok(body) => {
                    write!(writer, "={id}")?;
                    if !body.is_empty() {
                        write!(writer, " {body}")?;
                    }
                    writeln!(writer, "\n")?;
                }
                Err(error) => writeln!(writer, "?{id} {error}\n")?,
            }
            writer.flush()?;
            if quit {
                break;
            }
        }
        Ok(())
    }
}
fn arity(args: &[&str], count: usize) -> Result<(), String> {
    if args.len() == count {
        Ok(())
    } else {
        Err("syntax error".into())
    }
}
fn integer(s: &str) -> Result<usize, String> {
    s.parse().map_err(|_| "invalid integer".into())
}
fn color(s: &str) -> Result<Player, String> {
    match s.to_ascii_lowercase().as_str() {
        "b" | "black" => Ok(Player::Black),
        "w" | "white" => Ok(Player::White),
        _ => Err("invalid color".into()),
    }
}
fn side(player: Player) -> usize {
    usize::from(player == Player::White)
}
fn score_text(score: f32) -> String {
    if score == 0.0 {
        "0".into()
    } else {
        format!("{}+{}", if score > 0.0 { "B" } else { "W" }, score.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> SearchConfig {
        SearchConfig {
            simulations: 4,
            ..Default::default()
        }
    }
    #[test]
    fn framing_comments_ids_errors_and_quit_are_protocol_only() {
        let model = PolicyValueModel::random(8, 1);
        let mut engine = Engine::new(&model, config());
        let mut output = Vec::new();
        engine.run(io::Cursor::new("# comment\r\n1\tprotocol_version\r\n2 boardsize 9\n3 known_command play\n4 nonsense\n5 quit extra\n6 quit\n7 name\n"), &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "=1 2\n\n?2 unacceptable size\n\n=3 true\n\n?4 unknown command\n\n?5 syntax error\n\n=6\n\n"
        );
    }
    #[test]
    fn captures_passes_out_of_turn_play_and_undo_preserve_state() {
        let model = PolicyValueModel::random(8, 2);
        let mut engine = Engine::new(&model, config());
        engine.board = crate::scoring::tests::enclosed_dead();
        let before = engine.board.clone();
        engine.command("play", &["b", "C5"]).unwrap();
        assert_eq!(engine.board.cells()[Move::parse("c4").unwrap().0], 0);
        engine.command("komi", &["6.5"]).unwrap();
        engine.command("undo", &[]).unwrap();
        assert_eq!(engine.board.cells(), before.cells());
        assert_eq!(engine.board.komi(), 6.5);
        let unchanged = engine.board.clone();
        assert!(engine.command("play", &["white", "z99"]).is_err());
        assert!(engine.command("komi", &["NaN"]).is_err());
        assert_eq!(engine.board, unchanged);
        engine.command("clear_board", &[]).unwrap();
        engine.command("play", &["w", "D4"]).unwrap();
        assert_eq!(engine.board.cells()[Move::parse("d4").unwrap().0], -1);
        engine.command("clear_board", &[]).unwrap();
        engine.command("play", &["b", "pass"]).unwrap();
        engine.command("play", &["w", "pass"]).unwrap();
        assert_eq!(engine.command("final_score", &[]).unwrap(), "W+6.5");
        engine.command("undo", &[]).unwrap();
        assert_eq!(engine.board.consecutive_passes(), 1);
    }
    #[test]
    fn scoring_reports_dead_and_refuses_unsettled_final_claims() {
        let model = PolicyValueModel::random(8, 3);
        let mut engine = Engine::new(&model, config());
        engine.board = crate::scoring::tests::enclosed_dead();
        engine.command("play", &["b", "pass"]).unwrap();
        engine.command("play", &["w", "pass"]).unwrap();
        assert_eq!(
            engine.command("final_status_list", &["dead"]).unwrap(),
            "C4"
        );
        assert_eq!(engine.command("final_score", &[]).unwrap(), "B+353.5");
        engine.command("clear_board", &[]).unwrap();
        engine.command("play", &["b", "e5"]).unwrap();
        engine.command("play", &["w", "pass"]).unwrap();
        engine.command("play", &["b", "pass"]).unwrap();
        assert_eq!(
            engine.command("final_score", &[]).unwrap_err(),
            "cannot score"
        );
        assert_eq!(
            engine.command("final_status_list", &["unsettled"]).unwrap(),
            "E5"
        );
    }
    #[test]
    fn generated_move_is_legal_and_regression_does_not_change_board() {
        let model = PolicyValueModel::random(8, 4);
        let mut engine = Engine::new(&model, config());
        engine.command("fixed_handicap", &["2"]).unwrap();
        assert_eq!(engine.board.to_move(), Player::White);
        let before = engine.board.clone();
        let predicted = engine.command("reg_genmove", &["w"]).unwrap();
        assert_eq!(engine.board, before);
        assert!(before.is_legal(Move::parse(&predicted).unwrap()));
        let generated = engine.command("genmove", &["w"]).unwrap();
        assert_eq!(generated, predicted);
        engine.command("undo", &[]).unwrap();
        assert_eq!(engine.board, before);
    }
    #[test]
    fn clock_enters_and_renews_canadian_overtime() {
        let model = PolicyValueModel::random(8, 5);
        let mut engine = Engine::new(&model, config());
        engine.command("time_settings", &["10", "5", "2"]).unwrap();
        engine.consume_time(Player::Black, 1.0);
        assert_eq!(engine.clocks.unwrap()[0].remaining, 9.0);
        engine.consume_time(Player::Black, 9.0);
        assert_eq!(engine.clocks.unwrap()[0].stones, 2);
        engine.consume_time(Player::Black, 0.5);
        engine.consume_time(Player::Black, 0.5);
        assert_eq!(engine.clocks.unwrap()[0].remaining, 5.0);
        assert_eq!(engine.clocks.unwrap()[0].stones, 2);
    }

    #[test]
    fn ko_retake_is_rejected_and_undo_restores_ko_history() {
        let model = PolicyValueModel::random(8, 6);
        let mut engine = Engine::new(&model, config());
        let stones = ["a2", "b1", "c2"]
            .into_iter()
            .map(|s| (Move::parse(s).unwrap(), Player::Black))
            .chain(
                ["b2", "a3", "c3", "b4"]
                    .into_iter()
                    .map(|s| (Move::parse(s).unwrap(), Player::White)),
            )
            .collect::<Vec<_>>();
        engine.board = Board::from_position(&stones, Player::Black).unwrap();
        engine.command("play", &["b", "b3"]).unwrap();
        let ko = engine.board.clone();
        assert!(engine.command("play", &["w", "b2"]).is_err());
        assert_eq!(engine.board, ko);
        engine.command("play", &["w", "j9"]).unwrap();
        engine.command("undo", &[]).unwrap();
        assert_eq!(engine.board, ko);
        assert!(engine.command("play", &["w", "b2"]).is_err());
    }
}
