use crate::{
    game::{BOARD_SIZE, Board, Move, Player},
    mcts::{SearchConfig, search, search_timed},
    model::PolicyValueModel,
};
use std::{
    io::{self, BufRead, Write},
    time::Duration,
};

pub struct GomocupConfig {
    pub simulations: usize,
    pub cpuct: f32,
}

pub fn run(model: &PolicyValueModel, config: GomocupConfig) -> io::Result<()> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut board = Board::new();
    let mut stones: Vec<(Move, Player)> = Vec::new();
    let mut timeout_turn_ms = 5_000u64;
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let command = line.trim();
        if command.is_empty() {
            continue;
        }
        let (name, argument) = command
            .split_once(char::is_whitespace)
            .map_or((command, ""), |(name, rest)| (name, rest.trim()));
        match name.to_ascii_uppercase().as_str() {
            "START" => {
                if argument.parse::<usize>().ok() != Some(BOARD_SIZE) {
                    respond(&mut output, "ERROR unsupported board size")?;
                } else {
                    board = Board::new();
                    stones.clear();
                    respond(&mut output, "OK")?;
                }
            }
            "RECTSTART" => {
                let supported = argument.split_once(',').and_then(|(width, height)| {
                    Some((
                        width.trim().parse::<usize>().ok()?,
                        height.trim().parse().ok()?,
                    ))
                }) == Some((BOARD_SIZE, BOARD_SIZE));
                if supported {
                    board = Board::new();
                    stones.clear();
                    respond(&mut output, "OK")?;
                } else {
                    respond(&mut output, "ERROR unsupported board size")?;
                }
            }
            "RESTART" => {
                board = Board::new();
                stones.clear();
                respond(&mut output, "OK")?;
            }
            "BEGIN" => play_and_respond(
                &mut board,
                &mut stones,
                model,
                &config,
                timeout_turn_ms,
                &mut output,
            )?,
            "TURN" => {
                let Some(mv) = parse_move(argument) else {
                    respond(&mut output, "ERROR invalid coordinate")?;
                    continue;
                };
                let player = board.to_move();
                if !board.play(mv) {
                    respond(&mut output, "ERROR illegal move")?;
                    continue;
                }
                stones.push((mv, player));
                play_and_respond(
                    &mut board,
                    &mut stones,
                    model,
                    &config,
                    timeout_turn_ms,
                    &mut output,
                )?;
            }
            "BOARD" => {
                stones.clear();
                let mut entries = Vec::new();
                loop {
                    line.clear();
                    if input.read_line(&mut line)? == 0 {
                        return Ok(());
                    }
                    let entry = line.trim();
                    if entry.eq_ignore_ascii_case("DONE") {
                        break;
                    }
                    let fields = entry.split(',').map(str::trim).collect::<Vec<_>>();
                    if fields.len() != 3 {
                        continue;
                    }
                    let (Ok(x), Ok(y), Ok(field)) = (
                        fields[0].parse::<usize>(),
                        fields[1].parse::<usize>(),
                        fields[2].parse::<u8>(),
                    ) else {
                        continue;
                    };
                    if let Some(mv) = Move::new(y, x) {
                        if field == 1 || field == 2 {
                            entries.push((mv, field));
                        }
                    }
                }
                let own = entries.iter().filter(|(_, field)| *field == 1).count();
                let opponent = entries.len() - own;
                let own_player = if own == opponent {
                    Player::Black
                } else if opponent == own + 1 {
                    Player::White
                } else {
                    respond(&mut output, "ERROR invalid board")?;
                    continue;
                };
                stones.extend(entries.into_iter().map(|(mv, field)| {
                    (
                        mv,
                        if field == 1 {
                            own_player
                        } else {
                            own_player.other()
                        },
                    )
                }));
                let Some(restored) = Board::from_stones(&stones) else {
                    respond(&mut output, "ERROR invalid board")?;
                    continue;
                };
                board = restored;
                play_and_respond(
                    &mut board,
                    &mut stones,
                    model,
                    &config,
                    timeout_turn_ms,
                    &mut output,
                )?;
            }
            "TAKEBACK" => {
                let Some(mv) = parse_move(argument) else {
                    respond(&mut output, "ERROR invalid coordinate")?;
                    continue;
                };
                if let Some(index) = stones.iter().rposition(|&(stone, _)| stone == mv) {
                    stones.remove(index);
                    if let Some(restored) = Board::from_stones(&stones) {
                        board = restored;
                        respond(&mut output, "OK")?;
                    } else {
                        respond(&mut output, "ERROR invalid takeback")?;
                    }
                } else {
                    respond(&mut output, "ERROR unknown move")?;
                }
            }
            "INFO" => {
                if let Some((key, value)) = argument.split_once(char::is_whitespace) {
                    if key.eq_ignore_ascii_case("timeout_turn") {
                        if let Ok(ms) = value.trim().parse::<u64>() {
                            // 协议规定 0 表示尽快落子，而不是无限时。
                            timeout_turn_ms = ms;
                        }
                    }
                }
            }
            "ABOUT" => respond(
                &mut output,
                r#"name="GomokuAZ", version="0.1.0", author="mycve", country="CN""#,
            )?,
            "END" => break,
            _ => respond(&mut output, &format!("UNKNOWN {command}"))?,
        }
    }
    Ok(())
}

fn play_and_respond(
    board: &mut Board,
    stones: &mut Vec<(Move, Player)>,
    model: &PolicyValueModel,
    config: &GomocupConfig,
    timeout_turn_ms: u64,
    output: &mut impl Write,
) -> io::Result<()> {
    if board.outcome().is_some() {
        return respond(output, "ERROR game is already over");
    }
    let search_config = SearchConfig {
        simulations: if timeout_turn_ms == 0 {
            1
        } else {
            config.simulations
        },
        cpuct: config.cpuct,
        ..Default::default()
    };
    let result = if timeout_turn_ms == 0 {
        search(board, model, search_config)
    } else {
        let safety_ms = (timeout_turn_ms / 20).clamp(5, 100);
        let limit = Duration::from_millis(timeout_turn_ms.saturating_sub(safety_ms).max(1));
        search_timed(board, model, search_config, limit)
    };
    let Some(best) = result.first() else {
        return respond(output, "ERROR no legal move");
    };
    let mv = best.mv;
    let player = board.to_move();
    board.play(mv);
    stones.push((mv, player));
    respond(output, &format!("{},{}", mv.col(), mv.row()))
}

fn parse_move(text: &str) -> Option<Move> {
    let (x, y) = text.split_once(',')?;
    Move::new(y.trim().parse().ok()?, x.trim().parse().ok()?)
}

fn respond(output: &mut impl Write, message: &str) -> io::Result<()> {
    writeln!(output, "{message}")?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_coordinates_are_column_then_row() {
        let mv = parse_move("4,12").unwrap();
        assert_eq!(mv.col(), 4);
        assert_eq!(mv.row(), 12);
        assert!(parse_move("15,0").is_none());
        assert!(parse_move("broken").is_none());
    }
}
