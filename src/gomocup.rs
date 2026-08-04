use crate::{
    game::{BOARD_SIZE, Board, Move, Player},
    mcts::{SearchConfig, search_timed},
    model::PolicyValueModel,
};
use std::{
    io::{self, BufRead, Write},
    time::Duration,
};

pub struct GomocupConfig {
    pub cpuct: f32,
}

#[derive(Clone, Copy)]
struct ProtocolLimits {
    timeout_turn_ms: u64,
    time_left_ms: Option<u64>,
    unsupported_rule: Option<u32>,
}

pub fn run(model: &PolicyValueModel, config: GomocupConfig) -> io::Result<()> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut board = Board::new();
    let mut stones: Vec<(Move, Player)> = Vec::new();
    let mut limits = ProtocolLimits {
        timeout_turn_ms: 5_000,
        time_left_ms: None,
        unsupported_rule: None,
    };
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
            "BEGIN" => {
                board = Board::new();
                stones.clear();
                play_and_respond(&mut board, &mut stones, model, &config, limits, &mut output)?
            }
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
                play_and_respond(&mut board, &mut stones, model, &config, limits, &mut output)?;
            }
            "BOARD" => {
                let mut entries = Vec::new();
                let mut invalid_entry = false;
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
                        invalid_entry = true;
                        continue;
                    }
                    let (Ok(x), Ok(y), Ok(field)) = (
                        fields[0].parse::<usize>(),
                        fields[1].parse::<usize>(),
                        fields[2].parse::<u8>(),
                    ) else {
                        invalid_entry = true;
                        continue;
                    };
                    if let Some(mv) = Move::new(y, x)
                        && (field == 1 || field == 2)
                    {
                        entries.push((mv, field));
                    } else {
                        invalid_entry = true;
                    }
                }
                if invalid_entry {
                    respond(&mut output, "ERROR invalid board")?;
                    continue;
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
                let restored_stones = entries
                    .into_iter()
                    .map(|(mv, field)| {
                        (
                            mv,
                            if field == 1 {
                                own_player
                            } else {
                                own_player.other()
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let Some(restored) = Board::from_stones(&restored_stones) else {
                    respond(&mut output, "ERROR invalid board")?;
                    continue;
                };
                board = restored;
                stones = restored_stones;
                play_and_respond(&mut board, &mut stones, model, &config, limits, &mut output)?;
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
                    let value = value.trim();
                    if key.eq_ignore_ascii_case("timeout_turn") {
                        if let Ok(ms) = value.parse::<u64>() {
                            limits.timeout_turn_ms = ms;
                        }
                    } else if key.eq_ignore_ascii_case("time_left") {
                        if let Ok(ms) = value.parse::<i64>() {
                            limits.time_left_ms = Some(ms.max(0) as u64);
                        }
                    } else if key.eq_ignore_ascii_case("rule")
                        && let Ok(rule) = value.parse::<u32>()
                    {
                        limits.unsupported_rule = (rule != 0).then_some(rule);
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
    limits: ProtocolLimits,
    output: &mut impl Write,
) -> io::Result<()> {
    if let Some(rule) = limits.unsupported_rule {
        return respond(output, &format!("ERROR unsupported rule {rule}"));
    }
    if board.outcome().is_some() {
        return respond(output, "ERROR game is already over");
    }
    let budget_ms = turn_budget_ms(
        limits.timeout_turn_ms,
        limits.time_left_ms,
        board.move_count(),
    )?;
    let search_config = SearchConfig {
        simulations: usize::MAX,
        cpuct: config.cpuct,
        policy_softmax_temp: 1.0,
        root_policy_temperature_early: 1.0,
        root_policy_temperature: 1.0,
        ..Default::default()
    };
    let safety_ms = (budget_ms / 20).clamp(5, 100);
    let limit = Duration::from_millis(budget_ms.saturating_sub(safety_ms).max(1));
    let result = search_timed(board, model, search_config, limit);
    let Some(best) = result.first() else {
        return respond(output, "ERROR no legal move");
    };
    let mv = best.mv;
    let player = board.to_move();
    board.play(mv);
    stones.push((mv, player));
    respond(output, &format!("{},{}", mv.col(), mv.row()))
}

fn turn_budget_ms(
    timeout_turn_ms: u64,
    time_left_ms: Option<u64>,
    move_count: usize,
) -> io::Result<u64> {
    if timeout_turn_ms == 0 {
        return Err(io::Error::other("协议提供的 timeout_turn 必须大于 0"));
    }
    let Some(time_left_ms) = time_left_ms else {
        return Ok(timeout_turn_ms);
    };
    if time_left_ms == 0 {
        return Err(io::Error::other("协议提供的 time_left 必须大于 0"));
    }
    let remaining_own_turns =
        ((crate::game::CELL_COUNT - move_count).saturating_add(1) / 2).clamp(1, 20) as u64;
    let match_budget = time_left_ms.saturating_sub(100).max(1) / remaining_own_turns;
    Ok(timeout_turn_ms.min(match_budget.max(1)))
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

    #[test]
    fn turn_budget_respects_turn_and_match_limits() {
        assert!(turn_budget_ms(0, Some(10_000), 0).is_err());
        assert_eq!(turn_budget_ms(5_000, None, 0).unwrap(), 5_000);
        assert!(turn_budget_ms(5_000, Some(10_000), 0).unwrap() < 5_000);
        assert!(turn_budget_ms(5_000, Some(0), 0).is_err());
    }
}
