use crate::{
    mcts::SearchConfig,
    model::PolicyValueModel,
    replay::Sample,
    selfplay::{GeneratedGame, SelfplayStats, generate_one_detailed_match_controlled},
};
use std::{
    io,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
};

pub struct SelfplayGame {
    pub worker: usize,
    pub model_version: u64,
    pub samples: Vec<Sample>,
    pub stats: SelfplayStats,
}

pub struct AsyncSelfplay {
    receiver: Option<Receiver<SelfplayGame>>,
    handles: Vec<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    backlog: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

#[derive(Clone, Copy, Debug)]
pub struct SelfplayLeagueConfig {
    pub white_simulations: usize,
    pub current_selfplay_fraction: f32,
    pub current_white_history_fraction: f32,
    pub paired_history_fraction: f32,
}

impl AsyncSelfplay {
    pub fn start(
        model: Arc<RwLock<PolicyValueModel>>,
        history: Arc<RwLock<Vec<PolicyValueModel>>>,
        model_version: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
        workers: usize,
        queue_capacity: usize,
        search: SearchConfig,
        league: SelfplayLeagueConfig,
        seed: u64,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(queue_capacity.max(1));
        let backlog = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let sender = sender.clone();
            let model = Arc::clone(&model);
            let history = Arc::clone(&history);
            let model_version = Arc::clone(&model_version);
            let stop = Arc::clone(&stop);
            let backlog = Arc::clone(&backlog);
            let dropped = Arc::clone(&dropped);
            handles.push(thread::spawn(move || {
                worker_loop(
                    worker,
                    model,
                    history,
                    model_version,
                    stop,
                    backlog,
                    dropped,
                    sender,
                    search,
                    league,
                    seed,
                )
            }));
        }
        drop(sender);
        Self {
            receiver: Some(receiver),
            handles,
            stop,
            backlog,
            dropped,
        }
    }

    pub fn take_receiver(&mut self) -> Receiver<SelfplayGame> {
        self.receiver.take().expect("自博弈接收端只能获取一次")
    }

    pub fn receive(&self) -> io::Result<Option<SelfplayGame>> {
        self.receiver
            .as_ref()
            .expect("自博弈接收端已移交")
            .recv()
            .map(Some)
            .map_err(|_| io::Error::other("所有自博弈 Worker 均已退出"))
    }

    pub fn backlog(&self) -> usize {
        self.backlog.load(Ordering::Relaxed)
    }
    pub fn backlog_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.backlog)
    }
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn shutdown(self) -> io::Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        drop(self.receiver);
        for handle in self.handles {
            handle
                .join()
                .map_err(|_| io::Error::other("自博弈 Worker 异常退出"))?;
        }
        Ok(())
    }
}

fn worker_loop(
    worker: usize,
    model: Arc<RwLock<PolicyValueModel>>,
    history: Arc<RwLock<Vec<PolicyValueModel>>>,
    model_version: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    backlog: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
    sender: SyncSender<SelfplayGame>,
    search: SearchConfig,
    league: SelfplayLeagueConfig,
    seed: u64,
) {
    let mut game_index = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let version = model_version.load(Ordering::Acquire);
        let game_seed = seed
            ^ (worker as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ game_index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let snapshot = model.read().unwrap_or_else(|e| e.into_inner()).clone();
        let history_snapshot = {
            let pool = history.read().unwrap_or_else(|e| e.into_inner());
            (!pool.is_empty()).then(|| pool[history_index(game_seed, pool.len())].clone())
        };
        let mut white_search = search;
        white_search.simulations = league.white_simulations;
        let generated = match history_snapshot {
            None => {
                let mut game = generate_one_detailed_match_controlled(
                    &snapshot,
                    &snapshot,
                    search,
                    white_search,
                    true,
                    true,
                    game_seed,
                    Some(&stop),
                );
                game.stats.current_selfplay_games = 1;
                game
            }
            Some(history_model) => match league_mode(game_seed, league) {
                LeagueMode::CurrentSelfplay => {
                    let mut game = generate_one_detailed_match_controlled(
                        &snapshot,
                        &snapshot,
                        search,
                        white_search,
                        true,
                        true,
                        game_seed,
                        Some(&stop),
                    );
                    game.stats.current_selfplay_games = 1;
                    game
                }
                LeagueMode::CurrentWhite => {
                    let mut game = generate_one_detailed_match_controlled(
                        &history_model,
                        &snapshot,
                        search,
                        white_search,
                        false,
                        true,
                        game_seed,
                        Some(&stop),
                    );
                    game.stats.current_white_history_games = 1;
                    game
                }
                LeagueMode::PairedHistory => {
                    let first = generate_one_detailed_match_controlled(
                        &snapshot,
                        &history_model,
                        search,
                        white_search,
                        true,
                        false,
                        game_seed,
                        Some(&stop),
                    );
                    let second = generate_one_detailed_match_controlled(
                        &history_model,
                        &snapshot,
                        search,
                        white_search,
                        false,
                        true,
                        game_seed,
                        Some(&stop),
                    );
                    let mut pair = merge_generated(first, second);
                    pair.stats.paired_history_games = 2;
                    pair
                }
            },
        };
        crate::profile::flush_thread();
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let game = SelfplayGame {
            worker,
            model_version: version,
            samples: generated.samples,
            stats: generated.stats,
        };
        game_index += 1;
        backlog.fetch_add(1, Ordering::Relaxed);
        match sender.try_send(game) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                backlog.fetch_sub(1, Ordering::Relaxed);
                dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                backlog.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeagueMode {
    CurrentSelfplay,
    CurrentWhite,
    PairedHistory,
}

fn league_mode(seed: u64, config: SelfplayLeagueConfig) -> LeagueMode {
    // PairedHistory 一次产生两局；按 job 权重减半后，长期“局数”比例仍为配置的 70/20/10。
    let current = config.current_selfplay_fraction.max(0.0);
    let white = config.current_white_history_fraction.max(0.0);
    let paired_job = config.paired_history_fraction.max(0.0) * 0.5;
    let total = (current + white + paired_job).max(f32::EPSILON);
    let unit = ((mix64(seed ^ 0xA076_1D64_78BD_642F) >> 40) as f32) / (1_u32 << 24) as f32;
    if unit < current / total {
        LeagueMode::CurrentSelfplay
    } else if unit < (current + white) / total {
        LeagueMode::CurrentWhite
    } else {
        LeagueMode::PairedHistory
    }
}

fn history_index(seed: u64, len: usize) -> usize {
    (mix64(seed ^ 0xE703_7ED1_A0B4_28DB) as usize) % len.max(1)
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn merge_generated(mut first: GeneratedGame, second: GeneratedGame) -> GeneratedGame {
    first.samples.extend(second.samples);
    first.stats.add_assign(&second.stats);
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn league_job_weights_produce_configured_game_fractions() {
        let config = SelfplayLeagueConfig {
            white_simulations: 8,
            current_selfplay_fraction: 0.70,
            current_white_history_fraction: 0.20,
            paired_history_fraction: 0.10,
        };
        let mut current = 0usize;
        let mut white = 0usize;
        let mut paired = 0usize;
        for seed in 0..100_000_u64 {
            match league_mode(seed, config) {
                LeagueMode::CurrentSelfplay => current += 1,
                LeagueMode::CurrentWhite => white += 1,
                LeagueMode::PairedHistory => paired += 2,
            }
        }
        let games = (current + white + paired) as f32;
        assert!((current as f32 / games - 0.70).abs() < 0.01);
        assert!((white as f32 / games - 0.20).abs() < 0.01);
        assert!((paired as f32 / games - 0.10).abs() < 0.01);
    }
}
