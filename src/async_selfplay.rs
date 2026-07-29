use crate::{
    mcts::SearchConfig,
    model::PolicyValueModel,
    replay::Sample,
    selfplay::{SelfplayStats, generate_one_detailed},
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
}

impl AsyncSelfplay {
    pub fn start(
        model: Arc<RwLock<PolicyValueModel>>,
        model_version: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
        workers: usize,
        queue_capacity: usize,
        search: SearchConfig,
        seed: u64,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(queue_capacity.max(1));
        let backlog = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let sender = sender.clone();
            let model = Arc::clone(&model);
            let model_version = Arc::clone(&model_version);
            let stop = Arc::clone(&stop);
            let backlog = Arc::clone(&backlog);
            handles.push(thread::spawn(move || {
                worker_loop(
                    worker,
                    model,
                    model_version,
                    stop,
                    backlog,
                    sender,
                    search,
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
    model_version: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    backlog: Arc<AtomicUsize>,
    sender: SyncSender<SelfplayGame>,
    search: SearchConfig,
    seed: u64,
) {
    let mut game_index = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let version = model_version.load(Ordering::Acquire);
        let snapshot = model.read().unwrap_or_else(|e| e.into_inner()).clone();
        let game_seed = seed
            ^ (worker as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ game_index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let generated = generate_one_detailed(&snapshot, search, game_seed);
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
            }
            Err(TrySendError::Disconnected(_)) => {
                backlog.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
    }
}
