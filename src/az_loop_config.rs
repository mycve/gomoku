use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

pub const DEFAULT_CONFIG_PATH: &str = "gomoku.azloop.toml";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AzLoopConfig {
    pub format_version: u32,
    pub model_path: String,
    pub ema_model_path: String,
    pub best_model_path: String,
    pub replay_path: String,
    pub progress_path: String,
    pub simulations: usize,
    pub seed: u64,
    pub games_per_update: usize,
    pub selfplay_workers: usize,
    pub selfplay_queue_capacity: usize,
    pub learning_rate: f32,
    pub learning_rate_min: f32,
    pub learning_rate_decay: f32,
    pub batch_epochs: usize,
    pub batch_size: usize,
    pub gpu_devices: Vec<usize>,
    pub cpuct: f32,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub policy_softmax_temp: f32,
    pub moves_left_loss_weight: f32,
    pub ema_decay: f32,
    pub replay_capacity: usize,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games: usize,
    pub arena_simulations: usize,
    pub arena_opening_plies: usize,
    pub arena_promotion_rate: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopConfig {
    fn default() -> Self {
        Self {
            format_version: 1,
            model_path: "model.safetensors".into(),
            ema_model_path: "ema.safetensors".into(),
            best_model_path: "best.safetensors".into(),
            replay_path: "data/replay.jsonl".into(),
            progress_path: "data/azloop-progress.json".into(),
            simulations: 400,
            seed: 20260730,
            games_per_update: 16,
            selfplay_workers: 0,
            selfplay_queue_capacity: 0,
            learning_rate: 0.01,
            learning_rate_min: 0.001,
            learning_rate_decay: 0.995,
            batch_epochs: 2,
            batch_size: 256,
            gpu_devices: Vec::new(),
            cpuct: 1.5,
            temperature_start: 0.9,
            temperature_endgame: 0.35,
            temperature_decay_delay_plies: 12,
            temperature_decay_plies: 24,
            temperature_value_cutoff: 0.12,
            temperature_visit_offset: -0.8,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.25,
            policy_softmax_temp: 1.45,
            moves_left_loss_weight: 0.1,
            ema_decay: 0.999,
            replay_capacity: 100_000,
            checkpoint_interval: 10,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 20,
            arena_interval: 10,
            arena_games: 20,
            arena_simulations: 400,
            arena_opening_plies: 2,
            arena_promotion_rate: 0.55,
            tensorboard_logdir: "runs/gomoku".into(),
        }
    }
}

pub fn load_or_create(path: impl AsRef<Path>) -> io::Result<(AzLoopConfig, bool)> {
    let path = path.as_ref();
    if path.exists() {
        let config = toml::from_str(&fs::read_to_string(path)?).map_err(io::Error::other)?;
        Ok((config, false))
    } else {
        let config = AzLoopConfig::default();
        fs::write(
            path,
            toml::to_string_pretty(&config).map_err(io::Error::other)?,
        )?;
        Ok((config, true))
    }
}
