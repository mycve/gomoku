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
    pub selfplay_samples_per_update: usize,
    pub selfplay_workers: usize,
    pub selfplay_queue_capacity: usize,
    pub selfplay_random_opening_probability: f32,
    pub learning_rate: f32,
    pub learning_rate_min: f32,
    pub learning_rate_decay: f32,
    pub batch_epochs: usize,
    pub batch_size: usize,
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
    pub ema_decay: f32,
    pub replay_capacity: usize,
    pub replay_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_updates: u64,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games: usize,
    pub arena_simulations: usize,
    pub arena_opening_plies: usize,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopConfig {
    fn default() -> Self {
        Self {
            format_version: 3,
            model_path: "model.safetensors".into(),
            ema_model_path: "ema.safetensors".into(),
            best_model_path: "best.safetensors".into(),
            replay_path: "data/replay.jsonl".into(),
            progress_path: "data/azloop-progress.json".into(),
            simulations: 3000,
            seed: 20260730,
            selfplay_samples_per_update: 50_000,
            selfplay_workers: 196,
            selfplay_queue_capacity: 0,
            selfplay_random_opening_probability: 0.25,
            learning_rate: 0.0008,
            learning_rate_min: 0.0002,
            learning_rate_decay: 0.90,
            batch_epochs: 1,
            batch_size: 256,
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
            ema_decay: 0.999,
            replay_capacity: 500_000,
            replay_warmup_samples: 100_000,
            train_samples_per_update: 50_000,
            replay_recent_sample_fraction: 0.4,
            replay_recent_updates: 5,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 20,
            arena_interval: 10,
            arena_games: 100,
            arena_simulations: 3000,
            arena_opening_plies: 2,
            arena_promotion_rate: 0.55,
            arena_promotion_confidence_z: 1.28,
            tensorboard_logdir: "runs/gomoku".into(),
        }
    }
}

pub fn load_or_create(path: impl AsRef<Path>) -> io::Result<(AzLoopConfig, bool)> {
    let path = path.as_ref();
    if path.exists() {
        let config: AzLoopConfig =
            toml::from_str(&fs::read_to_string(path)?).map_err(io::Error::other)?;
        config.validate()?;
        Ok((config, false))
    } else {
        let config = AzLoopConfig::default();
        fs::write(path, DEFAULT_CONFIG_TEXT)?;
        Ok((config, true))
    }
}

impl AzLoopConfig {
    fn validate(&self) -> io::Result<()> {
        let finite = [
            ("learning_rate", self.learning_rate),
            (
                "selfplay_random_opening_probability",
                self.selfplay_random_opening_probability,
            ),
            ("learning_rate_min", self.learning_rate_min),
            ("learning_rate_decay", self.learning_rate_decay),
            ("cpuct", self.cpuct),
            ("temperature_start", self.temperature_start),
            ("temperature_endgame", self.temperature_endgame),
            ("temperature_value_cutoff", self.temperature_value_cutoff),
            ("temperature_visit_offset", self.temperature_visit_offset),
            ("root_dirichlet_alpha", self.root_dirichlet_alpha),
            ("root_exploration_fraction", self.root_exploration_fraction),
            ("policy_softmax_temp", self.policy_softmax_temp),
            ("ema_decay", self.ema_decay),
            ("arena_promotion_rate", self.arena_promotion_rate),
            (
                "arena_promotion_confidence_z",
                self.arena_promotion_confidence_z,
            ),
            (
                "replay_recent_sample_fraction",
                self.replay_recent_sample_fraction,
            ),
        ];
        for (name, value) in finite {
            if !value.is_finite() {
                return Err(io::Error::other(format!("配置 `{name}` 必须是有限数值")));
            }
        }
        if self.format_version != 3 {
            return Err(io::Error::other("仅支持 format_version = 3"));
        }
        if self.simulations == 0
            || self.selfplay_samples_per_update == 0
            || self.replay_warmup_samples == 0
            || self.batch_epochs == 0
            || self.batch_size == 0
            || self.train_samples_per_update == 0
        {
            return Err(io::Error::other(
                "simulations、selfplay_samples_per_update、replay_warmup_samples、batch_size 和 train_samples_per_update 必须大于 0",
            ));
        }
        if self.learning_rate <= 0.0
            || self.learning_rate_min < 0.0
            || !(0.0..=1.0).contains(&self.learning_rate_decay)
            || self.cpuct <= 0.0
            || self.temperature_start < 0.0
            || self.temperature_endgame < 0.0
            || self.temperature_value_cutoff < 0.0
            || self.root_dirichlet_alpha < 0.0
            || self.policy_softmax_temp <= 0.0
            || !(0.0..=1.0).contains(&self.root_exploration_fraction)
            || !(0.0..=1.0).contains(&self.ema_decay)
            || !(0.0..=1.0).contains(&self.arena_promotion_rate)
            || self.arena_promotion_confidence_z < 0.0
            || !(0.0..=1.0).contains(&self.replay_recent_sample_fraction)
            || !(0.0..=1.0).contains(&self.selfplay_random_opening_probability)
        {
            return Err(io::Error::other(
                "配置中的学习率、搜索或比例参数超出合法范围",
            ));
        }
        if self.arena_interval > 0 && (self.arena_games == 0 || self.arena_simulations == 0) {
            return Err(io::Error::other(
                "启用 Arena 时 arena_games 和 arena_simulations 必须大于 0",
            ));
        }
        if self.replay_warmup_samples > self.replay_capacity {
            return Err(io::Error::other(
                "replay_warmup_samples 不能超过 replay_capacity",
            ));
        }
        Ok(())
    }
}

const DEFAULT_CONFIG_TEXT: &str = r#"format_version = 3
model_path = "model.safetensors"
ema_model_path = "ema.safetensors"
best_model_path = "best.safetensors"
replay_path = "data/replay.jsonl"
progress_path = "data/azloop-progress.json"
simulations = 3000
seed = 20260730
selfplay_samples_per_update = 50000
selfplay_workers = 196
selfplay_queue_capacity = 0
selfplay_random_opening_probability = 0.25
learning_rate = 0.0008
learning_rate_min = 0.0002
learning_rate_decay = 0.90
batch_epochs = 1
batch_size = 256
cpuct = 1.5
temperature_start = 0.9
temperature_endgame = 0.35
temperature_decay_delay_plies = 12
temperature_decay_plies = 24
temperature_value_cutoff = 0.12
temperature_visit_offset = -0.8
root_dirichlet_alpha = 0.12
root_exploration_fraction = 0.25
policy_softmax_temp = 1.45
ema_decay = 0.999
replay_capacity = 500000
replay_warmup_samples = 100000
train_samples_per_update = 50000
replay_recent_sample_fraction = 0.4
replay_recent_updates = 5
checkpoint_interval = 20
checkpoint_dir = "checkpoints"
max_checkpoints = 20
arena_interval = 10
arena_games = 100
arena_simulations = 3000
arena_opening_plies = 2
arena_promotion_rate = 0.550000011920929
arena_promotion_confidence_z = 1.2799999713897705
tensorboard_logdir = "runs/gomoku"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_is_exact_and_valid() {
        let config: AzLoopConfig = toml::from_str(DEFAULT_CONFIG_TEXT).unwrap();
        config.validate().unwrap();
        assert_eq!(config.format_version, 3);
        assert_eq!(config.selfplay_samples_per_update, 50_000);
        assert_eq!(config.selfplay_workers, 196);
        assert_eq!(config.selfplay_random_opening_probability, 0.25);
        assert_eq!(config.replay_capacity, 500_000);
        assert_eq!(config.replay_warmup_samples, 100_000);
        assert_eq!(config.train_samples_per_update, 50_000);
        assert!(DEFAULT_CONFIG_TEXT.contains("learning_rate = 0.0008\n"));
        assert!(DEFAULT_CONFIG_TEXT.contains("arena_promotion_rate = 0.550000011920929\n"));
        assert_eq!(config.arena_promotion_confidence_z, 1.28);
    }
}
