use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

pub const DEFAULT_CONFIG_PATH: &str = "go19-v32.azloop.toml";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AzLoopConfig {
    pub format_version: u32,
    pub model_path: String,
    pub best_model_path: String,
    pub replay_path: String,
    pub progress_path: String,
    pub simulations: usize,
    pub hidden_size: usize,
    pub seed: u64,
    pub selfplay_samples_per_update: usize,
    pub selfplay_workers: usize,
    pub selfplay_queue_capacity: usize,
    pub learning_rate: f32,
    pub learning_rate_min: f32,
    pub learning_rate_warmup_steps: usize,
    pub learning_rate_cosine_steps: usize,
    pub batch_epochs: usize,
    pub batch_size: usize,
    pub cpuct: f32,
    pub cpuct_log: f32,
    pub cpuct_base: f32,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub root_dirichlet_total_concentration: f32,
    pub root_exploration_fraction: f32,
    pub root_policy_temperature: f32,
    pub replay_capacity: usize,
    pub replay_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_updates: u64,
    pub replay_policy_surprise_fraction: f32,
    pub replay_value_surprise_fraction: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games: usize,
    pub arena_opening_plies: usize,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_color_score_floor: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopConfig {
    fn default() -> Self {
        Self {
            format_version: 22,
            model_path: "go19-v32-model.safetensors".into(),
            best_model_path: "go19-v32-best.safetensors".into(),
            replay_path: "data/go19-v32/replay.bin.lz4".into(),
            progress_path: "data/go19-v32/azloop-progress.json".into(),
            simulations: 400,
            hidden_size: 128,
            seed: 20260730,
            selfplay_samples_per_update: 81920,
            selfplay_workers: 128,
            selfplay_queue_capacity: 0,
            learning_rate: 0.0008,
            learning_rate_min: 0.0001,
            learning_rate_warmup_steps: 200,
            learning_rate_cosine_steps: 10_000,
            batch_epochs: 1,
            batch_size: 256,
            cpuct: 2.0,
            cpuct_log: 1.2,
            cpuct_base: 500.0,
            temperature_start: 1.0,
            temperature_endgame: 0.05,
            temperature_decay_delay_plies: 12,
            temperature_decay_plies: 48,
            root_dirichlet_total_concentration: 10.83,
            root_exploration_fraction: 0.25,
            root_policy_temperature: 1.45,
            replay_capacity: 500000,
            replay_warmup_samples: 81920,
            train_samples_per_update: 81920,
            replay_recent_sample_fraction: 0.4,
            replay_recent_updates: 5,
            replay_policy_surprise_fraction: 0.4,
            replay_value_surprise_fraction: 0.1,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints/go19-v32".into(),
            max_checkpoints: 20,
            arena_interval: 20,
            arena_games: 100,
            arena_opening_plies: 2,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.28,
            arena_color_score_floor: 0.45,
            tensorboard_logdir: "runs/go19-v32".into(),
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
            ("learning_rate_min", self.learning_rate_min),
            ("cpuct", self.cpuct),
            ("cpuct_log", self.cpuct_log),
            ("cpuct_base", self.cpuct_base),
            ("temperature_start", self.temperature_start),
            ("temperature_endgame", self.temperature_endgame),
            (
                "root_dirichlet_total_concentration",
                self.root_dirichlet_total_concentration,
            ),
            ("root_exploration_fraction", self.root_exploration_fraction),
            ("root_policy_temperature", self.root_policy_temperature),
            ("arena_promotion_rate", self.arena_promotion_rate),
            (
                "arena_promotion_confidence_z",
                self.arena_promotion_confidence_z,
            ),
            ("arena_color_score_floor", self.arena_color_score_floor),
            (
                "replay_recent_sample_fraction",
                self.replay_recent_sample_fraction,
            ),
            (
                "replay_policy_surprise_fraction",
                self.replay_policy_surprise_fraction,
            ),
            (
                "replay_value_surprise_fraction",
                self.replay_value_surprise_fraction,
            ),
        ];
        for (name, value) in finite {
            if !value.is_finite() {
                return Err(io::Error::other(format!("配置 `{name}` 必须是有限数值")));
            }
        }
        if self.format_version != 22 {
            return Err(io::Error::other(
                "仅支持 format_version = 22；请重新生成配置",
            ));
        }
        if self.simulations == 0
            || self.hidden_size == 0
            || self.selfplay_samples_per_update == 0
            || self.replay_warmup_samples == 0
            || self.batch_epochs == 0
            || self.batch_size == 0
            || self.train_samples_per_update == 0
            || self.learning_rate_warmup_steps == 0
            || self.learning_rate_cosine_steps == 0
        {
            return Err(io::Error::other(
                "simulations、selfplay_samples_per_update、replay_warmup_samples、batch_size 和 train_samples_per_update 必须大于 0",
            ));
        }
        if self.learning_rate <= 0.0
            || self.learning_rate_min < 0.0
            || self.learning_rate_min > self.learning_rate
            || self.cpuct <= 0.0
            || self.cpuct_log < 0.0
            || self.cpuct_base <= 0.0
            || self.temperature_start < 0.0
            || self.temperature_endgame < 0.0
            || self.root_dirichlet_total_concentration < 0.0
            || self.root_policy_temperature <= 0.0
            || !(0.0..=1.0).contains(&self.root_exploration_fraction)
            || !(0.0..=1.0).contains(&self.arena_promotion_rate)
            || !(0.0..=1.0).contains(&self.arena_color_score_floor)
            || self.arena_promotion_confidence_z < 0.0
            || !(0.0..=1.0).contains(&self.replay_recent_sample_fraction)
            || !(0.0..=1.0).contains(&self.replay_policy_surprise_fraction)
            || !(0.0..=1.0).contains(&self.replay_value_surprise_fraction)
        {
            return Err(io::Error::other(
                "配置中的学习率、搜索或比例参数超出合法范围",
            ));
        }
        if self.arena_interval > 0 && self.arena_games == 0 {
            return Err(io::Error::other("启用 Arena 时 arena_games 必须大于 0"));
        }
        if self.replay_warmup_samples > self.replay_capacity {
            return Err(io::Error::other(
                "replay_warmup_samples 不能超过 replay_capacity",
            ));
        }
        if self.replay_policy_surprise_fraction + self.replay_value_surprise_fraction > 1.0 {
            return Err(io::Error::other(
                "Policy 和 Value surprise 抽样比例之和不能超过 1",
            ));
        }
        Ok(())
    }
}

const DEFAULT_CONFIG_TEXT: &str = r#"format_version = 22
model_path = "go19-v32-model.safetensors"
best_model_path = "go19-v32-best.safetensors"
replay_path = "data/go19-v32/replay.bin.lz4"
progress_path = "data/go19-v32/azloop-progress.json"
simulations = 400
hidden_size = 128
seed = 20260730
selfplay_samples_per_update = 81920
selfplay_workers = 128
selfplay_queue_capacity = 0
learning_rate = 0.0008
learning_rate_min = 0.0001
learning_rate_warmup_steps = 200
learning_rate_cosine_steps = 10000
batch_epochs = 1
batch_size = 256
cpuct = 2.0
cpuct_log = 1.2
cpuct_base = 500.0
temperature_start = 1.0
temperature_endgame = 0.05
temperature_decay_delay_plies = 12
temperature_decay_plies = 48
root_dirichlet_total_concentration = 10.83
root_exploration_fraction = 0.25
root_policy_temperature = 1.45
replay_capacity = 500000
replay_warmup_samples = 81920
train_samples_per_update = 81920
replay_recent_sample_fraction = 0.4
replay_recent_updates = 5
replay_policy_surprise_fraction = 0.4
replay_value_surprise_fraction = 0.1
checkpoint_interval = 20
checkpoint_dir = "checkpoints/go19-v32"
max_checkpoints = 20
arena_interval = 20
arena_games = 100
arena_opening_plies = 2
arena_promotion_rate = 0.5
arena_promotion_confidence_z = 1.28
arena_color_score_floor = 0.45
tensorboard_logdir = "runs/go19-v32"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_is_exact_and_valid() {
        let config: AzLoopConfig = toml::from_str(DEFAULT_CONFIG_TEXT).unwrap();
        config.validate().unwrap();
        assert_eq!(
            toml::to_string(&config).unwrap(),
            toml::to_string(&AzLoopConfig::default()).unwrap()
        );
        assert_eq!(config.format_version, 22);
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.selfplay_samples_per_update, 81920);
        assert_eq!(config.selfplay_workers, 128);
        assert_eq!(config.arena_games, 100);
        assert_eq!(config.replay_capacity, 500000);
        assert_eq!(config.replay_warmup_samples, 81920);
        assert_eq!(config.train_samples_per_update, 81920);
        assert!(DEFAULT_CONFIG_TEXT.contains("learning_rate = 0.0008\n"));
        assert!(DEFAULT_CONFIG_TEXT.contains("arena_promotion_rate = 0.5\n"));
        assert_eq!(config.arena_color_score_floor, 0.45);
        assert_eq!(config.arena_promotion_confidence_z, 1.28);
    }
}
