use clap::Parser;
use gomoku::{
    gomocup::{self, GomocupConfig},
    model::PolicyValueModel,
};
use std::{
    io,
    path::{Path, PathBuf},
};

/// 可交付的 Gomocup/Piskvork 五子棋协议引擎。
#[derive(Parser)]
#[command(name = "gomoku-engine", version, about)]
struct Args {
    /// Safetensors 模型路径。
    #[arg(default_value = "model.safetensors")]
    model: PathBuf,
    /// 每步搜索模拟次数上限；仍受协议 timeout_turn 限制。
    #[arg(long, default_value_t = 50000)]
    simulations: usize,
    /// PUCT 探索常数。
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let model_path = resolve_model_path(&args.model);
    let model = PolicyValueModel::load(&model_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("无法加载模型 `{}`: {error}", model_path.display()),
        )
    })?;
    gomocup::run(
        &model,
        GomocupConfig {
            simulations: args.simulations.max(1),
            cpuct: args.cpuct,
        },
    )
}

fn resolve_model_path(requested: &Path) -> PathBuf {
    if requested.exists() {
        return requested.to_path_buf();
    }
    let fallback = requested
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("best.safetensors");
    if fallback.exists() {
        fallback
    } else {
        requested.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_uses_sibling_best() {
        let directory = std::env::temp_dir().join(format!(
            "gomoku-engine-model-fallback-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let best = directory.join("best.safetensors");
        std::fs::write(&best, []).unwrap();
        assert_eq!(
            resolve_model_path(&directory.join("model.safetensors")),
            best
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
