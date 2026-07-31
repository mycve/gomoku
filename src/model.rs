//! 正式策略价值模型接口。
//!
//! 引擎、MCTS、自博弈和训练统一使用 CPU 稀疏 Transformer。

pub use crate::sparse_transformer::{
    SparseScratch as EvalScratch, SparseTransformerModel as PolicyValueModel,
};
