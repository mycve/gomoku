use crate::game::{Board, CELL_COUNT, Move, Player};
use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;
use std::{fs, io, path::Path};

pub const TOKEN_WIDTH: usize = 32;
pub const HEADS: usize = 2;
pub const HEAD_WIDTH: usize = TOKEN_WIDTH / HEADS;
pub const LAYERS: usize = 2;
pub const FF_WIDTH: usize = 64;
const LINE_COUNT: usize = 15 + 15 + 29 + 29;
const COLUMN_OFFSET: usize = 15;
const DIAGONAL_OFFSET: usize = 30;
const ANTI_DIAGONAL_OFFSET: usize = 59;
const CELL_LINES: [[u8; 4]; CELL_COUNT] = build_cell_lines();
const ALIGNED_MASKS: [[u64; 4]; CELL_COUNT] = build_aligned_masks();

const fn build_cell_lines() -> [[u8; 4]; CELL_COUNT] {
    let mut lines = [[0; 4]; CELL_COUNT];
    let mut cell = 0;
    while cell < CELL_COUNT {
        let row = cell / 15;
        let col = cell % 15;
        lines[cell] = [
            row as u8,
            (COLUMN_OFFSET + col) as u8,
            (DIAGONAL_OFFSET + row + 14 - col) as u8,
            (ANTI_DIAGONAL_OFFSET + row + col) as u8,
        ];
        cell += 1;
    }
    lines
}

const fn build_aligned_masks() -> [[u64; 4]; CELL_COUNT] {
    let mut masks = [[0; 4]; CELL_COUNT];
    let mut left = 0;
    while left < CELL_COUNT {
        let mut right = 0;
        while right < CELL_COUNT {
            let mut left_axis = 0;
            let mut is_aligned = false;
            while left_axis < 4 {
                let mut right_axis = 0;
                while right_axis < 4 {
                    if CELL_LINES[left][left_axis] == CELL_LINES[right][right_axis]
                        && left_axis == right_axis
                    {
                        is_aligned = true;
                    }
                    right_axis += 1;
                }
                left_axis += 1;
            }
            if is_aligned {
                masks[left][right / 64] |= 1_u64 << (right % 64);
            }
            right += 1;
        }
        left += 1;
    }
    masks
}

#[derive(Clone)]
pub(crate) struct Block {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) ff_up: Vec<f32>,
    pub(crate) ff_down: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseTransformerModel {
    pub(crate) stone_embedding: Vec<f32>,
    pub(crate) position_embedding: Vec<f32>,
    pub(crate) blocks: Vec<Block>,
    #[allow(dead_code)] // 训练接入后由优化器更新；推理使用加载时生成的预投影缓存。
    pub(crate) policy_query: Vec<f32>,
    projected_policy_query: Vec<f32>,
    pub(crate) policy_key: Vec<f32>,
    pub(crate) policy_value: Vec<f32>,
    pub(crate) policy_output: Vec<f32>,
    pub(crate) policy_bias: Vec<f32>,
    pub(crate) value_hidden: Vec<f32>,
    pub(crate) value_output: Vec<f32>,
}

pub struct SparseScratch {
    moves: Vec<Move>,
    tokens: Vec<[f32; TOKEN_WIDTH]>,
    normalized: Vec<[f32; TOKEN_WIDTH]>,
    q: Vec<[f32; TOKEN_WIDTH]>,
    k: Vec<[f32; TOKEN_WIDTH]>,
    v: Vec<[f32; TOKEN_WIDTH]>,
    next: Vec<[f32; TOKEN_WIDTH]>,
    policy_logits: Vec<f32>,
    policy_contexts: Vec<[f32; TOKEN_WIDTH]>,
    value_features: [f32; TOKEN_WIDTH],
    value_probabilities: [f32; 3],
    line_tokens: Vec<Vec<usize>>,
    active_lines: Vec<usize>,
}

#[derive(Clone)]
struct CachedLayer {
    output: Vec<[f32; TOKEN_WIDTH]>,
    k: Vec<[f32; TOKEN_WIDTH]>,
    v: Vec<[f32; TOKEN_WIDTH]>,
}

#[derive(Clone)]
struct PerspectiveCache {
    base: Vec<[f32; TOKEN_WIDTH]>,
    layers: Vec<CachedLayer>,
}

#[derive(Clone)]
pub struct EvalCache {
    moves: Vec<Move>,
    black: PerspectiveCache,
    white: PerspectiveCache,
}

impl Default for SparseScratch {
    fn default() -> Self {
        Self {
            moves: Vec::with_capacity(CELL_COUNT),
            tokens: Vec::with_capacity(CELL_COUNT),
            normalized: Vec::with_capacity(CELL_COUNT),
            q: Vec::with_capacity(CELL_COUNT),
            k: Vec::with_capacity(CELL_COUNT),
            v: Vec::with_capacity(CELL_COUNT),
            next: Vec::with_capacity(CELL_COUNT),
            policy_logits: Vec::with_capacity(CELL_COUNT),
            policy_contexts: Vec::with_capacity(CELL_COUNT),
            value_features: [0.0; TOKEN_WIDTH],
            value_probabilities: [0.0; 3],
            line_tokens: (0..LINE_COUNT).map(|_| Vec::with_capacity(15)).collect(),
            active_lines: Vec::with_capacity(LINE_COUNT),
        }
    }
}

impl SparseScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for SparseTransformerModel {
    fn default() -> Self {
        Self::random(TOKEN_WIDTH, 20260730)
    }
}

impl SparseTransformerModel {
    pub fn random(_requested_width: usize, seed: u64) -> Self {
        let mut rng = SplitMix64(seed);
        let embedding_scale = (2.0 / TOKEN_WIDTH as f32).sqrt();
        let matrix = |rng: &mut SplitMix64, input: usize, output: usize| {
            let scale = (2.0 / input as f32).sqrt() * 0.5;
            (0..input * output)
                .map(|_| rng.weight(scale))
                .collect::<Vec<_>>()
        };
        let blocks = (0..LAYERS)
            .map(|_| Block {
                q: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
                k: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
                v: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
                output: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
                ff_up: matrix(&mut rng, TOKEN_WIDTH, FF_WIDTH),
                ff_down: matrix(&mut rng, FF_WIDTH, TOKEN_WIDTH),
            })
            .collect();
        let position_embedding = (0..CELL_COUNT * TOKEN_WIDTH)
            .map(|_| rng.weight(embedding_scale))
            .collect::<Vec<_>>();
        let policy_query = matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH);
        let mut projected_policy_query = vec![0.0; CELL_COUNT * TOKEN_WIDTH];
        for position in 0..CELL_COUNT {
            let output: &mut [f32; TOKEN_WIDTH] = (&mut projected_policy_query
                [position * TOKEN_WIDTH..(position + 1) * TOKEN_WIDTH])
                .try_into()
                .unwrap();
            project(
                array_at(&position_embedding, position),
                &policy_query,
                output,
            );
        }
        Self {
            stone_embedding: (0..2 * TOKEN_WIDTH)
                .map(|_| rng.weight(embedding_scale))
                .collect(),
            position_embedding,
            blocks,
            policy_query,
            projected_policy_query,
            policy_key: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
            policy_value: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
            policy_output: (0..TOKEN_WIDTH)
                .map(|_| rng.weight(embedding_scale))
                .collect(),
            policy_bias: vec![0.0; CELL_COUNT],
            value_hidden: matrix(&mut rng, TOKEN_WIDTH, TOKEN_WIDTH),
            value_output: matrix(&mut rng, TOKEN_WIDTH, 3),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if let Some(parent) = path
            .as_ref()
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let vars = VarMap::new();
        insert(&vars, "format_version", &[1.0], (1,))?;
        insert(
            &vars,
            "stone_embedding",
            &self.stone_embedding,
            (2, TOKEN_WIDTH),
        )?;
        insert(
            &vars,
            "position_embedding",
            &self.position_embedding,
            (CELL_COUNT, TOKEN_WIDTH),
        )?;
        for (layer, block) in self.blocks.iter().enumerate() {
            insert(
                &vars,
                &format!("block.{layer}.q"),
                &block.q,
                (TOKEN_WIDTH, TOKEN_WIDTH),
            )?;
            insert(
                &vars,
                &format!("block.{layer}.k"),
                &block.k,
                (TOKEN_WIDTH, TOKEN_WIDTH),
            )?;
            insert(
                &vars,
                &format!("block.{layer}.v"),
                &block.v,
                (TOKEN_WIDTH, TOKEN_WIDTH),
            )?;
            insert(
                &vars,
                &format!("block.{layer}.output"),
                &block.output,
                (TOKEN_WIDTH, TOKEN_WIDTH),
            )?;
            insert(
                &vars,
                &format!("block.{layer}.ff_up"),
                &block.ff_up,
                (TOKEN_WIDTH, FF_WIDTH),
            )?;
            insert(
                &vars,
                &format!("block.{layer}.ff_down"),
                &block.ff_down,
                (FF_WIDTH, TOKEN_WIDTH),
            )?;
        }
        for (name, data) in [
            ("policy_query", &self.policy_query),
            ("policy_key", &self.policy_key),
            ("policy_value", &self.policy_value),
            ("value_hidden", &self.value_hidden),
        ] {
            insert(&vars, name, data, (TOKEN_WIDTH, TOKEN_WIDTH))?;
        }
        insert(&vars, "policy_output", &self.policy_output, (TOKEN_WIDTH,))?;
        insert(&vars, "policy_bias", &self.policy_bias, (CELL_COUNT,))?;
        insert(&vars, "value_output", &self.value_output, (TOKEN_WIDTH, 3))?;
        vars.save(path).map_err(candle_error)
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path.as_ref()).map_err(candle_error)?
        };
        if load(&tensors, "format_version")?.first().copied() != Some(1.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "不支持的稀疏 Transformer 模型版本",
            ));
        }
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            blocks.push(Block {
                q: load(&tensors, &format!("block.{layer}.q"))?,
                k: load(&tensors, &format!("block.{layer}.k"))?,
                v: load(&tensors, &format!("block.{layer}.v"))?,
                output: load(&tensors, &format!("block.{layer}.output"))?,
                ff_up: load(&tensors, &format!("block.{layer}.ff_up"))?,
                ff_down: load(&tensors, &format!("block.{layer}.ff_down"))?,
            });
        }
        let mut model = Self {
            stone_embedding: load(&tensors, "stone_embedding")?,
            position_embedding: load(&tensors, "position_embedding")?,
            blocks,
            policy_query: load(&tensors, "policy_query")?,
            projected_policy_query: vec![0.0; CELL_COUNT * TOKEN_WIDTH],
            policy_key: load(&tensors, "policy_key")?,
            policy_value: load(&tensors, "policy_value")?,
            policy_output: load(&tensors, "policy_output")?,
            policy_bias: load(&tensors, "policy_bias")?,
            value_hidden: load(&tensors, "value_hidden")?,
            value_output: load(&tensors, "value_output")?,
        };
        model.validate()?;
        model.rebuild_query_cache();
        Ok(model)
    }

    pub fn update_ema(&mut self, online: &Self, decay: f32) {
        let decay = decay.clamp(0.0, 1.0);
        let keep = 1.0 - decay;
        fn blend(left: &mut [f32], right: &[f32], decay: f32, keep: f32) {
            for (left, right) in left.iter_mut().zip(right) {
                *left = decay * *left + keep * *right;
            }
        }
        blend(
            &mut self.stone_embedding,
            &online.stone_embedding,
            decay,
            keep,
        );
        blend(
            &mut self.position_embedding,
            &online.position_embedding,
            decay,
            keep,
        );
        for (left, right) in self.blocks.iter_mut().zip(&online.blocks) {
            blend(&mut left.q, &right.q, decay, keep);
            blend(&mut left.k, &right.k, decay, keep);
            blend(&mut left.v, &right.v, decay, keep);
            blend(&mut left.output, &right.output, decay, keep);
            blend(&mut left.ff_up, &right.ff_up, decay, keep);
            blend(&mut left.ff_down, &right.ff_down, decay, keep);
        }
        blend(&mut self.policy_query, &online.policy_query, decay, keep);
        blend(&mut self.policy_key, &online.policy_key, decay, keep);
        blend(&mut self.policy_value, &online.policy_value, decay, keep);
        blend(&mut self.policy_output, &online.policy_output, decay, keep);
        blend(&mut self.policy_bias, &online.policy_bias, decay, keep);
        blend(&mut self.value_hidden, &online.value_hidden, decay, keep);
        blend(&mut self.value_output, &online.value_output, decay, keep);
        self.rebuild_query_cache();
    }

    fn rebuild_query_cache(&mut self) {
        for position in 0..CELL_COUNT {
            let input = *array_at(&self.position_embedding, position);
            let output: &mut [f32; TOKEN_WIDTH] = (&mut self.projected_policy_query
                [position * TOKEN_WIDTH..(position + 1) * TOKEN_WIDTH])
                .try_into()
                .unwrap();
            project(&input, &self.policy_query, output);
        }
    }

    fn validate(&self) -> io::Result<()> {
        let square = TOKEN_WIDTH * TOKEN_WIDTH;
        let valid = self.stone_embedding.len() == 2 * TOKEN_WIDTH
            && self.position_embedding.len() == CELL_COUNT * TOKEN_WIDTH
            && self.blocks.len() == LAYERS
            && self.blocks.iter().all(|block| {
                block.q.len() == square
                    && block.k.len() == square
                    && block.v.len() == square
                    && block.output.len() == square
                    && block.ff_up.len() == TOKEN_WIDTH * FF_WIDTH
                    && block.ff_down.len() == FF_WIDTH * TOKEN_WIDTH
            })
            && self.policy_query.len() == square
            && self.policy_key.len() == square
            && self.policy_value.len() == square
            && self.policy_output.len() == TOKEN_WIDTH
            && self.policy_bias.len() == CELL_COUNT
            && self.value_hidden.len() == square
            && self.value_output.len() == TOKEN_WIDTH * 3;
        if valid {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "稀疏 Transformer 模型张量尺寸错误",
            ))
        }
    }

    pub fn evaluate(&self, board: &Board) -> (Vec<(Move, f32)>, f32) {
        self.evaluate_with_scratch(board, 1.0, &mut SparseScratch::new())
    }

    pub fn cache(&self, board: &Board) -> EvalCache {
        let moves = board
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, &stone)| (stone != 0).then_some(Move(index)))
            .collect::<Vec<_>>();
        EvalCache {
            black: self.build_perspective_cache(board, &moves, Player::Black),
            white: self.build_perspective_cache(board, &moves, Player::White),
            moves,
        }
    }

    pub fn cache_after_move(&self, parent: &EvalCache, mv: Move, player: Player) -> EvalCache {
        let insertion = parent.moves.partition_point(|move_| move_.0 < mv.0);
        let mut moves = parent.moves.clone();
        moves.insert(insertion, mv);
        EvalCache {
            black: self.update_perspective_cache(
                &parent.black,
                &moves,
                insertion,
                mv,
                player,
                Player::Black,
            ),
            white: self.update_perspective_cache(
                &parent.white,
                &moves,
                insertion,
                mv,
                player,
                Player::White,
            ),
            moves,
        }
    }

    pub fn evaluate_cache_with_scratch(
        &self,
        board: &Board,
        cache: &EvalCache,
        policy_temperature: f32,
        scratch: &mut SparseScratch,
    ) -> (Vec<(Move, f32)>, f32) {
        let perspective = match board.to_move() {
            Player::Black => &cache.black,
            Player::White => &cache.white,
        };
        scratch.moves.clone_from(&cache.moves);
        scratch.tokens.clone_from(
            perspective
                .layers
                .last()
                .map(|layer| &layer.output)
                .unwrap_or(&perspective.base),
        );
        rebuild_line_index(scratch);
        let policy = self.policy(board, policy_temperature, scratch);
        let value = self.value(scratch);
        (policy, value)
    }

    fn build_perspective_cache(
        &self,
        board: &Board,
        moves: &[Move],
        perspective: Player,
    ) -> PerspectiveCache {
        let base = moves
            .iter()
            .map(|&mv| self.base_token(mv, board.cells()[mv.0], perspective))
            .collect::<Vec<_>>();
        let mut input = base.clone();
        let mut layers = Vec::with_capacity(LAYERS);
        for block in &self.blocks {
            let layer = build_cached_layer(block, moves, &input);
            input = layer.output.clone();
            layers.push(layer);
        }
        PerspectiveCache { base, layers }
    }

    fn update_perspective_cache(
        &self,
        parent: &PerspectiveCache,
        moves: &[Move],
        insertion: usize,
        mv: Move,
        player: Player,
        perspective: Player,
    ) -> PerspectiveCache {
        let token = self.base_token(mv, player.stone(), perspective);
        let base = inserted(&parent.base, insertion, token);
        let mut input = base.clone();
        let mut changed = vec![false; moves.len()];
        changed[insertion] = true;
        let mut layers = Vec::with_capacity(LAYERS);
        for (block, old) in self.blocks.iter().zip(&parent.layers) {
            let mut k = inserted(&old.k, insertion, [0.0; TOKEN_WIDTH]);
            let mut v = inserted(&old.v, insertion, [0.0; TOKEN_WIDTH]);
            for index in 0..moves.len() {
                if changed[index] {
                    project_kv(block, &input[index], &mut k[index], &mut v[index]);
                }
            }
            let mut affected = vec![false; moves.len()];
            for index in 0..moves.len() {
                affected[index] = changed
                    .iter()
                    .enumerate()
                    .any(|(other, &changed)| changed && aligned(moves[index], moves[other]));
            }
            let mut output = inserted(&old.output, insertion, [0.0; TOKEN_WIDTH]);
            for index in 0..moves.len() {
                if affected[index] {
                    output[index] = compute_block_output(block, moves, &input, &k, &v, index);
                }
            }
            input = output.clone();
            changed = affected;
            layers.push(CachedLayer { output, k, v });
        }
        PerspectiveCache { base, layers }
    }

    fn base_token(&self, mv: Move, stone: i8, perspective: Player) -> [f32; TOKEN_WIDTH] {
        let side = usize::from(stone != perspective.stone());
        let mut token = [0.0; TOKEN_WIDTH];
        for dimension in 0..TOKEN_WIDTH {
            token[dimension] = self.stone_embedding[side * TOKEN_WIDTH + dimension]
                + self.position_embedding[mv.0 * TOKEN_WIDTH + dimension];
        }
        token
    }

    pub fn evaluate_with_scratch(
        &self,
        board: &Board,
        policy_temperature: f32,
        scratch: &mut SparseScratch,
    ) -> (Vec<(Move, f32)>, f32) {
        self.encode_stones(board, scratch);
        for block in &self.blocks {
            apply_block(block, scratch);
        }
        let policy = self.policy(board, policy_temperature, scratch);
        let value = self.value(scratch);
        (policy, value)
    }

    fn encode_stones(&self, board: &Board, scratch: &mut SparseScratch) {
        scratch.moves.clear();
        scratch.tokens.clear();
        for (index, &stone) in board.cells().iter().enumerate() {
            if stone == 0 {
                continue;
            }
            scratch.moves.push(Move(index));
            scratch
                .tokens
                .push(self.base_token(Move(index), stone, board.to_move()));
        }
        rebuild_line_index(scratch);
    }

    fn policy(
        &self,
        board: &Board,
        temperature: f32,
        scratch: &mut SparseScratch,
    ) -> Vec<(Move, f32)> {
        scratch.k.resize(scratch.tokens.len(), [0.0; TOKEN_WIDTH]);
        scratch.v.resize(scratch.tokens.len(), [0.0; TOKEN_WIDTH]);
        for (index, token) in scratch.tokens.iter().enumerate() {
            project(token, &self.policy_key, &mut scratch.k[index]);
            project(token, &self.policy_value, &mut scratch.v[index]);
        }
        let legal = board.search_candidates();
        scratch.policy_logits.clear();
        scratch.policy_contexts.clear();
        scratch.policy_logits.reserve(legal.len());
        scratch.policy_contexts.reserve(legal.len());
        let scale = (HEAD_WIDTH as f32).sqrt().recip();
        for &mv in &legal {
            let position = array_at(&self.position_embedding, mv.0);
            let query = array_at(&self.projected_policy_query, mv.0);
            let mut context = [0.0; TOKEN_WIDTH];
            let lines = line_ids(mv);
            attend_all_heads(
                query,
                scale,
                lines
                    .iter()
                    .flat_map(|&line| scratch.line_tokens[line].iter())
                    .map(|&index| (&scratch.k[index], &scratch.v[index])),
                &mut context,
            );
            for dimension in 0..TOKEN_WIDTH {
                context[dimension] += position[dimension];
            }
            scratch
                .policy_logits
                .push(dot(&context, &self.policy_output) + self.policy_bias[mv.0]);
            scratch.policy_contexts.push(context);
        }
        softmax_moves(&legal, &scratch.policy_logits, temperature)
    }

    fn value(&self, scratch: &mut SparseScratch) -> f32 {
        if scratch.tokens.is_empty() {
            return 0.0;
        }
        let mut pooled = [0.0; TOKEN_WIDTH];
        for token in &scratch.tokens {
            for dimension in 0..TOKEN_WIDTH {
                pooled[dimension] += token[dimension] / scratch.tokens.len() as f32;
            }
        }
        project(&pooled, &self.value_hidden, &mut scratch.value_features);
        for value in &mut scratch.value_features {
            *value = value.max(0.0);
        }
        let mut logits = [0.0; 3];
        for (output, logit) in logits.iter_mut().enumerate() {
            for dimension in 0..TOKEN_WIDTH {
                *logit +=
                    scratch.value_features[dimension] * self.value_output[dimension * 3 + output];
            }
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        scratch.value_probabilities = logits.map(|value| (value - maximum).exp());
        let sum = scratch.value_probabilities.iter().sum::<f32>();
        for probability in &mut scratch.value_probabilities {
            *probability /= sum;
        }
        scratch.value_probabilities[0] - scratch.value_probabilities[2]
    }

    /// CPU 友好的在线头部更新。Transformer 主体保持稳定，Policy/Value 输出头参与 SGD；
    /// 后续训练内核会沿同一接口逐步加入稀疏 Attention 的完整反向传播。
    pub(crate) fn train_heads(
        &mut self,
        board: &Board,
        target_policy: &[(Move, f32)],
        target_value: f32,
        learning_rate: f32,
        scratch: &mut SparseScratch,
    ) -> (f32, f32) {
        let (policy, _) = self.evaluate_with_scratch(board, 1.0, scratch);
        let target_sum = target_policy
            .iter()
            .map(|(_, probability)| probability.max(0.0))
            .sum::<f32>()
            .max(1e-12);
        let mut policy_loss = 0.0;
        for (index, &(mv, probability)) in policy.iter().enumerate() {
            let target = target_policy
                .iter()
                .find_map(|&(target_move, value)| (target_move == mv).then_some(value.max(0.0)))
                .unwrap_or(0.0)
                / target_sum;
            policy_loss -= target * probability.max(1e-12).ln();
            let gradient = probability - target;
            for dimension in 0..TOKEN_WIDTH {
                self.policy_output[dimension] -=
                    learning_rate * gradient * scratch.policy_contexts[index][dimension];
            }
            self.policy_bias[mv.0] -= learning_rate * gradient;
        }
        let target_wdl = if target_value > 0.5 {
            [1.0, 0.0, 0.0]
        } else if target_value < -0.5 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut value_loss = 0.0;
        for output in 0..3 {
            value_loss -= target_wdl[output] * scratch.value_probabilities[output].max(1e-12).ln();
            let gradient = scratch.value_probabilities[output] - target_wdl[output];
            for dimension in 0..TOKEN_WIDTH {
                self.value_output[dimension * 3 + output] -=
                    learning_rate * gradient * scratch.value_features[dimension];
            }
        }
        (policy_loss, value_loss)
    }
}

fn inserted<T: Copy>(values: &[T], index: usize, value: T) -> Vec<T> {
    let mut output = Vec::with_capacity(values.len() + 1);
    output.extend_from_slice(&values[..index]);
    output.push(value);
    output.extend_from_slice(&values[index..]);
    output
}

fn build_cached_layer(block: &Block, moves: &[Move], input: &[[f32; TOKEN_WIDTH]]) -> CachedLayer {
    let mut k = vec![[0.0; TOKEN_WIDTH]; input.len()];
    let mut v = vec![[0.0; TOKEN_WIDTH]; input.len()];
    for index in 0..input.len() {
        project_kv(block, &input[index], &mut k[index], &mut v[index]);
    }
    let output = (0..input.len())
        .map(|index| compute_block_output(block, moves, input, &k, &v, index))
        .collect();
    CachedLayer { output, k, v }
}

fn project_kv(
    block: &Block,
    input: &[f32; TOKEN_WIDTH],
    k: &mut [f32; TOKEN_WIDTH],
    v: &mut [f32; TOKEN_WIDTH],
) {
    let mut normalized = [0.0; TOKEN_WIDTH];
    rms_norm(input, &mut normalized);
    project(&normalized, &block.k, k);
    project(&normalized, &block.v, v);
}

fn compute_block_output(
    block: &Block,
    moves: &[Move],
    input: &[[f32; TOKEN_WIDTH]],
    k: &[[f32; TOKEN_WIDTH]],
    v: &[[f32; TOKEN_WIDTH]],
    index: usize,
) -> [f32; TOKEN_WIDTH] {
    let mut normalized = [0.0; TOKEN_WIDTH];
    rms_norm(&input[index], &mut normalized);
    let mut q = [0.0; TOKEN_WIDTH];
    project(&normalized, &block.q, &mut q);
    let mut attention = [0.0; TOKEN_WIDTH];
    attend_all_heads(
        &q,
        (HEAD_WIDTH as f32).sqrt().recip(),
        moves
            .iter()
            .enumerate()
            .filter_map(|(other, &mv)| aligned(moves[index], mv).then_some((&k[other], &v[other]))),
        &mut attention,
    );
    let mut projected = [0.0; TOKEN_WIDTH];
    project(&attention, &block.output, &mut projected);
    for dimension in 0..TOKEN_WIDTH {
        projected[dimension] += input[index][dimension];
    }
    rms_norm(&projected, &mut normalized);
    let mut expanded = [0.0; FF_WIDTH];
    project_wide(&normalized, &block.ff_up, &mut expanded);
    for value in &mut expanded {
        *value = value.max(0.0);
    }
    let mut reduced = [0.0; TOKEN_WIDTH];
    project_wide(&expanded, &block.ff_down, &mut reduced);
    for dimension in 0..TOKEN_WIDTH {
        projected[dimension] += reduced[dimension];
    }
    projected
}

fn rebuild_line_index(scratch: &mut SparseScratch) {
    for &line in &scratch.active_lines {
        scratch.line_tokens[line].clear();
    }
    scratch.active_lines.clear();
    for (token_index, &mv) in scratch.moves.iter().enumerate() {
        for line in line_ids(mv) {
            if scratch.line_tokens[line].is_empty() {
                scratch.active_lines.push(line);
            }
            scratch.line_tokens[line].push(token_index);
        }
    }
}

fn apply_block(block: &Block, scratch: &mut SparseScratch) {
    let count = scratch.tokens.len();
    scratch.normalized.resize(count, [0.0; TOKEN_WIDTH]);
    scratch.q.resize(count, [0.0; TOKEN_WIDTH]);
    scratch.k.resize(count, [0.0; TOKEN_WIDTH]);
    scratch.v.resize(count, [0.0; TOKEN_WIDTH]);
    scratch.next.resize(count, [0.0; TOKEN_WIDTH]);
    for index in 0..count {
        rms_norm(&scratch.tokens[index], &mut scratch.normalized[index]);
        project(&scratch.normalized[index], &block.q, &mut scratch.q[index]);
        project(&scratch.normalized[index], &block.k, &mut scratch.k[index]);
        project(&scratch.normalized[index], &block.v, &mut scratch.v[index]);
    }
    let scale = (HEAD_WIDTH as f32).sqrt().recip();
    for index in 0..count {
        let mut attention = [0.0; TOKEN_WIDTH];
        attend_all_heads(
            &scratch.q[index],
            scale,
            scratch.moves.iter().enumerate().filter_map(|(other, &mv)| {
                aligned(scratch.moves[index], mv).then_some((&scratch.k[other], &scratch.v[other]))
            }),
            &mut attention,
        );
        let mut projected = [0.0; TOKEN_WIDTH];
        project(&attention, &block.output, &mut projected);
        for dimension in 0..TOKEN_WIDTH {
            scratch.next[index][dimension] =
                scratch.tokens[index][dimension] + projected[dimension];
        }
        let mut normalized = [0.0; TOKEN_WIDTH];
        rms_norm(&scratch.next[index], &mut normalized);
        let mut expanded = [0.0; FF_WIDTH];
        project_wide(&normalized, &block.ff_up, &mut expanded);
        for value in &mut expanded {
            *value = value.max(0.0);
        }
        let mut reduced = [0.0; TOKEN_WIDTH];
        project_wide(&expanded, &block.ff_down, &mut reduced);
        for dimension in 0..TOKEN_WIDTH {
            scratch.next[index][dimension] += reduced[dimension];
        }
    }
    std::mem::swap(&mut scratch.tokens, &mut scratch.next);
}

fn attend_all_heads<'a>(
    query: &[f32; TOKEN_WIDTH],
    scale: f32,
    keys: impl Iterator<Item = (&'a [f32; TOKEN_WIDTH], &'a [f32; TOKEN_WIDTH])>,
    output: &mut [f32; TOKEN_WIDTH],
) {
    let mut maximum = [f32::NEG_INFINITY; HEADS];
    let mut denominator = [0.0; HEADS];
    for (key, value) in keys {
        for head in 0..HEADS {
            let start = head * HEAD_WIDTH;
            let end = start + HEAD_WIDTH;
            let score = dot(&query[start..end], &key[start..end]) * scale;
            if score <= maximum[head] {
                let weight = (score - maximum[head]).exp();
                denominator[head] += weight;
                for dimension in start..end {
                    output[dimension] += weight * value[dimension];
                }
            } else {
                let rescale = (maximum[head] - score).exp();
                denominator[head] = denominator[head] * rescale + 1.0;
                for dimension in start..end {
                    output[dimension] = output[dimension] * rescale + value[dimension];
                }
                maximum[head] = score;
            }
        }
    }
    for head in 0..HEADS {
        if denominator[head] > 0.0 {
            let start = head * HEAD_WIDTH;
            let end = start + HEAD_WIDTH;
            for value in &mut output[start..end] {
                *value /= denominator[head];
            }
        }
    }
}

fn aligned(left: Move, right: Move) -> bool {
    ALIGNED_MASKS[left.0][right.0 / 64] & (1_u64 << (right.0 % 64)) != 0
}

fn line_ids(mv: Move) -> [usize; 4] {
    CELL_LINES[mv.0].map(usize::from)
}

fn array_at(values: &[f32], index: usize) -> &[f32; TOKEN_WIDTH] {
    values[index * TOKEN_WIDTH..(index + 1) * TOKEN_WIDTH]
        .try_into()
        .unwrap()
}

fn project(input: &[f32; TOKEN_WIDTH], weights: &[f32], output: &mut [f32; TOKEN_WIDTH]) {
    project_wide(input, weights, output);
}

fn project_wide(input: &[f32], weights: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if output.len().is_multiple_of(4) && output.len() <= 64 {
            // SAFETY: AArch64 guarantees NEON; slice bounds and output width are checked here.
            unsafe { project_wide_neon(input, weights, output) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if output.len().is_multiple_of(8)
            && output.len() <= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime feature detection and slice bounds satisfy the kernel contract.
            unsafe { project_wide_avx2(input, weights, output) };
            return;
        }
    }
    project_wide_scalar(input, weights, output);
}

fn project_wide_scalar(input: &[f32], weights: &[f32], output: &mut [f32]) {
    output.fill(0.0);
    for (input_index, &value) in input.iter().enumerate() {
        let row = &weights[input_index * output.len()..(input_index + 1) * output.len()];
        for output_index in 0..output.len() {
            output[output_index] += value * row[output_index];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_wide_neon(input: &[f32], weights: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    let chunks = output.len() / 4;
    let mut sums: [float32x4_t; 16] = [vdupq_n_f32(0.0); 16];
    for (input_index, &value) in input.iter().enumerate() {
        let scalar = vdupq_n_f32(value);
        let row = input_index * output.len();
        for (chunk, sum) in sums[..chunks].iter_mut().enumerate() {
            let weight = unsafe { vld1q_f32(weights.as_ptr().add(row + chunk * 4)) };
            *sum = vfmaq_f32(*sum, scalar, weight);
        }
    }
    for (chunk, sum) in sums[..chunks].iter().enumerate() {
        unsafe { vst1q_f32(output.as_mut_ptr().add(chunk * 4), *sum) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project_wide_avx2(input: &[f32], weights: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };
    let chunks = output.len() / 8;
    let mut sums: [__m256; 8] = [_mm256_setzero_ps(); 8];
    for (input_index, &value) in input.iter().enumerate() {
        let scalar = _mm256_set1_ps(value);
        let row = input_index * output.len();
        for (chunk, sum) in sums[..chunks].iter_mut().enumerate() {
            let weight = unsafe { _mm256_loadu_ps(weights.as_ptr().add(row + chunk * 8)) };
            *sum = _mm256_fmadd_ps(scalar, weight, *sum);
        }
    }
    for (chunk, sum) in sums[..chunks].iter().enumerate() {
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(chunk * 8), *sum) };
    }
}

fn rms_norm(input: &[f32; TOKEN_WIDTH], output: &mut [f32; TOKEN_WIDTH]) {
    let rms =
        (input.iter().map(|value| value * value).sum::<f32>() / TOKEN_WIDTH as f32 + 1e-6).sqrt();
    for dimension in 0..TOKEN_WIDTH {
        output[dimension] = input[dimension] / rms;
    }
}

fn softmax_moves(moves: &[Move], logits: &[f32], temperature: f32) -> Vec<(Move, f32)> {
    if moves.is_empty() {
        return Vec::new();
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inverse_temperature = temperature.max(1e-3).recip();
    let sum = logits
        .iter()
        .map(|value| ((value - maximum) * inverse_temperature).exp())
        .sum::<f32>();
    moves
        .iter()
        .copied()
        .zip(
            logits
                .iter()
                .map(|value| ((value - maximum) * inverse_temperature).exp() / sum),
        )
        .collect()
}

fn insert(vars: &VarMap, name: &str, data: &[f32], shape: impl Into<Shape>) -> io::Result<()> {
    let var = Var::from_slice(data, shape, &Device::Cpu).map_err(candle_error)?;
    vars.data()
        .lock()
        .unwrap_or_else(|_| panic!("模型变量锁损坏"))
        .insert(name.into(), var);
    Ok(())
}

fn load(tensors: &candle_core::safetensors::MmapedSafetensors, name: &str) -> io::Result<Vec<f32>> {
    let tensor = tensors.load(name, &Device::Cpu).map_err(candle_error)?;
    if tensor.dtype() != DType::F32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("张量 `{name}` 不是 F32"),
        ));
    }
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_error)
}

fn candle_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn weight(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    #[test]
    fn sparse_model_returns_every_legal_move() {
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let model = SparseTransformerModel::random(TOKEN_WIDTH, 7);
        let (policy, value) = model.evaluate(&board);
        assert_eq!(policy.len(), CELL_COUNT - 1);
        assert!(
            (policy
                .iter()
                .map(|(_, probability)| probability)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-5
        );
        assert!(value.is_finite());
    }

    #[test]
    fn blocked_stones_remain_tokens() {
        let stones = [
            (Move::new(7, 7).unwrap(), Player::Black),
            (Move::new(7, 6).unwrap(), Player::White),
            (Move::new(7, 8).unwrap(), Player::White),
            (Move::new(6, 7).unwrap(), Player::White),
            (Move::new(8, 7).unwrap(), Player::White),
            (Move::new(0, 0).unwrap(), Player::Black),
            (Move::new(0, 14).unwrap(), Player::Black),
            (Move::new(14, 0).unwrap(), Player::Black),
        ];
        let board = Board::from_stones(&stones).unwrap();
        let model = SparseTransformerModel::random(TOKEN_WIDTH, 9);
        let mut scratch = SparseScratch::default();
        model.encode_stones(&board, &mut scratch);
        assert_eq!(scratch.tokens.len(), stones.len());
    }

    #[test]
    fn sparse_model_roundtrip_preserves_outputs() {
        let path = std::env::temp_dir().join(format!(
            "gomoku-sparse-roundtrip-{}-{}.safetensors",
            std::process::id(),
            SplitMix64(11).next()
        ));
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        assert!(board.play(Move::new(7, 8).unwrap()));
        let model = SparseTransformerModel::random(TOKEN_WIDTH, 13);
        let expected = model.evaluate(&board);
        model.save(&path).unwrap();
        let restored = SparseTransformerModel::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.evaluate(&board), expected);
    }

    #[test]
    fn line_index_matches_alignment_scan_for_empty_candidates() {
        let stones = [
            (Move::new(7, 7).unwrap(), Player::Black),
            (Move::new(2, 7).unwrap(), Player::White),
            (Move::new(9, 9).unwrap(), Player::Black),
            (Move::new(4, 10).unwrap(), Player::White),
        ];
        let board = Board::from_stones(&stones).unwrap();
        let model = SparseTransformerModel::random(TOKEN_WIDTH, 17);
        let mut scratch = SparseScratch::new();
        model.encode_stones(&board, &mut scratch);
        for candidate in board.search_candidates() {
            let mut indexed = line_ids(candidate)
                .iter()
                .flat_map(|&line| scratch.line_tokens[line].iter().copied())
                .collect::<Vec<_>>();
            indexed.sort_unstable();
            let scanned = scratch
                .moves
                .iter()
                .enumerate()
                .filter_map(|(index, &stone)| aligned(candidate, stone).then_some(index))
                .collect::<Vec<_>>();
            assert_eq!(indexed, scanned);
        }
    }

    #[test]
    fn incremental_dual_perspective_cache_matches_full_forward() {
        let model = SparseTransformerModel::random(TOKEN_WIDTH, 23);
        let mut board = Board::new();
        let mut cache = model.cache(&board);
        let mut scratch = SparseScratch::new();
        for text in ["h8", "i8", "h9", "g7", "j10", "f10", "k7", "e6"] {
            let mv = Move::parse(text).unwrap();
            let player = board.to_move();
            cache = model.cache_after_move(&cache, mv, player);
            assert!(board.play(mv));
            let full = model.evaluate(&board);
            let incremental = model.evaluate_cache_with_scratch(&board, &cache, 1.0, &mut scratch);
            assert_eq!(full.0.len(), incremental.0.len());
            for ((full_move, full_probability), (cached_move, cached_probability)) in
                full.0.iter().zip(&incremental.0)
            {
                assert_eq!(full_move, cached_move);
                assert!((full_probability - cached_probability).abs() < 1e-6);
            }
            assert!((full.1 - incremental.1).abs() < 1e-6);
        }
    }
}
