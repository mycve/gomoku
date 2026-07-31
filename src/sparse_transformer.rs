use crate::game::{Board, CELL_COUNT, Move};

pub const TOKEN_WIDTH: usize = 32;
pub const HEADS: usize = 2;
pub const HEAD_WIDTH: usize = TOKEN_WIDTH / HEADS;
pub const LAYERS: usize = 2;
pub const FF_WIDTH: usize = 64;

#[derive(Clone)]
struct Block {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    output: Vec<f32>,
    ff_up: Vec<f32>,
    ff_down: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseTransformerModel {
    stone_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    blocks: Vec<Block>,
    #[allow(dead_code)] // 训练接入后由优化器更新；推理使用加载时生成的预投影缓存。
    policy_query: Vec<f32>,
    projected_policy_query: Vec<f32>,
    policy_key: Vec<f32>,
    policy_value: Vec<f32>,
    policy_output: Vec<f32>,
    policy_bias: Vec<f32>,
    value_hidden: Vec<f32>,
    value_output: Vec<f32>,
}

#[derive(Default)]
pub struct SparseScratch {
    moves: Vec<Move>,
    tokens: Vec<[f32; TOKEN_WIDTH]>,
    normalized: Vec<[f32; TOKEN_WIDTH]>,
    q: Vec<[f32; TOKEN_WIDTH]>,
    k: Vec<[f32; TOKEN_WIDTH]>,
    v: Vec<[f32; TOKEN_WIDTH]>,
    next: Vec<[f32; TOKEN_WIDTH]>,
    policy_logits: Vec<f32>,
}

impl SparseTransformerModel {
    pub fn random(seed: u64) -> Self {
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

    pub fn evaluate(&self, board: &Board, scratch: &mut SparseScratch) -> (Vec<(Move, f32)>, f32) {
        self.encode_stones(board, scratch);
        for block in &self.blocks {
            apply_block(block, scratch);
        }
        let policy = self.policy(board, scratch);
        let value = self.value(scratch);
        (policy, value)
    }

    fn encode_stones(&self, board: &Board, scratch: &mut SparseScratch) {
        scratch.moves.clear();
        scratch.tokens.clear();
        let us = board.to_move().stone();
        for (index, &stone) in board.cells().iter().enumerate() {
            if stone == 0 {
                continue;
            }
            let side = usize::from(stone != us);
            let mut token = [0.0; TOKEN_WIDTH];
            for dimension in 0..TOKEN_WIDTH {
                token[dimension] = self.stone_embedding[side * TOKEN_WIDTH + dimension]
                    + self.position_embedding[index * TOKEN_WIDTH + dimension];
            }
            scratch.moves.push(Move(index));
            scratch.tokens.push(token);
        }
    }

    fn policy(&self, board: &Board, scratch: &mut SparseScratch) -> Vec<(Move, f32)> {
        scratch.k.resize(scratch.tokens.len(), [0.0; TOKEN_WIDTH]);
        scratch.v.resize(scratch.tokens.len(), [0.0; TOKEN_WIDTH]);
        for (index, token) in scratch.tokens.iter().enumerate() {
            project(token, &self.policy_key, &mut scratch.k[index]);
            project(token, &self.policy_value, &mut scratch.v[index]);
        }
        let legal = board.search_candidates();
        scratch.policy_logits.clear();
        scratch.policy_logits.reserve(legal.len());
        let scale = (HEAD_WIDTH as f32).sqrt().recip();
        for &mv in &legal {
            let position = array_at(&self.position_embedding, mv.0);
            let query = array_at(&self.projected_policy_query, mv.0);
            let mut context = [0.0; TOKEN_WIDTH];
            for head in 0..HEADS {
                attend(
                    query,
                    head,
                    scale,
                    scratch
                        .moves
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &stone)| {
                            aligned(mv, stone).then_some((&scratch.k[index], &scratch.v[index]))
                        }),
                    &mut context,
                );
            }
            for dimension in 0..TOKEN_WIDTH {
                context[dimension] += position[dimension];
            }
            scratch
                .policy_logits
                .push(dot(&context, &self.policy_output) + self.policy_bias[mv.0]);
        }
        softmax_moves(&legal, &scratch.policy_logits)
    }

    fn value(&self, scratch: &SparseScratch) -> f32 {
        if scratch.tokens.is_empty() {
            return 0.0;
        }
        let mut pooled = [0.0; TOKEN_WIDTH];
        for token in &scratch.tokens {
            for dimension in 0..TOKEN_WIDTH {
                pooled[dimension] += token[dimension] / scratch.tokens.len() as f32;
            }
        }
        let mut hidden = [0.0; TOKEN_WIDTH];
        project(&pooled, &self.value_hidden, &mut hidden);
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        let mut logits = [0.0; 3];
        for (output, logit) in logits.iter_mut().enumerate() {
            for dimension in 0..TOKEN_WIDTH {
                *logit += hidden[dimension] * self.value_output[dimension * 3 + output];
            }
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities = logits.map(|value| (value - maximum).exp());
        let sum = probabilities.iter().sum::<f32>();
        probabilities[0] / sum - probabilities[2] / sum
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
        for head in 0..HEADS {
            attend(
                &scratch.q[index],
                head,
                scale,
                scratch.moves.iter().enumerate().filter_map(|(other, &mv)| {
                    aligned(scratch.moves[index], mv)
                        .then_some((&scratch.k[other], &scratch.v[other]))
                }),
                &mut attention,
            );
        }
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

fn attend<'a>(
    query: &[f32; TOKEN_WIDTH],
    head: usize,
    scale: f32,
    keys: impl Iterator<Item = (&'a [f32; TOKEN_WIDTH], &'a [f32; TOKEN_WIDTH])>,
    output: &mut [f32; TOKEN_WIDTH],
) {
    let start = head * HEAD_WIDTH;
    let end = start + HEAD_WIDTH;
    let mut maximum = f32::NEG_INFINITY;
    let mut denominator = 0.0;
    for (key, value) in keys {
        let score = dot(&query[start..end], &key[start..end]) * scale;
        if score <= maximum {
            let weight = (score - maximum).exp();
            denominator += weight;
            for dimension in start..end {
                output[dimension] += weight * value[dimension];
            }
        } else {
            let rescale = (maximum - score).exp();
            denominator = denominator * rescale + 1.0;
            for dimension in start..end {
                output[dimension] = output[dimension] * rescale + value[dimension];
            }
            maximum = score;
        }
    }
    if denominator > 0.0 {
        for value in &mut output[start..end] {
            *value /= denominator;
        }
    }
}

fn aligned(left: Move, right: Move) -> bool {
    left.row() == right.row()
        || left.col() == right.col()
        || left.row().abs_diff(right.row()) == left.col().abs_diff(right.col())
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
    output.fill(0.0);
    for (input_index, &value) in input.iter().enumerate() {
        let row = &weights[input_index * output.len()..(input_index + 1) * output.len()];
        for output_index in 0..output.len() {
            output[output_index] += value * row[output_index];
        }
    }
}

fn rms_norm(input: &[f32; TOKEN_WIDTH], output: &mut [f32; TOKEN_WIDTH]) {
    let rms =
        (input.iter().map(|value| value * value).sum::<f32>() / TOKEN_WIDTH as f32 + 1e-6).sqrt();
    for dimension in 0..TOKEN_WIDTH {
        output[dimension] = input[dimension] / rms;
    }
}

fn softmax_moves(moves: &[Move], logits: &[f32]) -> Vec<(Move, f32)> {
    if moves.is_empty() {
        return Vec::new();
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f32>();
    moves
        .iter()
        .copied()
        .zip(logits.iter().map(|value| (value - maximum).exp() / sum))
        .collect()
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
        let model = SparseTransformerModel::random(7);
        let (policy, value) = model.evaluate(&board, &mut SparseScratch::default());
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
        let model = SparseTransformerModel::random(9);
        let mut scratch = SparseScratch::default();
        model.encode_stones(&board, &mut scratch);
        assert_eq!(scratch.tokens.len(), stones.len());
    }
}
