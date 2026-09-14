use crate::game::{ACTION_COUNT, BOARD_SIZE, Board, CELL_COUNT, Move, Player};
use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;
use std::{fs, io, path::Path};

pub const MOVE_COUNT_INPUT: usize = CELL_COUNT * 2;
pub const PASS_INPUT: usize = MOVE_COUNT_INPUT + 1;
pub const ROLE_INPUT_START: usize = PASS_INPUT + 1;
pub const ROLE_COUNT: usize = 2;
pub const GO_INPUT_START: usize = ROLE_INPUT_START + ROLE_COUNT;
pub const INPUT_SIZE: usize = GO_INPUT_START + crate::features::GO_FEATURE_SIZE;
/// 与 chineseai 默认主干宽度一致；围棋实验沿用盘面与局部方向特征。
pub const DEFAULT_HIDDEN_SIZE: usize = 128;
pub const VALUE_HEAD_SIZE: usize = 96;
pub const WDL_SIZE: usize = 3;
pub const SHORT_VALUE_HEADS: usize = 3;
pub const STONE_TYPES: usize = 2;
pub const AXIS_FEATURES: usize = STONE_TYPES * BOARD_SIZE;
pub const DIAGONAL_FEATURES: usize = STONE_TYPES * (BOARD_SIZE * 2 - 1);
pub const LOCAL_AXES: usize = 4;
pub const LOCAL_RADIUS: usize = 4;
pub const LOCAL_RAY_PATTERNS: usize = 4usize.pow(LOCAL_RADIUS as u32);
pub const LOCAL_AXIS_PATTERNS: usize = LOCAL_RAY_PATTERNS * (LOCAL_RAY_PATTERNS + 1) / 2;
/// 局部特征的主要容量放在 32,896 种精确方向棋形上，
/// 类似 chineseai 把大部分参数放在稀疏战术表，而不是盲目加宽稠密主干。
pub const LOCAL_AXIS_FEATURE_SIZE: usize = 16;
pub const LOCAL_CANDIDATE_SIZE: usize = LOCAL_AXIS_FEATURE_SIZE * 2;
pub const VALUE_LOCAL_SIZE: usize = LOCAL_CANDIDATE_SIZE * 2;
/// Policy 只保留一个很窄的上下文投影；大容量主干专供可增量更新的 value。
pub const POLICY_HEAD_SIZE: usize = 32;
/// 大容量表直接输出策略 logit，不展开为宽激活。
pub const POLICY_TACTICAL_SIZE: usize = 1 << 22;
pub const ROLE_ADAPTER_RANK: usize = 8;
pub const REGION_COUNT: usize = 9;
pub const REGION_FEATURE_SIZE: usize = 8;
pub const REGION_TOTAL_SIZE: usize = REGION_COUNT * REGION_FEATURE_SIZE;
const FORMAT_VERSION: f32 = 30.0;
const LOCAL_BOUNDARY: u8 = u8::MAX;
const LOCAL_NEIGHBORS: [u8; ACTION_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS] = build_local_neighbors();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyValueArch {
    pub hidden_size: usize,
}

impl PolicyValueArch {
    pub const fn default_const() -> Self {
        Self {
            hidden_size: DEFAULT_HIDDEN_SIZE,
        }
    }

    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        Self { hidden_size }
    }

    pub fn validate(self) -> io::Result<()> {
        if self.hidden_size == 0 {
            return Err(io::Error::other("hidden_size 必须大于 0"));
        }
        Ok(())
    }
}

impl Default for PolicyValueArch {
    fn default() -> Self {
        Self::default_const()
    }
}

const fn build_local_neighbors() -> [u8; ACTION_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS] {
    let mut table = [LOCAL_BOUNDARY; ACTION_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS];
    let directions = [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)];
    let mut cell = 0;
    while cell < CELL_COUNT {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        let mut axis = 0;
        while axis < LOCAL_AXES {
            let mut ray = 0;
            while ray < 2 {
                let sign = if ray == 0 { -1 } else { 1 };
                let mut distance = 1;
                while distance <= LOCAL_RADIUS {
                    let next_row = row + directions[axis].0 * distance as i32 * sign;
                    let next_col = col + directions[axis].1 * distance as i32 * sign;
                    let slot =
                        (((cell * LOCAL_AXES + axis) * 2 + ray) * LOCAL_RADIUS) + distance - 1;
                    if next_row >= 0
                        && next_col >= 0
                        && next_row < BOARD_SIZE as i32
                        && next_col < BOARD_SIZE as i32
                    {
                        table[slot] = (next_row as usize * BOARD_SIZE + next_col as usize) as u8;
                    }
                    distance += 1;
                }
                ray += 1;
            }
            axis += 1;
        }
        cell += 1;
    }
    table
}

#[derive(Clone)]
pub struct PolicyValueModel {
    pub hidden_size: usize,
    pub(crate) input_hidden: Vec<f32>,
    pub(crate) stone_hidden: Vec<f32>,
    pub(crate) rank_hidden: Vec<f32>,
    pub(crate) file_hidden: Vec<f32>,
    pub(crate) diagonal_hidden: Vec<f32>,
    pub(crate) anti_diagonal_hidden: Vec<f32>,
    pub(crate) hidden_bias: Vec<f32>,
    pub(crate) role_adapter_down: Vec<f32>,
    pub(crate) role_adapter_up: Vec<f32>,
    pub(crate) region_embedding: Vec<f32>,
    pub(crate) policy_global: Vec<f32>,
    pub(crate) policy_global_bias: Vec<f32>,
    pub(crate) policy_dynamic: Vec<f32>,
    pub(crate) policy_output: Vec<f32>,
    pub(crate) policy_bias: Vec<f32>,
    pub(crate) policy_tactical: Vec<f32>,
    pub(crate) local_axis_embedding: Vec<f32>,
    pub(crate) local_axis_scale: Vec<f32>,
    pub(crate) local_axis_bias: Vec<f32>,
    local_axis_features: Vec<f32>,
    pub(crate) policy_local: Vec<f32>,
    pub(crate) value_head_hidden: Vec<f32>,
    pub(crate) value_region_hidden: Vec<f32>,
    pub(crate) value_local_output: Vec<f32>,
    pub(crate) value_head_bias: Vec<f32>,
    pub(crate) value_head_hidden2: Vec<f32>,
    pub(crate) value_head_bias2: Vec<f32>,
    pub(crate) value_head_output: Vec<f32>,
    /// 训练专用的 4/12/32 ply 短期价值头，推理不读取。
    pub(crate) short_value_head_output: Vec<f32>,
    pub(crate) short_value_head_bias: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct EvalAccumulator {
    black: Vec<f32>,
    white: Vec<f32>,
    black_regions: Vec<f32>,
    white_regions: Vec<f32>,
    move_count: usize,
}

pub(crate) struct EvalScratch {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    local_candidate: Vec<f32>,
    local_value: Vec<f32>,
    value1: Vec<f32>,
    value2: Vec<f32>,
    policy_global: Vec<f32>,
    policy_dynamic: Vec<f32>,
    role_adapter: Vec<f32>,
    local_states: Vec<u8>,
}

impl EvalScratch {
    pub(crate) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: Vec::with_capacity(hidden_size),
            logits: Vec::with_capacity(ACTION_COUNT),
            local_candidate: vec![0.0; LOCAL_CANDIDATE_SIZE],
            local_value: vec![0.0; VALUE_LOCAL_SIZE],
            value1: Vec::with_capacity(VALUE_HEAD_SIZE),
            value2: Vec::with_capacity(VALUE_HEAD_SIZE),
            policy_global: Vec::with_capacity(POLICY_HEAD_SIZE),
            policy_dynamic: vec![0.0; LOCAL_CANDIDATE_SIZE],
            role_adapter: vec![0.0; ROLE_ADAPTER_RANK],
            local_states: vec![0; CELL_COUNT],
        }
    }
}

impl Default for PolicyValueModel {
    fn default() -> Self {
        Self::random(DEFAULT_HIDDEN_SIZE, 20260730)
    }
}

impl PolicyValueModel {
    pub fn arch(&self) -> PolicyValueArch {
        PolicyValueArch::with_hidden_size(self.hidden_size)
    }

    pub fn random_with_arch(arch: PolicyValueArch, seed: u64) -> Self {
        arch.validate().expect("9×9围棋网络架构必须合法");
        Self::random(arch.hidden_size, seed)
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        let hidden_size = hidden_size.max(1);
        let mut rng = SplitMix64(seed);
        let input_scale = (2.0 / INPUT_SIZE as f32).sqrt();
        let head_scale = (2.0 / hidden_size as f32).sqrt() * 0.25;
        let input_hidden = (0..INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(input_scale))
            .collect();
        let policy_global = (0..POLICY_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight(head_scale))
            .collect();
        let policy_bias = vec![0.0; ACTION_COUNT];
        let mut model = Self {
            hidden_size,
            input_hidden,
            stone_hidden: vec![0.0; STONE_TYPES * hidden_size],
            rank_hidden: vec![0.0; AXIS_FEATURES * hidden_size],
            file_hidden: vec![0.0; AXIS_FEATURES * hidden_size],
            diagonal_hidden: vec![0.0; DIAGONAL_FEATURES * hidden_size],
            anti_diagonal_hidden: vec![0.0; DIAGONAL_FEATURES * hidden_size],
            hidden_bias: vec![0.0; hidden_size],
            role_adapter_down: (0..ROLE_COUNT * ROLE_ADAPTER_RANK * hidden_size)
                .map(|_| rng.weight((2.0 / hidden_size as f32).sqrt()))
                .collect(),
            // Zero-up initialization preserves the original trunk at migration/startup.
            role_adapter_up: vec![0.0; ROLE_COUNT * hidden_size * ROLE_ADAPTER_RANK],
            region_embedding: (0..STONE_TYPES * REGION_COUNT * REGION_FEATURE_SIZE)
                .map(|_| rng.weight((2.0 / REGION_FEATURE_SIZE as f32).sqrt() * 0.25))
                .collect(),
            policy_global,
            policy_global_bias: vec![0.0; POLICY_HEAD_SIZE],
            policy_dynamic: vec![0.0; LOCAL_CANDIDATE_SIZE * POLICY_HEAD_SIZE],
            policy_output: (0..ACTION_COUNT * POLICY_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / POLICY_HEAD_SIZE as f32).sqrt() * 0.25))
                .collect(),
            policy_bias,
            policy_tactical: vec![0.0; POLICY_TACTICAL_SIZE],
            local_axis_embedding: (0..LOCAL_AXIS_PATTERNS * LOCAL_AXIS_FEATURE_SIZE)
                .map(|_| rng.weight((2.0 / LOCAL_AXIS_FEATURE_SIZE as f32).sqrt() * 0.25))
                .collect(),
            local_axis_scale: vec![1.0; 2 * LOCAL_AXIS_FEATURE_SIZE],
            local_axis_bias: vec![0.0; 2 * LOCAL_AXIS_FEATURE_SIZE],
            local_axis_features: Vec::new(),
            policy_local: vec![0.0; LOCAL_CANDIDATE_SIZE],
            value_head_hidden: (0..hidden_size * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / hidden_size as f32).sqrt() * 0.5))
                .collect(),
            value_region_hidden: vec![0.0; REGION_TOTAL_SIZE * VALUE_HEAD_SIZE],
            value_local_output: vec![0.0; WDL_SIZE * VALUE_LOCAL_SIZE],
            value_head_bias: vec![0.0; VALUE_HEAD_SIZE],
            value_head_hidden2: (0..VALUE_HEAD_SIZE * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / VALUE_HEAD_SIZE as f32).sqrt() * 0.5))
                .collect(),
            value_head_bias2: vec![0.0; VALUE_HEAD_SIZE],
            value_head_output: vec![0.0; VALUE_HEAD_SIZE * WDL_SIZE],
            short_value_head_output: vec![0.0; SHORT_VALUE_HEADS * WDL_SIZE * VALUE_HEAD_SIZE],
            short_value_head_bias: vec![0.0; SHORT_VALUE_HEADS * WDL_SIZE],
        };
        model.refresh_local_axis_features();
        model
    }

    pub fn evaluate(&self, board: &Board) -> (Vec<(Move, f32)>, f32) {
        let accumulator = self.accumulator(board);
        self.evaluate_accumulator(board, &accumulator)
    }

    pub(crate) fn evaluate_accumulator(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
    ) -> (Vec<(Move, f32)>, f32) {
        self.evaluate_accumulator_with_temperature(board, accumulator, 1.0)
    }

    pub(crate) fn evaluate_accumulator_with_temperature(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        policy_temperature: f32,
    ) -> (Vec<(Move, f32)>, f32) {
        let mut scratch = EvalScratch::new(self.hidden_size);
        self.evaluate_accumulator_with_scratch(board, accumulator, policy_temperature, &mut scratch)
    }

    pub(crate) fn evaluate_accumulator_with_scratch(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        policy_temperature: f32,
        scratch: &mut EvalScratch,
    ) -> (Vec<(Move, f32)>, f32) {
        debug_assert_eq!(accumulator.move_count, board.move_count());
        let preactivation = match board.to_move() {
            Player::Black => &accumulator.black,
            Player::White => &accumulator.white,
        };
        let regions = match board.to_move() {
            Player::Black => &accumulator.black_regions,
            Player::White => &accumulator.white_regions,
        };
        self.evaluate_preactivation_with_scratch(
            board,
            preactivation,
            regions,
            board.to_move(),
            policy_temperature,
            scratch,
        )
    }

    fn evaluate_preactivation_with_scratch(
        &self,
        board: &Board,
        preactivation: &[f32],
        regions: &[f32],
        role: Player,
        policy_temperature: f32,
        scratch: &mut EvalScratch,
    ) -> (Vec<(Move, f32)>, f32) {
        crate::scope_profile!("model.evaluate_incremental");
        {
            crate::scope_profile!("model.activate_norm");
            self.activate_hidden_into(preactivation, &mut scratch.hidden);
            self.apply_role_adapter(role, &mut scratch.hidden, &mut scratch.role_adapter);
        }
        let moves = board.search_candidates();
        if moves.is_empty() {
            return (Vec::new(), 0.0);
        }
        {
            crate::scope_profile!("model.policy_logits");
            scratch.policy_global.clear();
            scratch
                .policy_global
                .extend_from_slice(&self.policy_global_bias);
            for output in 0..POLICY_HEAD_SIZE {
                let start = output * self.hidden_size;
                scratch.policy_global[output] += dot(
                    &scratch.hidden,
                    &self.policy_global[start..start + self.hidden_size],
                );
            }
            for value in &mut scratch.policy_global {
                *value = value.max(0.0);
            }
            scratch.policy_dynamic.copy_from_slice(&self.policy_local);
            for (local, weight) in scratch.policy_dynamic.iter_mut().enumerate() {
                let start = local * POLICY_HEAD_SIZE;
                *weight += dot(
                    &scratch.policy_global,
                    &self.policy_dynamic[start..start + POLICY_HEAD_SIZE],
                );
            }
            scratch.logits.clear();
            scratch.local_value.fill(0.0);
            scratch.local_value[LOCAL_CANDIDATE_SIZE..].fill(f32::NEG_INFINITY);
            let us = board.to_move().stone();
            for (state, &stone) in scratch.local_states.iter_mut().zip(board.cells()) {
                *state = if stone == us {
                    1
                } else if stone == -us {
                    2
                } else {
                    0
                };
            }
            for &mv in &moves {
                let tactical = self.local_candidate_into(
                    &scratch.local_states,
                    mv,
                    &mut scratch.local_candidate,
                );
                let tactical_logit = tactical
                    .into_iter()
                    .map(|index| self.policy_tactical[index])
                    .sum::<f32>();
                scratch.logits.push(
                    self.policy_logit(
                        &scratch.policy_global,
                        &scratch.policy_dynamic,
                        &scratch.local_candidate,
                        mv,
                    ) + tactical_logit,
                );
                for (i, &value) in scratch.local_candidate.iter().enumerate() {
                    scratch.local_value[i] += value;
                    scratch.local_value[LOCAL_CANDIDATE_SIZE + i] =
                        scratch.local_value[LOCAL_CANDIDATE_SIZE + i].max(value);
                }
            }
        }
        let max = scratch
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let inverse_temperature = policy_temperature.max(1e-3).recip();
        let sum: f32 = scratch
            .logits
            .iter()
            .map(|x| ((x - max) * inverse_temperature).exp())
            .sum();
        let policy = moves
            .iter()
            .copied()
            .zip(
                scratch
                    .logits
                    .iter()
                    .copied()
                    .map(|x| ((x - max) * inverse_temperature).exp() / sum),
            )
            .collect();
        crate::scope_profile!("model.value_head");
        let inverse_moves = 1.0 / moves.len() as f32;
        for value in &mut scratch.local_value[..LOCAL_CANDIDATE_SIZE] {
            *value *= inverse_moves;
        }
        scratch.value1.clear();
        scratch.value1.extend_from_slice(&self.value_head_bias);
        for (output, value) in scratch.value1.iter_mut().enumerate() {
            let start = output * self.hidden_size;
            *value += dot(
                &scratch.hidden,
                &self.value_head_hidden[start..start + self.hidden_size],
            ) + dot(
                regions,
                &self.value_region_hidden
                    [output * REGION_TOTAL_SIZE..(output + 1) * REGION_TOTAL_SIZE],
            );
        }
        for x in &mut scratch.value1 {
            *x = x.max(0.0);
        }
        scratch.value2.clear();
        scratch.value2.extend_from_slice(&self.value_head_bias2);
        for (output, value) in scratch.value2.iter_mut().enumerate() {
            let start = output * VALUE_HEAD_SIZE;
            *value += dot(
                &scratch.value1,
                &self.value_head_hidden2[start..start + VALUE_HEAD_SIZE],
            );
        }
        for x in &mut scratch.value2 {
            *x = x.max(0.0);
        }
        let mut wdl = [0.0_f32; WDL_SIZE];
        for (output, logit) in wdl.iter_mut().enumerate() {
            let start = output * VALUE_HEAD_SIZE;
            *logit = dot(
                &scratch.value2,
                &self.value_head_output[start..start + VALUE_HEAD_SIZE],
            ) + dot(
                &scratch.local_value,
                &self.value_local_output
                    [output * VALUE_LOCAL_SIZE..(output + 1) * VALUE_LOCAL_SIZE],
            );
        }
        let wdl_max = wdl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let wdl_sum: f32 = wdl.iter().map(|x| (x - wdl_max).exp()).sum();
        let wdl = wdl.map(|x| (x - wdl_max).exp() / wdl_sum);
        let value = wdl[0] - wdl[2];
        (policy, value)
    }

    pub(crate) fn accumulator(&self, board: &Board) -> EvalAccumulator {
        crate::scope_profile!("model.accumulator_root");
        let mut accumulator = EvalAccumulator {
            black: self.hidden_bias.clone(),
            white: self.hidden_bias.clone(),
            black_regions: vec![0.0; REGION_TOTAL_SIZE],
            white_regions: vec![0.0; REGION_TOTAL_SIZE],
            move_count: 0,
        };
        self.add_role_to_slices(&mut accumulator.black, &mut accumulator.white);
        for (sq, &stone) in board.cells().iter().enumerate() {
            if stone == 0 {
                continue;
            }
            self.add_stone(
                &mut accumulator,
                Move(sq),
                if stone == Player::Black.stone() {
                    Player::Black
                } else {
                    Player::White
                },
            );
        }
        self.set_move_count(&mut accumulator, board.move_count());
        for h in 0..self.hidden_size {
            let change = self.input_hidden[PASS_INPUT * self.hidden_size + h]
                * board.consecutive_passes() as f32;
            accumulator.black[h] += change;
            accumulator.white[h] += change;
        }
        for (perspective, hidden) in [
            (Player::Black, &mut accumulator.black),
            (Player::White, &mut accumulator.white),
        ] {
            for (feature, value) in crate::features::encode(board, perspective)
                .into_iter()
                .enumerate()
            {
                if value == 0.0 {
                    continue;
                }
                let start = (GO_INPUT_START + feature) * self.hidden_size;
                for h in 0..self.hidden_size {
                    hidden[h] += self.input_hidden[start + h] * value;
                }
            }
        }
        accumulator
    }

    /// MCTS 使用连续 arena 保存累加器，避免每个节点为两个 Vec 单独分配内存。
    pub(crate) fn accumulator_into_arena(&self, board: &Board, arena: &mut Vec<f32>) -> usize {
        let accumulator = self.accumulator(board);
        let offset = arena.len();
        arena.extend_from_slice(&accumulator.black);
        arena.extend_from_slice(&accumulator.white);
        arena.extend_from_slice(&accumulator.black_regions);
        arena.extend_from_slice(&accumulator.white_regions);
        offset
    }

    pub(crate) fn evaluate_arena_with_scratch(
        &self,
        board: &Board,
        arena: &[f32],
        offset: usize,
        policy_temperature: f32,
        scratch: &mut EvalScratch,
    ) -> (Vec<(Move, f32)>, f32) {
        let width = self.accumulator_width();
        let side_offset = match board.to_move() {
            Player::Black => offset,
            Player::White => offset + self.hidden_size,
        };
        let region_offset = match board.to_move() {
            Player::Black => offset + self.hidden_size * 2,
            Player::White => offset + self.hidden_size * 2 + REGION_TOTAL_SIZE,
        };
        debug_assert!(offset + width <= arena.len());
        self.evaluate_preactivation_with_scratch(
            board,
            &arena[side_offset..side_offset + self.hidden_size],
            &arena[region_offset..region_offset + REGION_TOTAL_SIZE],
            board.to_move(),
            policy_temperature,
            scratch,
        )
    }

    fn add_stone(&self, accumulator: &mut EvalAccumulator, mv: Move, player: Player) {
        self.add_stone_to_slices(&mut accumulator.black, &mut accumulator.white, mv, player);
        self.add_region_to_slices(
            &mut accumulator.black_regions,
            &mut accumulator.white_regions,
            mv,
            player,
        );
    }

    pub(crate) fn accumulator_width(&self) -> usize {
        2 * (self.hidden_size + REGION_TOTAL_SIZE)
    }

    fn add_region_to_slices(&self, black: &mut [f32], white: &mut [f32], mv: Move, player: Player) {
        let region = (mv.row() / (BOARD_SIZE / 3)) * 3 + mv.col() / (BOARD_SIZE / 3);
        for (perspective, values) in [(Player::Black, black), (Player::White, white)] {
            let side = usize::from(player != perspective);
            let source = (region * STONE_TYPES + side) * REGION_FEATURE_SIZE;
            let target = region * REGION_FEATURE_SIZE;
            for i in 0..REGION_FEATURE_SIZE {
                values[target + i] += self.region_embedding[source + i];
            }
        }
    }

    fn add_role_to_slices(&self, black: &mut [f32], white: &mut [f32]) {
        for (role, hidden) in [(Player::Black, black), (Player::White, white)] {
            let offset = (ROLE_INPUT_START + role_index(role)) * self.hidden_size;
            for (h, value) in hidden.iter_mut().enumerate() {
                *value += self.input_hidden[offset + h];
            }
        }
    }

    fn add_stone_to_slices(&self, black: &mut [f32], white: &mut [f32], mv: Move, player: Player) {
        for (perspective, hidden) in [(Player::Black, black), (Player::White, white)] {
            let side = usize::from(player != perspective);
            let exact = (side * CELL_COUNT + mv.0) * self.hidden_size;
            let rank = (side * BOARD_SIZE + mv.row()) * self.hidden_size;
            let file = (side * BOARD_SIZE + mv.col()) * self.hidden_size;
            let diagonal = (side * (BOARD_SIZE * 2 - 1) + mv.row() + BOARD_SIZE - 1 - mv.col())
                * self.hidden_size;
            let anti_diagonal =
                (side * (BOARD_SIZE * 2 - 1) + mv.row() + mv.col()) * self.hidden_size;
            let stone = side * self.hidden_size;
            for (h, value) in hidden.iter_mut().enumerate() {
                *value += self.input_hidden[exact + h]
                    + self.stone_hidden[stone + h]
                    + self.rank_hidden[rank + h]
                    + self.file_hidden[file + h]
                    + self.diagonal_hidden[diagonal + h]
                    + self.anti_diagonal_hidden[anti_diagonal + h];
            }
        }
    }

    fn set_move_count(&self, accumulator: &mut EvalAccumulator, move_count: usize) {
        let delta = (move_count as f32 - accumulator.move_count as f32) / CELL_COUNT as f32;
        let rule_offset = MOVE_COUNT_INPUT * self.hidden_size;
        for h in 0..self.hidden_size {
            let change = self.input_hidden[rule_offset + h] * delta;
            accumulator.black[h] += change;
            accumulator.white[h] += change;
        }
        accumulator.move_count = move_count;
    }

    fn activate_hidden_into(&self, preactivation: &[f32], hidden: &mut Vec<f32>) {
        hidden.clear();
        hidden.extend_from_slice(preactivation);
        for x in hidden.iter_mut() {
            *x = x.max(0.0);
        }
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len().max(1) as f32 + 1.0e-6)
            .sqrt();
        for x in hidden.iter_mut() {
            *x /= rms;
        }
    }

    fn apply_role_adapter(&self, role: Player, hidden: &mut [f32], adapter: &mut [f32]) {
        let role = role_index(role);
        let down = role * ROLE_ADAPTER_RANK * self.hidden_size;
        for (rank, value) in adapter.iter_mut().enumerate() {
            let start = down + rank * self.hidden_size;
            *value = dot(
                hidden,
                &self.role_adapter_down[start..start + self.hidden_size],
            )
            .max(0.0);
        }
        let up = role * self.hidden_size * ROLE_ADAPTER_RANK;
        for (output, value) in hidden.iter_mut().enumerate() {
            let start = up + output * ROLE_ADAPTER_RANK;
            *value += dot(
                adapter,
                &self.role_adapter_up[start..start + ROLE_ADAPTER_RANK],
            );
        }
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len().max(1) as f32 + 1.0e-6)
            .sqrt();
        for value in hidden {
            *value /= rms;
        }
    }

    fn local_candidate_into(
        &self,
        states: &[u8],
        mv: Move,
        output: &mut [f32],
    ) -> [usize; LOCAL_AXES] {
        output.fill(0.0);
        let (mean, max) = output.split_at_mut(LOCAL_AXIS_FEATURE_SIZE);
        max.fill(f32::NEG_INFINITY);
        let mut feature_starts = [0; LOCAL_AXES];
        let mut tactical = [0; LOCAL_AXES];
        for axis in 0..LOCAL_AXES {
            let (first_code, second_code) = local_ray_codes_from_states(states, mv, axis);
            let pattern = second_code * (second_code + 1) / 2 + first_code;
            tactical[axis] = policy_tactical_index(mv, axis, pattern);
            feature_starts[axis] =
                ((axis / 2) * LOCAL_AXIS_PATTERNS + pattern) * LOCAL_AXIS_FEATURE_SIZE;
        }
        aggregate_axis_features(&self.local_axis_features, feature_starts, mean, max);
        tactical
    }

    pub(crate) fn refresh_local_axis_features(&mut self) {
        self.local_axis_features
            .resize(2 * LOCAL_AXIS_PATTERNS * LOCAL_AXIS_FEATURE_SIZE, 0.0);
        for kind in 0..2 {
            let transform = kind * LOCAL_AXIS_FEATURE_SIZE;
            for pattern in 0..LOCAL_AXIS_PATTERNS {
                let source = pattern * LOCAL_AXIS_FEATURE_SIZE;
                let target = (kind * LOCAL_AXIS_PATTERNS + pattern) * LOCAL_AXIS_FEATURE_SIZE;
                for i in 0..LOCAL_AXIS_FEATURE_SIZE {
                    self.local_axis_features[target + i] = self.local_axis_embedding[source + i]
                        * self.local_axis_scale[transform + i]
                        + self.local_axis_bias[transform + i];
                }
            }
        }
    }

    fn policy_logit(&self, global: &[f32], dynamic: &[f32], local: &[f32], mv: Move) -> f32 {
        let weights = &self.policy_output[mv.0 * POLICY_HEAD_SIZE..(mv.0 + 1) * POLICY_HEAD_SIZE];
        dot(global, weights) + dot(local, dynamic) + self.policy_bias[mv.0]
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let vars = VarMap::new();
        insert(&vars, "format_version", &[FORMAT_VERSION], (1,))?;
        insert(
            &vars,
            "input_hidden",
            &self.input_hidden,
            (INPUT_SIZE, self.hidden_size),
        )?;
        insert(
            &vars,
            "stone_hidden",
            &self.stone_hidden,
            (STONE_TYPES, self.hidden_size),
        )?;
        insert(
            &vars,
            "rank_hidden",
            &self.rank_hidden,
            (AXIS_FEATURES, self.hidden_size),
        )?;
        insert(
            &vars,
            "file_hidden",
            &self.file_hidden,
            (AXIS_FEATURES, self.hidden_size),
        )?;
        insert(
            &vars,
            "diagonal_hidden",
            &self.diagonal_hidden,
            (DIAGONAL_FEATURES, self.hidden_size),
        )?;
        insert(
            &vars,
            "anti_diagonal_hidden",
            &self.anti_diagonal_hidden,
            (DIAGONAL_FEATURES, self.hidden_size),
        )?;
        insert(&vars, "hidden_bias", &self.hidden_bias, (self.hidden_size,))?;
        insert(
            &vars,
            "role_adapter_down",
            &self.role_adapter_down,
            (ROLE_COUNT, ROLE_ADAPTER_RANK, self.hidden_size),
        )?;
        insert(
            &vars,
            "role_adapter_up",
            &self.role_adapter_up,
            (ROLE_COUNT, self.hidden_size, ROLE_ADAPTER_RANK),
        )?;
        insert(
            &vars,
            "region_embedding",
            &self.region_embedding,
            (REGION_COUNT, STONE_TYPES, REGION_FEATURE_SIZE),
        )?;
        insert(
            &vars,
            "policy_global",
            &self.policy_global,
            (POLICY_HEAD_SIZE, self.hidden_size),
        )?;
        insert(
            &vars,
            "policy_global_bias",
            &self.policy_global_bias,
            (POLICY_HEAD_SIZE,),
        )?;
        insert(
            &vars,
            "policy_dynamic",
            &self.policy_dynamic,
            (LOCAL_CANDIDATE_SIZE, POLICY_HEAD_SIZE),
        )?;
        insert(
            &vars,
            "policy_output",
            &self.policy_output,
            (ACTION_COUNT, POLICY_HEAD_SIZE),
        )?;
        insert(&vars, "policy_bias", &self.policy_bias, (ACTION_COUNT,))?;
        insert(
            &vars,
            "policy_tactical",
            &self.policy_tactical,
            (POLICY_TACTICAL_SIZE,),
        )?;
        insert(
            &vars,
            "local_axis_embedding",
            &self.local_axis_embedding,
            (LOCAL_AXIS_PATTERNS, LOCAL_AXIS_FEATURE_SIZE),
        )?;
        insert(
            &vars,
            "local_axis_scale",
            &self.local_axis_scale,
            (2, LOCAL_AXIS_FEATURE_SIZE),
        )?;
        insert(
            &vars,
            "local_axis_bias",
            &self.local_axis_bias,
            (2, LOCAL_AXIS_FEATURE_SIZE),
        )?;
        insert(
            &vars,
            "policy_local",
            &self.policy_local,
            (LOCAL_CANDIDATE_SIZE,),
        )?;
        insert(
            &vars,
            "value_head_hidden",
            &self.value_head_hidden,
            (VALUE_HEAD_SIZE, self.hidden_size),
        )?;
        insert(
            &vars,
            "value_region_hidden",
            &self.value_region_hidden,
            (VALUE_HEAD_SIZE, REGION_TOTAL_SIZE),
        )?;
        insert(
            &vars,
            "value_head_bias",
            &self.value_head_bias,
            (VALUE_HEAD_SIZE,),
        )?;
        insert(
            &vars,
            "value_local_output",
            &self.value_local_output,
            (WDL_SIZE, VALUE_LOCAL_SIZE),
        )?;
        insert(
            &vars,
            "value_head_hidden2",
            &self.value_head_hidden2,
            (VALUE_HEAD_SIZE, VALUE_HEAD_SIZE),
        )?;
        insert(
            &vars,
            "value_head_bias2",
            &self.value_head_bias2,
            (VALUE_HEAD_SIZE,),
        )?;
        insert(
            &vars,
            "value_head_output",
            &self.value_head_output,
            (WDL_SIZE, VALUE_HEAD_SIZE),
        )?;
        insert(
            &vars,
            "short_value_head_output",
            &self.short_value_head_output,
            (SHORT_VALUE_HEADS, WDL_SIZE, VALUE_HEAD_SIZE),
        )?;
        insert(
            &vars,
            "short_value_head_bias",
            &self.short_value_head_bias,
            (SHORT_VALUE_HEADS, WDL_SIZE),
        )?;
        vars.save(path).map_err(candle_error)
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path.as_ref()).map_err(candle_error)?
        };
        let version = load(&tensors, "format_version")?;
        let version = version.first().copied().unwrap_or_default();
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "不支持的9×9围棋模型版本",
            ));
        }
        let hidden_bias = load(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len();
        if hidden_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "9×9围棋模型隐藏层不能为空",
            ));
        }
        let model = Self {
            hidden_size,
            input_hidden: load(&tensors, "input_hidden")?,
            stone_hidden: load(&tensors, "stone_hidden")?,
            rank_hidden: load(&tensors, "rank_hidden")?,
            file_hidden: load(&tensors, "file_hidden")?,
            diagonal_hidden: load(&tensors, "diagonal_hidden")?,
            anti_diagonal_hidden: load(&tensors, "anti_diagonal_hidden")?,
            hidden_bias,
            role_adapter_down: load(&tensors, "role_adapter_down")?,
            role_adapter_up: load(&tensors, "role_adapter_up")?,
            region_embedding: load(&tensors, "region_embedding")?,
            policy_global: load(&tensors, "policy_global")?,
            policy_global_bias: load(&tensors, "policy_global_bias")?,
            policy_dynamic: load(&tensors, "policy_dynamic")?,
            policy_output: load(&tensors, "policy_output")?,
            policy_bias: load(&tensors, "policy_bias")?,
            policy_tactical: load(&tensors, "policy_tactical")?,
            local_axis_embedding: load(&tensors, "local_axis_embedding")?,
            local_axis_scale: load(&tensors, "local_axis_scale")?,
            local_axis_bias: load(&tensors, "local_axis_bias")?,
            local_axis_features: Vec::new(),
            policy_local: load(&tensors, "policy_local")?,
            value_head_hidden: load(&tensors, "value_head_hidden")?,
            value_region_hidden: load(&tensors, "value_region_hidden")?,
            value_local_output: load(&tensors, "value_local_output")?,
            value_head_bias: load(&tensors, "value_head_bias")?,
            value_head_hidden2: load(&tensors, "value_head_hidden2")?,
            value_head_bias2: load(&tensors, "value_head_bias2")?,
            value_head_output: load(&tensors, "value_head_output")?,
            short_value_head_output: load(&tensors, "short_value_head_output")?,
            short_value_head_bias: load(&tensors, "short_value_head_bias")?,
        };
        if model.input_hidden.len() != INPUT_SIZE * hidden_size
            || model.stone_hidden.len() != STONE_TYPES * hidden_size
            || model.rank_hidden.len() != AXIS_FEATURES * hidden_size
            || model.file_hidden.len() != AXIS_FEATURES * hidden_size
            || model.diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.anti_diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.role_adapter_down.len() != ROLE_COUNT * ROLE_ADAPTER_RANK * hidden_size
            || model.role_adapter_up.len() != ROLE_COUNT * hidden_size * ROLE_ADAPTER_RANK
            || model.region_embedding.len() != STONE_TYPES * REGION_COUNT * REGION_FEATURE_SIZE
            || model.policy_global.len() != POLICY_HEAD_SIZE * hidden_size
            || model.policy_global_bias.len() != POLICY_HEAD_SIZE
            || model.policy_dynamic.len() != LOCAL_CANDIDATE_SIZE * POLICY_HEAD_SIZE
            || model.policy_output.len() != ACTION_COUNT * POLICY_HEAD_SIZE
            || model.policy_bias.len() != ACTION_COUNT
            || model.policy_tactical.len() != POLICY_TACTICAL_SIZE
            || model.local_axis_embedding.len() != LOCAL_AXIS_PATTERNS * LOCAL_AXIS_FEATURE_SIZE
            || model.local_axis_scale.len() != 2 * LOCAL_AXIS_FEATURE_SIZE
            || model.local_axis_bias.len() != 2 * LOCAL_AXIS_FEATURE_SIZE
            || model.policy_local.len() != LOCAL_CANDIDATE_SIZE
            || model.value_head_hidden.len() != hidden_size * VALUE_HEAD_SIZE
            || model.value_region_hidden.len() != VALUE_HEAD_SIZE * REGION_TOTAL_SIZE
            || model.value_local_output.len() != WDL_SIZE * VALUE_LOCAL_SIZE
            || model.value_head_bias.len() != VALUE_HEAD_SIZE
            || model.value_head_hidden2.len() != VALUE_HEAD_SIZE * VALUE_HEAD_SIZE
            || model.value_head_bias2.len() != VALUE_HEAD_SIZE
            || model.value_head_output.len() != VALUE_HEAD_SIZE * WDL_SIZE
            || model.short_value_head_output.len() != SHORT_VALUE_HEADS * WDL_SIZE * VALUE_HEAD_SIZE
            || model.short_value_head_bias.len() != SHORT_VALUE_HEADS * WDL_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "9×9围棋模型张量尺寸错误",
            ));
        }
        let mut model = model;
        model.refresh_local_axis_features();
        Ok(model)
    }
}

fn role_index(player: Player) -> usize {
    match player {
        Player::Black => 0,
        Player::White => 1,
    }
}

pub(crate) fn local_ray_codes(board: &Board, mv: Move, dr: i32, dc: i32) -> (usize, usize) {
    let us = board.to_move().stone();
    let axis = match (dr, dc) {
        (1, 0) | (-1, 0) => 0,
        (0, 1) | (0, -1) => 1,
        (1, 1) | (-1, -1) => 2,
        (1, -1) | (-1, 1) => 3,
        _ => unreachable!("invalid local-pattern axis"),
    };
    let encode = |ray: usize| {
        let mut code = 0;
        let mut place = 1;
        let start = ((mv.0 * LOCAL_AXES + axis) * 2 + ray) * LOCAL_RADIUS;
        for &cell in &LOCAL_NEIGHBORS[start..start + LOCAL_RADIUS] {
            let state = if cell == LOCAL_BOUNDARY {
                3
            } else {
                match board.cells()[cell as usize] {
                    stone if stone == us => 1,
                    stone if stone == -us => 2,
                    _ => 0,
                }
            };
            code += state * place;
            place *= 4;
        }
        code as usize
    };
    let rays = (encode(0), encode(1));
    if rays.0 <= rays.1 {
        rays
    } else {
        (rays.1, rays.0)
    }
}

fn local_ray_codes_from_states(states: &[u8], mv: Move, axis: usize) -> (usize, usize) {
    let encode = |ray: usize| {
        let start = ((mv.0 * LOCAL_AXES + axis) * 2 + ray) * LOCAL_RADIUS;
        let mut code = 0_usize;
        let mut place = 1_usize;
        for &cell in &LOCAL_NEIGHBORS[start..start + LOCAL_RADIUS] {
            let state = if cell == LOCAL_BOUNDARY {
                3
            } else {
                states[cell as usize] as usize
            };
            code += state * place;
            place *= 4;
        }
        code
    };
    let rays = (encode(0), encode(1));
    if rays.0 <= rays.1 {
        rays
    } else {
        (rays.1, rays.0)
    }
}

pub(crate) fn policy_tactical_index(mv: Move, axis: usize, pattern: usize) -> usize {
    let mut value = pattern as u64
        ^ (mv.0 as u64).wrapping_mul(0x9E37_79B9)
        ^ (axis as u64).wrapping_mul(0x85EB_CA6B);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    (value as usize) & (POLICY_TACTICAL_SIZE - 1)
}

#[inline(always)]
fn aggregate_axis_features(
    features: &[f32],
    starts: [usize; LOCAL_AXES],
    mean: &mut [f32],
    max: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            aggregate_axis_features_avx2(features, starts, mean, max);
        }
        return;
    }
    for start in starts {
        let axis = &features[start..start + LOCAL_AXIS_FEATURE_SIZE];
        for i in 0..LOCAL_AXIS_FEATURE_SIZE {
            mean[i] += axis[i] / LOCAL_AXES as f32;
            max[i] = max[i].max(axis[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn aggregate_axis_features_avx2(
    features: &[f32],
    starts: [usize; LOCAL_AXES],
    mean: &mut [f32],
    max: &mut [f32],
) {
    use std::arch::x86_64::*;
    let quarter = _mm256_set1_ps(1.0 / LOCAL_AXES as f32);
    for offset in (0..LOCAL_AXIS_FEATURE_SIZE).step_by(8) {
        let mut sum = _mm256_setzero_ps();
        let mut maximum = _mm256_set1_ps(f32::NEG_INFINITY);
        for start in starts {
            let value = unsafe { _mm256_loadu_ps(features.as_ptr().add(start + offset)) };
            sum = _mm256_add_ps(sum, _mm256_mul_ps(value, quarter));
            maximum = _mm256_max_ps(maximum, value);
        }
        unsafe {
            _mm256_storeu_ps(mean.as_mut_ptr().add(offset), sum);
            _mm256_storeu_ps(max.as_mut_ptr().add(offset), maximum);
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if a.len() >= 16 {
        return unsafe { dot_neon(a, b) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if a.len() >= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_avx2_fma(a, b) };
        }
        if a.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    let mut sums = [0.0_f32; 4];
    let chunks = a.len() / 4;
    for index in 0..chunks {
        let offset = index * 4;
        sums[0] += a[offset] * b[offset];
        sums[1] += a[offset + 1] * b[offset + 1];
        sums[2] += a[offset + 2] * b[offset + 2];
        sums[3] += a[offset + 3] * b[offset + 3];
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for index in chunks * 4..a.len() {
        sum += a[index] * b[index];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let chunks = left.len() / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    for chunk in 0..chunks {
        let index = chunk * 16;
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(left.as_ptr().add(index)),
                vld1q_f32(right.as_ptr().add(index)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(left.as_ptr().add(index + 4)),
                vld1q_f32(right.as_ptr().add(index + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(left.as_ptr().add(index + 8)),
                vld1q_f32(right.as_ptr().add(index + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(left.as_ptr().add(index + 12)),
                vld1q_f32(right.as_ptr().add(index + 12)),
            );
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for index in chunks * 16..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 32;
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index)),
                _mm256_loadu_ps(right.as_ptr().add(index)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 8)),
                _mm256_loadu_ps(right.as_ptr().add(index + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 16)),
                _mm256_loadu_ps(right.as_ptr().add(index + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 24)),
                _mm256_loadu_ps(right.as_ptr().add(index + 24)),
                acc3,
            );
        }
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    for index in chunks * 32..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.as_ptr().add(index)),
                    _mm256_loadu_ps(right.as_ptr().add(index)),
                ),
            );
        }
    }
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    for index in chunks * 8..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.as_ptr().add(index)),
                    _mm256_loadu_ps(right.as_ptr().add(index)),
                ),
            );
        }
    }
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    for index in chunks * 8..left.len() {
        sum += left[index] * right[index];
    }
    sum
}
fn candle_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
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
        .and_then(|x| x.to_vec1::<f32>())
        .map_err(candle_error)
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn weight(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_axis_aggregation_matches_scalar_formula() {
        let features = (0..LOCAL_AXES * LOCAL_AXIS_FEATURE_SIZE)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect::<Vec<_>>();
        let starts = [
            0,
            LOCAL_AXIS_FEATURE_SIZE,
            LOCAL_AXIS_FEATURE_SIZE * 2,
            LOCAL_AXIS_FEATURE_SIZE * 3,
        ];
        let mut mean = vec![0.0; LOCAL_AXIS_FEATURE_SIZE];
        let mut max = vec![f32::NEG_INFINITY; LOCAL_AXIS_FEATURE_SIZE];
        aggregate_axis_features(&features, starts, &mut mean, &mut max);
        for i in 0..LOCAL_AXIS_FEATURE_SIZE {
            let mut expected_mean = 0.0;
            let mut expected_max = f32::NEG_INFINITY;
            for start in starts {
                let value = features[start + i];
                expected_mean += value / LOCAL_AXES as f32;
                expected_max = expected_max.max(value);
            }
            assert!((mean[i] - expected_mean).abs() < 1.0e-6);
            assert_eq!(max[i], expected_max);
        }
    }

    #[test]
    fn local_outputs_start_without_manual_bias() {
        let model = PolicyValueModel::random(8, 5);
        assert!(model.policy_local.iter().all(|&weight| weight == 0.0));
        assert!(model.policy_dynamic.iter().all(|&weight| weight == 0.0));
        assert!(model.value_local_output.iter().all(|&weight| weight == 0.0));
    }

    #[test]
    fn short_value_heads_do_not_affect_inference() {
        let model = PolicyValueModel::random(8, 7);
        let mut changed = model.clone();
        changed.short_value_head_output.fill(123.0);
        changed.short_value_head_bias.fill(-45.0);
        let board = Board::new();
        assert_eq!(model.evaluate(&board), changed.evaluate(&board));
    }

    #[test]
    fn local_axis_encoding_is_reflection_invariant() {
        let mut board = Board::new();
        for text in ["d5", "c5", "e5", "a1", "f5", "a2"] {
            assert!(board.play(Move::parse(text).unwrap()));
        }
        let candidate = Move::parse("g5").unwrap();
        let forward = local_ray_codes(&board, candidate, 0, 1);
        let backward = local_ray_codes(&board, candidate, 0, -1);
        assert_eq!(forward, backward);
    }

    #[test]
    fn model_roundtrip_preserves_local_parameters() {
        let path = std::env::temp_dir().join(format!(
            "gomoku-model-roundtrip-{}-{}.safetensors",
            std::process::id(),
            SplitMix64(11).next()
        ));
        let model = PolicyValueModel::random(8, 3);
        let mut board = Board::new();
        assert!(board.play(Move::parse("h8").unwrap()));
        assert!(board.play(Move::parse("h9").unwrap()));
        let expected_output = model.evaluate(&board);
        model.save(&path).unwrap();
        let restored = PolicyValueModel::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(restored.local_axis_embedding, model.local_axis_embedding);
        assert_eq!(restored.local_axis_scale, model.local_axis_scale);
        assert_eq!(restored.local_axis_bias, model.local_axis_bias);
        assert_eq!(restored.policy_local, model.policy_local);
        assert_eq!(restored.policy_dynamic, model.policy_dynamic);
        assert_eq!(restored.value_local_output, model.value_local_output);
        assert_eq!(restored.role_adapter_down, model.role_adapter_down);
        assert_eq!(restored.role_adapter_up, model.role_adapter_up);
        assert_eq!(restored.region_embedding, model.region_embedding);
        assert_eq!(restored.value_region_hidden, model.value_region_hidden);
        assert_eq!(restored.evaluate(&board), expected_output);
    }

    #[test]
    fn accumulator_contains_role_and_matches_relative_inputs() {
        let model = PolicyValueModel::random(12, 29);
        let mut board = Board::new();
        assert!(board.play(Move::parse("h8").unwrap()));
        assert!(board.play(Move::parse("h9").unwrap()));
        assert!(board.play(Move::parse("j8").unwrap()));
        let accumulator = model.accumulator(&board);
        for (role, actual) in [
            (Player::Black, &accumulator.black),
            (Player::White, &accumulator.white),
        ] {
            let mut expected = model.hidden_bias.clone();
            let role_offset = (ROLE_INPUT_START + role_index(role)) * model.hidden_size;
            for h in 0..model.hidden_size {
                expected[h] += model.input_hidden[role_offset + h];
            }
            for (sq, &stone) in board.cells().iter().enumerate() {
                if stone == 0 {
                    continue;
                }
                let player = if stone == Player::Black.stone() {
                    Player::Black
                } else {
                    Player::White
                };
                let side = usize::from(player != role);
                let mv = Move(sq);
                let offsets = [
                    (side * CELL_COUNT + sq) * model.hidden_size,
                    (side * BOARD_SIZE + mv.row()) * model.hidden_size,
                    (side * BOARD_SIZE + mv.col()) * model.hidden_size,
                    (side * (BOARD_SIZE * 2 - 1) + mv.row() + BOARD_SIZE - 1 - mv.col())
                        * model.hidden_size,
                    (side * (BOARD_SIZE * 2 - 1) + mv.row() + mv.col()) * model.hidden_size,
                    side * model.hidden_size,
                ];
                for h in 0..model.hidden_size {
                    expected[h] += model.input_hidden[offsets[0] + h]
                        + model.rank_hidden[offsets[1] + h]
                        + model.file_hidden[offsets[2] + h]
                        + model.diagonal_hidden[offsets[3] + h]
                        + model.anti_diagonal_hidden[offsets[4] + h]
                        + model.stone_hidden[offsets[5] + h];
                }
            }
            for (feature, value) in crate::features::encode(&board, role)
                .into_iter()
                .enumerate()
            {
                for h in 0..model.hidden_size {
                    expected[h] += model.input_hidden
                        [(GO_INPUT_START + feature) * model.hidden_size + h]
                        * value;
                }
            }
            let phase = board.move_count() as f32 / CELL_COUNT as f32;
            for h in 0..model.hidden_size {
                expected[h] += model.input_hidden[MOVE_COUNT_INPUT * model.hidden_size + h] * phase;
                assert!((expected[h] - actual[h]).abs() < 1.0e-6);
            }
        }
    }
}
