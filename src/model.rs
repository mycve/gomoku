use crate::game::{BOARD_SIZE, Board, CELL_COUNT, Move, Player};
use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;
use std::{fs, io, path::Path};

pub const INPUT_SIZE: usize = CELL_COUNT * 2 + 1;
pub const DEFAULT_HIDDEN_SIZE: usize = 192;
pub const VALUE_HEAD_SIZE: usize = 96;
pub const WDL_SIZE: usize = 3;
pub const STONE_TYPES: usize = 2;
pub const AXIS_FEATURES: usize = STONE_TYPES * 15;
pub const DIAGONAL_FEATURES: usize = STONE_TYPES * (BOARD_SIZE * 2 - 1);
pub const LOCAL_AXES: usize = 4;
pub const LOCAL_RADIUS: usize = 4;
pub const LOCAL_RAY_PATTERNS: usize = 4usize.pow(LOCAL_RADIUS as u32);
pub const LOCAL_AXIS_PATTERNS: usize = LOCAL_RAY_PATTERNS * (LOCAL_RAY_PATTERNS + 1) / 2;
pub const LOCAL_AXIS_FEATURE_SIZE: usize = 2;
pub const LOCAL_CANDIDATE_SIZE: usize = LOCAL_AXIS_FEATURE_SIZE * 2;
pub const VALUE_LOCAL_SIZE: usize = LOCAL_CANDIDATE_SIZE * 2;
const FORMAT_VERSION: f32 = 10.0;
const LOCAL_BOUNDARY: u8 = u8::MAX;
const LOCAL_NEIGHBORS: [u8; CELL_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS] = build_local_neighbors();

const fn build_local_neighbors() -> [u8; CELL_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS] {
    let mut table = [LOCAL_BOUNDARY; CELL_COUNT * LOCAL_AXES * 2 * LOCAL_RADIUS];
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
    pub(crate) policy_hidden: Vec<f32>,
    pub(crate) policy_bias: Vec<f32>,
    pub(crate) local_axis_embedding: Vec<f32>,
    pub(crate) policy_local: Vec<f32>,
    pub(crate) value_head_hidden: Vec<f32>,
    pub(crate) value_local_output: Vec<f32>,
    pub(crate) value_head_bias: Vec<f32>,
    pub(crate) value_head_hidden2: Vec<f32>,
    pub(crate) value_head_bias2: Vec<f32>,
    pub(crate) value_head_output: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct EvalAccumulator {
    black: Vec<f32>,
    white: Vec<f32>,
    move_count: usize,
}

pub(crate) struct EvalScratch {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    local_candidate: Vec<f32>,
    local_value: Vec<f32>,
    value1: Vec<f32>,
    value2: Vec<f32>,
}

impl EvalScratch {
    pub(crate) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: Vec::with_capacity(hidden_size),
            logits: Vec::with_capacity(CELL_COUNT),
            local_candidate: vec![0.0; LOCAL_CANDIDATE_SIZE],
            local_value: vec![0.0; VALUE_LOCAL_SIZE],
            value1: Vec::with_capacity(VALUE_HEAD_SIZE),
            value2: Vec::with_capacity(VALUE_HEAD_SIZE),
        }
    }
}

impl Default for PolicyValueModel {
    fn default() -> Self {
        Self::random(DEFAULT_HIDDEN_SIZE, 20260730)
    }
}

impl PolicyValueModel {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        let hidden_size = hidden_size.max(1);
        let mut rng = SplitMix64(seed);
        let input_scale = (2.0 / INPUT_SIZE as f32).sqrt();
        let head_scale = (2.0 / hidden_size as f32).sqrt() * 0.25;
        let input_hidden = (0..INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(input_scale))
            .collect();
        let policy_hidden = (0..CELL_COUNT * hidden_size)
            .map(|_| rng.weight(head_scale))
            .collect();
        let policy_bias = vec![0.0; CELL_COUNT];
        Self {
            hidden_size,
            input_hidden,
            stone_hidden: vec![0.0; STONE_TYPES * hidden_size],
            rank_hidden: vec![0.0; AXIS_FEATURES * hidden_size],
            file_hidden: vec![0.0; AXIS_FEATURES * hidden_size],
            diagonal_hidden: vec![0.0; DIAGONAL_FEATURES * hidden_size],
            anti_diagonal_hidden: vec![0.0; DIAGONAL_FEATURES * hidden_size],
            hidden_bias: vec![0.0; hidden_size],
            policy_hidden,
            policy_bias,
            local_axis_embedding: (0..LOCAL_AXIS_PATTERNS * LOCAL_AXIS_FEATURE_SIZE)
                .map(|_| rng.weight((2.0 / LOCAL_AXIS_FEATURE_SIZE as f32).sqrt() * 0.25))
                .collect(),
            policy_local: vec![0.0; LOCAL_CANDIDATE_SIZE],
            value_head_hidden: (0..hidden_size * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / hidden_size as f32).sqrt() * 0.5))
                .collect(),
            value_local_output: vec![0.0; WDL_SIZE * VALUE_LOCAL_SIZE],
            value_head_bias: vec![0.0; VALUE_HEAD_SIZE],
            value_head_hidden2: (0..VALUE_HEAD_SIZE * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / VALUE_HEAD_SIZE as f32).sqrt() * 0.5))
                .collect(),
            value_head_bias2: vec![0.0; VALUE_HEAD_SIZE],
            value_head_output: vec![0.0; VALUE_HEAD_SIZE * WDL_SIZE],
        }
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
        crate::scope_profile!("model.evaluate_incremental");
        debug_assert_eq!(accumulator.move_count, board.move_count());
        {
            crate::scope_profile!("model.activate_norm");
            self.activate_hidden_into(
                match board.to_move() {
                    Player::Black => &accumulator.black,
                    Player::White => &accumulator.white,
                },
                &mut scratch.hidden,
            );
        }
        let moves = board.legal_moves();
        if moves.is_empty() {
            return (Vec::new(), 0.0);
        }
        {
            crate::scope_profile!("model.policy_logits");
            scratch.logits.clear();
            scratch.local_value.fill(0.0);
            scratch.local_value[LOCAL_CANDIDATE_SIZE..].fill(f32::NEG_INFINITY);
            for &mv in &moves {
                self.local_candidate_into(board, mv, &mut scratch.local_candidate);
                scratch.logits.push(self.policy_logit(
                    &scratch.hidden,
                    &scratch.local_candidate,
                    mv,
                ));
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
        for x in &mut scratch.local_value[..LOCAL_CANDIDATE_SIZE] {
            *x *= inverse_moves;
        }
        scratch.value1.clear();
        scratch.value1.extend_from_slice(&self.value_head_bias);
        for (output, value) in scratch.value1.iter_mut().enumerate() {
            let start = output * self.hidden_size;
            *value += dot(
                &scratch.hidden,
                &self.value_head_hidden[start..start + self.hidden_size],
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
            move_count: 0,
        };
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
        accumulator
    }

    pub(crate) fn accumulator_after_move(
        &self,
        parent: &EvalAccumulator,
        mv: Move,
        player: Player,
    ) -> EvalAccumulator {
        crate::scope_profile!("model.accumulator_update");
        let mut child = parent.clone();
        self.add_stone(&mut child, mv, player);
        self.set_move_count(&mut child, parent.move_count + 1);
        child
    }

    fn add_stone(&self, accumulator: &mut EvalAccumulator, mv: Move, player: Player) {
        for (perspective, hidden) in [
            (Player::Black, &mut accumulator.black),
            (Player::White, &mut accumulator.white),
        ] {
            let side = usize::from(player != perspective);
            let exact = (side * CELL_COUNT + mv.0) * self.hidden_size;
            let rank = (side * 15 + mv.row()) * self.hidden_size;
            let file = (side * 15 + mv.col()) * self.hidden_size;
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
        let rule_offset = (INPUT_SIZE - 1) * self.hidden_size;
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

    fn local_candidate_into(&self, board: &Board, mv: Move, output: &mut [f32]) {
        output.fill(0.0);
        let (mean, max) = output.split_at_mut(LOCAL_AXIS_FEATURE_SIZE);
        max.fill(f32::NEG_INFINITY);
        for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            let (first_code, second_code) = local_ray_codes(board, mv, dr, dc);
            let pattern = second_code * (second_code + 1) / 2 + first_code;
            let axis_feature = &self.local_axis_embedding
                [pattern * LOCAL_AXIS_FEATURE_SIZE..(pattern + 1) * LOCAL_AXIS_FEATURE_SIZE];
            for i in 0..LOCAL_AXIS_FEATURE_SIZE {
                mean[i] += axis_feature[i] / LOCAL_AXES as f32;
                max[i] = max[i].max(axis_feature[i]);
            }
        }
    }

    fn policy_logit(&self, hidden: &[f32], local: &[f32], mv: Move) -> f32 {
        dot(
            hidden,
            &self.policy_hidden[mv.0 * self.hidden_size..(mv.0 + 1) * self.hidden_size],
        ) + dot(local, &self.policy_local)
            + self.policy_bias[mv.0]
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
            "policy_hidden",
            &self.policy_hidden,
            (CELL_COUNT, self.hidden_size),
        )?;
        insert(&vars, "policy_bias", &self.policy_bias, (CELL_COUNT,))?;
        insert(
            &vars,
            "local_axis_embedding",
            &self.local_axis_embedding,
            (LOCAL_AXIS_PATTERNS, LOCAL_AXIS_FEATURE_SIZE),
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
                "不支持的五子棋模型版本",
            ));
        }
        let hidden_bias = load(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len();
        let model = Self {
            hidden_size,
            input_hidden: load(&tensors, "input_hidden")?,
            stone_hidden: load(&tensors, "stone_hidden")?,
            rank_hidden: load(&tensors, "rank_hidden")?,
            file_hidden: load(&tensors, "file_hidden")?,
            diagonal_hidden: load(&tensors, "diagonal_hidden")?,
            anti_diagonal_hidden: load(&tensors, "anti_diagonal_hidden")?,
            hidden_bias,
            policy_hidden: load(&tensors, "policy_hidden")?,
            policy_bias: load(&tensors, "policy_bias")?,
            local_axis_embedding: load(&tensors, "local_axis_embedding")?,
            policy_local: load(&tensors, "policy_local")?,
            value_head_hidden: load(&tensors, "value_head_hidden")?,
            value_local_output: load(&tensors, "value_local_output")?,
            value_head_bias: load(&tensors, "value_head_bias")?,
            value_head_hidden2: load(&tensors, "value_head_hidden2")?,
            value_head_bias2: load(&tensors, "value_head_bias2")?,
            value_head_output: load(&tensors, "value_head_output")?,
        };
        if model.input_hidden.len() != INPUT_SIZE * hidden_size
            || model.stone_hidden.len() != STONE_TYPES * hidden_size
            || model.rank_hidden.len() != AXIS_FEATURES * hidden_size
            || model.file_hidden.len() != AXIS_FEATURES * hidden_size
            || model.diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.anti_diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.policy_hidden.len() != CELL_COUNT * hidden_size
            || model.policy_bias.len() != CELL_COUNT
            || model.local_axis_embedding.len() != LOCAL_AXIS_PATTERNS * LOCAL_AXIS_FEATURE_SIZE
            || model.policy_local.len() != LOCAL_CANDIDATE_SIZE
            || model.value_head_hidden.len() != hidden_size * VALUE_HEAD_SIZE
            || model.value_local_output.len() != WDL_SIZE * VALUE_LOCAL_SIZE
            || model.value_head_bias.len() != VALUE_HEAD_SIZE
            || model.value_head_hidden2.len() != VALUE_HEAD_SIZE * VALUE_HEAD_SIZE
            || model.value_head_bias2.len() != VALUE_HEAD_SIZE
            || model.value_head_output.len() != VALUE_HEAD_SIZE * WDL_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "五子棋模型张量尺寸错误",
            ));
        }
        Ok(model)
    }

    pub fn update_ema(&mut self, online: &Self, decay: f32) {
        let decay = decay.clamp(0.0, 1.0);
        let keep = 1.0 - decay;
        macro_rules! blend {
            ($field:ident) => {
                for (ema, current) in self.$field.iter_mut().zip(&online.$field) {
                    *ema = decay * *ema + keep * *current;
                }
            };
        }
        blend!(input_hidden);
        blend!(stone_hidden);
        blend!(rank_hidden);
        blend!(file_hidden);
        blend!(diagonal_hidden);
        blend!(anti_diagonal_hidden);
        blend!(hidden_bias);
        blend!(policy_hidden);
        blend!(policy_bias);
        blend!(local_axis_embedding);
        blend!(policy_local);
        blend!(value_head_hidden);
        blend!(value_local_output);
        blend!(value_head_bias);
        blend!(value_head_hidden2);
        blend!(value_head_bias2);
        blend!(value_head_output);
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
    fn ema_blends_every_parameter_group() {
        let mut online = PolicyValueModel::random(8, 2);
        let mut ema = PolicyValueModel::random(8, 1);
        assert!(ema.policy_bias.iter().all(|&bias| bias == 0.0));
        online.policy_bias[0] = 1.0;
        let before = ema.policy_bias[0];
        ema.update_ema(&online, 0.75);
        let expected = before * 0.75 + online.policy_bias[0] * 0.25;
        assert!((ema.policy_bias[0] - expected).abs() < 1e-6);
        assert_eq!(ema.policy_bias.len(), CELL_COUNT);
    }

    #[test]
    fn local_axis_encoding_is_reflection_invariant() {
        let mut board = Board::new();
        for text in ["h8", "g8", "i8", "a1", "j8", "a2"] {
            assert!(board.play(Move::parse(text).unwrap()));
        }
        let candidate = Move::parse("k8").unwrap();
        let forward = local_ray_codes(&board, candidate, 0, 1);
        let backward = local_ray_codes(&board, candidate, 0, -1);
        assert_eq!(forward, backward);
    }

    #[test]
    fn v10_model_roundtrip_preserves_local_parameters() {
        let path = std::env::temp_dir().join(format!(
            "gomoku-v10-roundtrip-{}-{}.safetensors",
            std::process::id(),
            SplitMix64(11).next()
        ));
        let model = PolicyValueModel::random(8, 3);
        model.save(&path).unwrap();
        let restored = PolicyValueModel::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(restored.local_axis_embedding, model.local_axis_embedding);
        assert_eq!(restored.policy_local, model.policy_local);
        assert_eq!(restored.value_local_output, model.value_local_output);
    }
}
