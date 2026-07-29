use crate::game::{
    BOARD_SIZE, Board, CELL_COUNT, GLOBAL_TACTICAL_FEATURES, Move, Player, TACTICAL_FEATURES,
    add_global_tactical_features, normalize_global_tactical_features, tactical_features,
};
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
const FORMAT_VERSION: f32 = 8.0;

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
    pub(crate) policy_tactical: Vec<f32>,
    pub(crate) value_head_hidden: Vec<f32>,
    pub(crate) global_tactical_value: Vec<f32>,
    pub(crate) value_head_bias: Vec<f32>,
    pub(crate) value_head_hidden2: Vec<f32>,
    pub(crate) value_head_bias2: Vec<f32>,
    pub(crate) value_head_output: Vec<f32>,
    pub(crate) moves_left_output: Vec<f32>,
    pub(crate) moves_left_bias: Vec<f32>,
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
    value1: Vec<f32>,
    value2: Vec<f32>,
}

impl EvalScratch {
    pub(crate) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: Vec::with_capacity(hidden_size),
            logits: Vec::with_capacity(CELL_COUNT),
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
        let mut policy_bias = vec![0.0; CELL_COUNT];
        for (sq, bias) in policy_bias.iter_mut().enumerate() {
            let r = sq / 15;
            let c = sq % 15;
            *bias = 0.15 * (1.0 - ((r as f32 - 7.0).abs() + (c as f32 - 7.0).abs()) / 14.0);
        }
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
            policy_tactical: vec![0.0; TACTICAL_FEATURES],
            value_head_hidden: (0..hidden_size * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / hidden_size as f32).sqrt() * 0.5))
                .collect(),
            global_tactical_value: vec![0.0; GLOBAL_TACTICAL_FEATURES * VALUE_HEAD_SIZE],
            value_head_bias: vec![0.0; VALUE_HEAD_SIZE],
            value_head_hidden2: (0..VALUE_HEAD_SIZE * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / VALUE_HEAD_SIZE as f32).sqrt() * 0.5))
                .collect(),
            value_head_bias2: vec![0.0; VALUE_HEAD_SIZE],
            value_head_output: vec![0.0; VALUE_HEAD_SIZE * WDL_SIZE],
            moves_left_output: vec![0.0; VALUE_HEAD_SIZE],
            moves_left_bias: vec![0.0],
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
        let mut global_tactical = [0.0; GLOBAL_TACTICAL_FEATURES];
        {
            crate::scope_profile!("model.policy_logits");
            scratch.logits.clear();
            for &mv in &moves {
                let tactical = tactical_features(board, mv);
                add_global_tactical_features(&mut global_tactical, &tactical);
                scratch
                    .logits
                    .push(self.policy_logit(&scratch.hidden, mv, &tactical));
            }
            normalize_global_tactical_features(&mut global_tactical);
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
            .into_iter()
            .zip(
                scratch
                    .logits
                    .iter()
                    .copied()
                    .map(|x| ((x - max) * inverse_temperature).exp() / sum),
            )
            .collect();
        crate::scope_profile!("model.value_head");
        scratch.value1.clear();
        scratch.value1.extend_from_slice(&self.value_head_bias);
        for (output, value) in scratch.value1.iter_mut().enumerate() {
            let start = output * self.hidden_size;
            *value += dot(
                &scratch.hidden,
                &self.value_head_hidden[start..start + self.hidden_size],
            ) + dot(
                &global_tactical,
                &self.global_tactical_value
                    [output * GLOBAL_TACTICAL_FEATURES..(output + 1) * GLOBAL_TACTICAL_FEATURES],
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
            for h in 0..self.hidden_size {
                hidden[h] += self.input_hidden[exact + h]
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

    fn policy_logit(&self, hidden: &[f32], mv: Move, tactical: &[f32; TACTICAL_FEATURES]) -> f32 {
        dot(
            hidden,
            &self.policy_hidden[mv.0 * self.hidden_size..(mv.0 + 1) * self.hidden_size],
        ) + self.policy_bias[mv.0]
            + dot(tactical, &self.policy_tactical)
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
            "policy_tactical",
            &self.policy_tactical,
            (TACTICAL_FEATURES,),
        )?;
        insert(
            &vars,
            "value_head_hidden",
            &self.value_head_hidden,
            (VALUE_HEAD_SIZE, self.hidden_size),
        )?;
        insert(
            &vars,
            "global_tactical_value",
            &self.global_tactical_value,
            (VALUE_HEAD_SIZE, GLOBAL_TACTICAL_FEATURES),
        )?;
        insert(
            &vars,
            "value_head_bias",
            &self.value_head_bias,
            (VALUE_HEAD_SIZE,),
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
            "moves_left_output",
            &self.moves_left_output,
            (VALUE_HEAD_SIZE, 1),
        )?;
        insert(&vars, "moves_left_bias", &self.moves_left_bias, (1,))?;
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
            policy_tactical: load(&tensors, "policy_tactical")?,
            value_head_hidden: load(&tensors, "value_head_hidden")?,
            global_tactical_value: load(&tensors, "global_tactical_value")?,
            value_head_bias: load(&tensors, "value_head_bias")?,
            value_head_hidden2: load(&tensors, "value_head_hidden2")?,
            value_head_bias2: load(&tensors, "value_head_bias2")?,
            value_head_output: load(&tensors, "value_head_output")?,
            moves_left_output: load(&tensors, "moves_left_output")?,
            moves_left_bias: load(&tensors, "moves_left_bias")?,
        };
        if model.input_hidden.len() != INPUT_SIZE * hidden_size
            || model.stone_hidden.len() != STONE_TYPES * hidden_size
            || model.rank_hidden.len() != AXIS_FEATURES * hidden_size
            || model.file_hidden.len() != AXIS_FEATURES * hidden_size
            || model.diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.anti_diagonal_hidden.len() != DIAGONAL_FEATURES * hidden_size
            || model.policy_hidden.len() != CELL_COUNT * hidden_size
            || model.policy_bias.len() != CELL_COUNT
            || model.policy_tactical.len() != TACTICAL_FEATURES
            || model.value_head_hidden.len() != hidden_size * VALUE_HEAD_SIZE
            || model.global_tactical_value.len() != GLOBAL_TACTICAL_FEATURES * VALUE_HEAD_SIZE
            || model.value_head_bias.len() != VALUE_HEAD_SIZE
            || model.value_head_hidden2.len() != VALUE_HEAD_SIZE * VALUE_HEAD_SIZE
            || model.value_head_bias2.len() != VALUE_HEAD_SIZE
            || model.value_head_output.len() != VALUE_HEAD_SIZE * WDL_SIZE
            || model.moves_left_output.len() != VALUE_HEAD_SIZE
            || model.moves_left_bias.len() != 1
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
        blend!(policy_tactical);
        blend!(value_head_hidden);
        blend!(global_tactical_value);
        blend!(value_head_bias);
        blend!(value_head_hidden2);
        blend!(value_head_bias2);
        blend!(value_head_output);
        blend!(moves_left_output);
        blend!(moves_left_bias);
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
        let online = PolicyValueModel::random(8, 2);
        let mut ema = PolicyValueModel::random(8, 1);
        let before = ema.policy_bias[0];
        ema.update_ema(&online, 0.75);
        let expected = before * 0.75 + online.policy_bias[0] * 0.25;
        assert!((ema.policy_bias[0] - expected).abs() < 1e-6);
        assert_eq!(ema.moves_left_output.len(), VALUE_HEAD_SIZE);
    }
}
