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
pub const VALUE_PATTERN_SIZE: usize = LOCAL_AXIS_FEATURE_SIZE;
const FORMAT_VERSION: f32 = 13.0;
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
    pub(crate) value_head_bias: Vec<f32>,
    pub(crate) value_head_hidden2: Vec<f32>,
    pub(crate) value_head_bias2: Vec<f32>,
    pub(crate) value_head_output: Vec<f32>,
    pub(crate) value_pattern_output: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct EvalAccumulator {
    black: Vec<f32>,
    white: Vec<f32>,
    local_black: Vec<f32>,
    local_white: Vec<f32>,
    move_count: usize,
    pub(crate) hash: u64,
}

pub(crate) struct EvalScratch {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    local_candidate: Vec<f32>,
    value1: Vec<f32>,
    value2: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ValuePathBenchmark {
    pub iterations: usize,
    pub policy_value_seconds: f64,
    pub value_seconds: f64,
    pub value: f32,
    pub update_seconds: f64,
}

impl EvalScratch {
    pub(crate) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: Vec::with_capacity(hidden_size),
            logits: Vec::with_capacity(CELL_COUNT),
            local_candidate: vec![0.0; LOCAL_CANDIDATE_SIZE],
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
            value_head_bias: vec![0.0; VALUE_HEAD_SIZE],
            value_head_hidden2: (0..VALUE_HEAD_SIZE * VALUE_HEAD_SIZE)
                .map(|_| rng.weight((2.0 / VALUE_HEAD_SIZE as f32).sqrt() * 0.5))
                .collect(),
            value_head_bias2: vec![0.0; VALUE_HEAD_SIZE],
            value_head_output: vec![0.0; VALUE_HEAD_SIZE * WDL_SIZE],
            value_pattern_output: vec![0.0; VALUE_PATTERN_SIZE * WDL_SIZE],
        }
    }

    pub fn evaluate(&self, board: &Board) -> (Vec<(Move, f32)>, f32) {
        let accumulator = self.accumulator(board);
        self.evaluate_accumulator(board, &accumulator)
    }

    /// 唯一的局面价值入口；只读取棋盘增量状态，与候选生成策略无关。
    pub fn evaluate_value(&self, board: &Board) -> f32 {
        let accumulator = self.accumulator(board);
        let mut scratch = EvalScratch::new(self.hidden_size);
        self.evaluate_value_with_scratch(board, &accumulator, &mut scratch)
    }

    /// 返回指定合法着法的未归一化排序分；MCTS 可批量调用，PVS 可延迟调用。
    pub fn evaluate_move_logit(&self, board: &Board, mv: Move) -> Option<f32> {
        if !board.is_legal(mv) {
            return None;
        }
        let accumulator = self.accumulator(board);
        let mut scratch = EvalScratch::new(self.hidden_size);
        self.activate_accumulator(board, &accumulator, &mut scratch.hidden);
        self.local_candidate_into(board, mv, &mut scratch.local_candidate);
        Some(self.policy_logit(&scratch.hidden, &scratch.local_candidate, mv))
    }

    pub(crate) fn evaluate_value_accumulator(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        scratch: &mut EvalScratch,
    ) -> f32 {
        self.evaluate_value_with_scratch(board, accumulator, scratch)
    }

    pub(crate) fn move_logits_accumulator(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        moves: &[Move],
        scratch: &mut EvalScratch,
    ) -> Vec<(Move, f32)> {
        self.activate_accumulator(board, accumulator, &mut scratch.hidden);
        let mut scored = Vec::with_capacity(moves.len());
        for &mv in moves {
            self.local_candidate_into(board, mv, &mut scratch.local_candidate);
            scored.push((
                mv,
                self.policy_logit(&scratch.hidden, &scratch.local_candidate, mv),
            ));
        }
        scored
    }

    /// 比较完整 Policy+Value 与单独 Value 的推理成本。
    pub fn benchmark_value_paths(&self, board: &Board, iterations: usize) -> ValuePathBenchmark {
        let iterations = iterations.max(1);
        let accumulator = self.accumulator(board);
        let mut full_scratch = EvalScratch::new(self.hidden_size);
        let mut value_scratch = EvalScratch::new(self.hidden_size);

        let started = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(
                self.evaluate_accumulator_with_scratch(board, &accumulator, 1.0, &mut full_scratch)
                    .1,
            );
        }
        let policy_value_seconds = started.elapsed().as_secs_f64();

        let started = std::time::Instant::now();
        let mut value = 0.0;
        for _ in 0..iterations {
            value = std::hint::black_box(self.evaluate_value_with_scratch(
                board,
                &accumulator,
                &mut value_scratch,
            ));
        }
        let value_seconds = started.elapsed().as_secs_f64();

        let update_seconds = if let Some(&mv) = board.search_candidates().first() {
            let mut child_board = board.clone();
            let player = child_board.to_move();
            let _ = child_board.play(mv);
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                std::hint::black_box(self.accumulator_after_move(
                    &accumulator,
                    board,
                    &child_board,
                    mv,
                    player,
                ));
            }
            started.elapsed().as_secs_f64()
        } else {
            0.0
        };

        ValuePathBenchmark {
            iterations,
            policy_value_seconds,
            value_seconds,
            value,
            update_seconds,
        }
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
        self.activate_accumulator(board, accumulator, &mut scratch.hidden);
        let moves = board.search_candidates();
        if moves.is_empty() {
            return (Vec::new(), 0.0);
        }
        {
            crate::scope_profile!("model.policy_logits");
            scratch.logits.clear();
            for &mv in &moves {
                self.local_candidate_into(board, mv, &mut scratch.local_candidate);
                scratch.logits.push(self.policy_logit(
                    &scratch.hidden,
                    &scratch.local_candidate,
                    mv,
                ));
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
        let value = self.value_from_features(scratch, self.local_for(board, accumulator));
        (policy, value)
    }

    fn evaluate_value_with_scratch(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        scratch: &mut EvalScratch,
    ) -> f32 {
        crate::scope_profile!("model.evaluate_value");
        debug_assert_eq!(accumulator.move_count, board.move_count());
        self.activate_accumulator(board, accumulator, &mut scratch.hidden);
        self.value_from_features(scratch, self.local_for(board, accumulator))
    }

    fn activate_accumulator(
        &self,
        board: &Board,
        accumulator: &EvalAccumulator,
        hidden: &mut Vec<f32>,
    ) {
        crate::scope_profile!("model.activate_norm");
        self.activate_hidden_into(
            match board.to_move() {
                Player::Black => &accumulator.black,
                Player::White => &accumulator.white,
            },
            hidden,
        );
    }

    fn local_for<'a>(&self, board: &Board, accumulator: &'a EvalAccumulator) -> &'a [f32] {
        match board.to_move() {
            Player::Black => &accumulator.local_black,
            Player::White => &accumulator.local_white,
        }
    }

    fn value_from_features(&self, scratch: &mut EvalScratch, local: &[f32]) -> f32 {
        crate::scope_profile!("model.value_head");
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
            );
            *logit += dot(
                local,
                &self.value_pattern_output
                    [output * VALUE_PATTERN_SIZE..(output + 1) * VALUE_PATTERN_SIZE],
            );
        }
        let wdl_max = wdl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let wdl_sum: f32 = wdl.iter().map(|x| (x - wdl_max).exp()).sum();
        let wdl = wdl.map(|x| (x - wdl_max).exp() / wdl_sum);
        wdl[0] - wdl[2]
    }

    pub(crate) fn accumulator(&self, board: &Board) -> EvalAccumulator {
        crate::scope_profile!("model.accumulator_root");
        let mut accumulator = EvalAccumulator {
            black: self.hidden_bias.clone(),
            white: self.hidden_bias.clone(),
            local_black: self.local_board_summary(board, Player::Black),
            local_white: self.local_board_summary(board, Player::White),
            move_count: 0,
            hash: board_hash(board),
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
        board_before: &Board,
        board_after: &Board,
        mv: Move,
        player: Player,
    ) -> EvalAccumulator {
        crate::scope_profile!("model.accumulator_update");
        let mut child = parent.clone();
        self.add_stone(&mut child, mv, player);
        self.update_local_after_move(&mut child, board_before, board_after, mv);
        self.set_move_count(&mut child, parent.move_count + 1);
        child.hash ^= zobrist_piece(mv, player) ^ ZOBRIST_SIDE;
        child
    }

    fn local_board_summary(&self, board: &Board, perspective: Player) -> Vec<f32> {
        let mut summary = vec![0.0; VALUE_PATTERN_SIZE];
        let mut axis_feature = [0.0; VALUE_PATTERN_SIZE];
        for index in 0..CELL_COUNT {
            for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                self.local_axis_feature_for_player_into(
                    board,
                    Move(index),
                    dr,
                    dc,
                    perspective,
                    &mut axis_feature,
                );
                for (sum, value) in summary.iter_mut().zip(&axis_feature) {
                    *sum += *value / (CELL_COUNT * LOCAL_AXES) as f32;
                }
            }
        }
        summary
    }

    fn update_local_after_move(
        &self,
        accumulator: &mut EvalAccumulator,
        board_before: &Board,
        board_after: &Board,
        mv: Move,
    ) {
        self.update_local_contribution(accumulator, board_before, mv, -1.0);
        self.update_local_contribution(accumulator, board_after, mv, 1.0);
    }

    fn update_local_contribution(
        &self,
        accumulator: &mut EvalAccumulator,
        board: &Board,
        mv: Move,
        scale: f32,
    ) {
        let mut feature = [0.0; VALUE_PATTERN_SIZE];
        for (dr, dc) in [(1_i32, 0_i32), (0, 1), (1, 1), (1, -1)] {
            for sign in [-1_i32, 1] {
                for distance in 1..=LOCAL_RADIUS as i32 {
                    let row = mv.row() as i32 + dr * sign * distance;
                    let col = mv.col() as i32 + dc * sign * distance;
                    let Some(center) = (row >= 0
                        && col >= 0
                        && row < BOARD_SIZE as i32
                        && col < BOARD_SIZE as i32)
                        .then(|| Move(row as usize * BOARD_SIZE + col as usize))
                    else {
                        continue;
                    };
                    for perspective in [Player::Black, Player::White] {
                        self.local_axis_feature_for_player_into(
                            board,
                            center,
                            dr,
                            dc,
                            perspective,
                            &mut feature,
                        );
                        for i in 0..VALUE_PATTERN_SIZE {
                            let change = scale * feature[i] / (CELL_COUNT * LOCAL_AXES) as f32;
                            match perspective {
                                Player::Black => {
                                    accumulator.local_black[i] += change;
                                }
                                Player::White => {
                                    accumulator.local_white[i] += change;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn accumulator_prepare_move(
        &self,
        accumulator: &mut EvalAccumulator,
        board: &Board,
        mv: Move,
        player: Player,
    ) {
        self.update_local_contribution(accumulator, board, mv, -1.0);
        self.add_stone_scaled(accumulator, mv, player, 1.0);
        self.set_move_count(accumulator, accumulator.move_count + 1);
        accumulator.hash ^= zobrist_piece(mv, player) ^ ZOBRIST_SIDE;
    }

    pub(crate) fn accumulator_finish_move(
        &self,
        accumulator: &mut EvalAccumulator,
        board: &Board,
        mv: Move,
    ) {
        self.update_local_contribution(accumulator, board, mv, 1.0);
    }

    fn add_stone(&self, accumulator: &mut EvalAccumulator, mv: Move, player: Player) {
        self.add_stone_scaled(accumulator, mv, player, 1.0);
    }

    fn add_stone_scaled(
        &self,
        accumulator: &mut EvalAccumulator,
        mv: Move,
        player: Player,
        scale: f32,
    ) {
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
                *value += scale
                    * (self.input_hidden[exact + h]
                        + self.stone_hidden[stone + h]
                        + self.rank_hidden[rank + h]
                        + self.file_hidden[file + h]
                        + self.diagonal_hidden[diagonal + h]
                        + self.anti_diagonal_hidden[anti_diagonal + h]);
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
        self.local_candidate_for_player_into(board, mv, board.to_move(), output);
    }

    fn local_candidate_for_player_into(
        &self,
        board: &Board,
        mv: Move,
        perspective: Player,
        output: &mut [f32],
    ) {
        output.fill(0.0);
        let (mean, max) = output.split_at_mut(LOCAL_AXIS_FEATURE_SIZE);
        max.fill(f32::NEG_INFINITY);
        for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            let mut axis_feature = [0.0; LOCAL_AXIS_FEATURE_SIZE];
            self.local_axis_feature_for_player_into(
                board,
                mv,
                dr,
                dc,
                perspective,
                &mut axis_feature,
            );
            for i in 0..LOCAL_AXIS_FEATURE_SIZE {
                mean[i] += axis_feature[i] / LOCAL_AXES as f32;
                max[i] = max[i].max(axis_feature[i]);
            }
        }
    }

    fn local_axis_feature_for_player_into(
        &self,
        board: &Board,
        mv: Move,
        dr: i32,
        dc: i32,
        perspective: Player,
        output: &mut [f32],
    ) {
        let (first_code, second_code) = local_ray_codes_for_player(board, mv, dr, dc, perspective);
        let pattern = second_code * (second_code + 1) / 2 + first_code;
        output.copy_from_slice(
            &self.local_axis_embedding
                [pattern * LOCAL_AXIS_FEATURE_SIZE..(pattern + 1) * LOCAL_AXIS_FEATURE_SIZE],
        );
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
            "value_pattern_output",
            &self.value_pattern_output,
            (WDL_SIZE, VALUE_PATTERN_SIZE),
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
        if hidden_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "五子棋模型隐藏层不能为空",
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
            policy_hidden: load(&tensors, "policy_hidden")?,
            policy_bias: load(&tensors, "policy_bias")?,
            local_axis_embedding: load(&tensors, "local_axis_embedding")?,
            policy_local: load(&tensors, "policy_local")?,
            value_head_hidden: load(&tensors, "value_head_hidden")?,
            value_head_bias: load(&tensors, "value_head_bias")?,
            value_head_hidden2: load(&tensors, "value_head_hidden2")?,
            value_head_bias2: load(&tensors, "value_head_bias2")?,
            value_head_output: load(&tensors, "value_head_output")?,
            value_pattern_output: load(&tensors, "value_pattern_output")?,
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
            || model.value_head_bias.len() != VALUE_HEAD_SIZE
            || model.value_head_hidden2.len() != VALUE_HEAD_SIZE * VALUE_HEAD_SIZE
            || model.value_head_bias2.len() != VALUE_HEAD_SIZE
            || model.value_head_output.len() != VALUE_HEAD_SIZE * WDL_SIZE
            || model.value_pattern_output.len() != VALUE_PATTERN_SIZE * WDL_SIZE
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
        blend!(value_head_bias);
        blend!(value_head_hidden2);
        blend!(value_head_bias2);
        blend!(value_head_output);
        blend!(value_pattern_output);
    }
}

const ZOBRIST_SIDE: u64 = 0xA5A5_5A5A_D3C7_B19D;

fn zobrist_piece(mv: Move, player: Player) -> u64 {
    let mut value = mv.0 as u64
        ^ if player == Player::Black {
            0x9E37_79B9_7F4A_7C15
        } else {
            0xD1B5_4A32_D192_ED03
        };
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn board_hash(board: &Board) -> u64 {
    let mut hash = if board.to_move() == Player::White {
        ZOBRIST_SIDE
    } else {
        0
    };
    for (index, &stone) in board.cells().iter().enumerate() {
        let player = if stone == Player::Black.stone() {
            Some(Player::Black)
        } else if stone == Player::White.stone() {
            Some(Player::White)
        } else {
            None
        };
        if let Some(player) = player {
            hash ^= zobrist_piece(Move(index), player);
        }
    }
    hash
}

pub(crate) fn local_ray_codes(board: &Board, mv: Move, dr: i32, dc: i32) -> (usize, usize) {
    local_ray_codes_for_player(board, mv, dr, dc, board.to_move())
}

fn local_ray_codes_for_player(
    board: &Board,
    mv: Move,
    dr: i32,
    dc: i32,
    perspective: Player,
) -> (usize, usize) {
    let us = perspective.stone();
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
        online.value_head_output[0] = 1.0;
        let before = ema.policy_bias[0];
        let value_before = ema.value_head_output[0];
        ema.update_ema(&online, 0.75);
        let expected = before * 0.75 + online.policy_bias[0] * 0.25;
        let value_expected = value_before * 0.75 + online.value_head_output[0] * 0.25;
        assert!((ema.policy_bias[0] - expected).abs() < 1e-6);
        assert!((ema.value_head_output[0] - value_expected).abs() < 1e-6);
        assert_eq!(ema.policy_bias.len(), CELL_COUNT);
    }

    #[test]
    fn local_outputs_start_without_manual_bias() {
        let model = PolicyValueModel::random(8, 5);
        assert!(model.policy_local.iter().all(|&weight| weight == 0.0));
        assert!(model.value_head_output.iter().all(|&weight| weight == 0.0));
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
    fn lazy_move_logits_match_batched_policy_softmax() {
        let model = PolicyValueModel::random(16, 29);
        let mut board = Board::new();
        assert!(board.play(Move::parse("h8").unwrap()));
        let (policy, _) = model.evaluate(&board);
        let first = policy[0];
        let second = policy[1];
        let first_logit = model.evaluate_move_logit(&board, first.0).unwrap();
        let second_logit = model.evaluate_move_logit(&board, second.0).unwrap();

        let expected_ratio = (first_logit - second_logit).exp();
        let actual_ratio = first.1 / second.1;
        assert!((expected_ratio - actual_ratio).abs() < 1e-5);
        assert!(
            model
                .evaluate_move_logit(&board, Move::parse("h8").unwrap())
                .is_none()
        );
    }

    #[test]
    fn value_incremental_state_matches_rebuilt_state() {
        let mut model = PolicyValueModel::random(16, 19);
        model.value_head_output[0] = 1.0;
        model.value_head_output[VALUE_HEAD_SIZE * 2] = -1.0;
        model.value_pattern_output[0] = 1.0;
        model.value_pattern_output[VALUE_PATTERN_SIZE * 2] = -1.0;

        let mut board = Board::new();
        assert!(board.play(Move::parse("h8").unwrap()));
        let parent = model.accumulator(&board);
        let player = board.to_move();
        let mv = Move::parse("h9").unwrap();
        let board_before = board.clone();
        assert!(board.play(mv));

        let child = model.accumulator_after_move(&parent, &board_before, &board, mv, player);
        let rebuilt = model.accumulator(&board);
        for (incremental, rebuilt) in child.local_black.iter().zip(&rebuilt.local_black) {
            assert!((incremental - rebuilt).abs() < 1e-6);
        }
        for (incremental, rebuilt) in child.local_white.iter().zip(&rebuilt.local_white) {
            assert!((incremental - rebuilt).abs() < 1e-6);
        }
        let mut child_scratch = EvalScratch::new(model.hidden_size);
        let mut rebuilt_scratch = EvalScratch::new(model.hidden_size);
        let child_value = model.evaluate_value_with_scratch(&board, &child, &mut child_scratch);
        let rebuilt_value =
            model.evaluate_value_with_scratch(&board, &rebuilt, &mut rebuilt_scratch);

        assert!((child_value - rebuilt_value).abs() < 1e-6);
        assert!((model.evaluate_value(&board) - rebuilt_value).abs() < 1e-6);
    }

    #[test]
    fn local_board_summary_is_symmetry_invariant() {
        let model = PolicyValueModel::random(16, 31);
        let mut board = Board::new();
        for text in ["h8", "h9", "i8", "g9", "j7"] {
            assert!(board.play(Move::parse(text).unwrap()));
        }
        let black = model.local_board_summary(&board, Player::Black);
        let white = model.local_board_summary(&board, Player::White);
        for symmetry in 0..8 {
            let transformed = board.transformed(symmetry);
            let transformed_black = model.local_board_summary(&transformed, Player::Black);
            let transformed_white = model.local_board_summary(&transformed, Player::White);
            for (left, right) in black.iter().zip(transformed_black) {
                assert!((left - right).abs() < 1e-6);
            }
            for (left, right) in white.iter().zip(transformed_white) {
                assert!((left - right).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn v13_model_roundtrip_preserves_policy_and_value_parameters() {
        let path = std::env::temp_dir().join(format!(
            "gomoku-v13-roundtrip-{}-{}.safetensors",
            std::process::id(),
            SplitMix64(11).next()
        ));
        let model = PolicyValueModel::random(8, 3);
        model.save(&path).unwrap();
        let restored = PolicyValueModel::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(restored.local_axis_embedding, model.local_axis_embedding);
        assert_eq!(restored.policy_local, model.policy_local);
        assert_eq!(restored.value_head_output, model.value_head_output);
        assert_eq!(restored.value_pattern_output, model.value_pattern_output);
    }
}
