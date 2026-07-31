use crate::{
    game::CELL_COUNT,
    model::{
        AXIS_FEATURES, DIAGONAL_FEATURES, INPUT_SIZE, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE,
        LOCAL_AXIS_PATTERNS, LOCAL_CANDIDATE_SIZE, PolicyValueModel, STONE_TYPES, VALUE_HEAD_SIZE,
        VALUE_LOCAL_SIZE, WDL_SIZE, local_ray_codes,
    },
    replay::Sample,
    selfplay::TrainStats,
};
use candle_core::{Device, Tensor, Var, backprop::GradStore};
use candle_nn::{
    ops::log_softmax,
    optim::{AdamW, Optimizer, ParamsAdamW},
};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn training_device_name(requested: usize) -> io::Result<String> {
    Ok(make_device(requested)?.1)
}

pub fn train(
    model: &mut PolicyValueModel,
    samples: &[Sample],
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    requested_device: usize,
) -> io::Result<TrainStats> {
    let mut session = TrainingSession::new(model, None, requested_device, learning_rate)?;
    session.train_controlled(
        model,
        None,
        samples,
        epochs,
        learning_rate,
        batch_size,
        1.0,
        None,
    )
}

pub struct TrainingSession {
    replica: Replica,
    ema: Option<Replica>,
    optimizer: AdamW,
}

impl TrainingSession {
    pub fn new(
        model: &PolicyValueModel,
        ema_model: Option<&PolicyValueModel>,
        requested_device: usize,
        learning_rate: f32,
    ) -> io::Result<Self> {
        let (device, _) = make_device(requested_device)?;
        let replica = Replica::new(model, &device)?;
        let ema = ema_model
            .map(|model| Replica::new(model, &device))
            .transpose()?;
        let optimizer = AdamW::new(
            replica.vars(),
            ParamsAdamW {
                lr: learning_rate as f64,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 1e-4,
            },
        )
        .map_err(err)?;
        Ok(Self {
            replica,
            ema,
            optimizer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_controlled(
        &mut self,
        model: &mut PolicyValueModel,
        ema_model: Option<&mut PolicyValueModel>,
        samples: &[Sample],
        epochs: usize,
        learning_rate: f32,
        batch_size: usize,
        ema_decay: f32,
        stop: Option<&AtomicBool>,
    ) -> io::Result<TrainStats> {
        if samples.is_empty() || epochs == 0 || learning_rate <= 0.0 {
            return Ok(TrainStats::default());
        }
        self.optimizer.set_learning_rate(learning_rate as f64);
        let mut stats = TrainStats::default();
        for _ in 0..epochs {
            for batch in samples.chunks(batch_size.max(1)) {
                if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                    self.copy_models(model, ema_model)?;
                    return Ok(finalize_stats(stats));
                }
                let output = self.replica.backward(batch)?;
                stats.samples += output.samples;
                stats.policy_loss += output.policy_sum;
                stats.value_loss += output.value_sum;
                {
                    crate::scope_profile!("train.optimizer_step");
                    self.optimizer.step(&output.grads).map_err(err)?;
                }
                stats.optimizer_steps += 1;
                if let Some(ema) = &self.ema {
                    ema.update_ema_from(&self.replica, ema_decay)?;
                }
            }
        }
        self.copy_models(model, ema_model)?;
        Ok(finalize_stats(stats))
    }

    fn copy_models(
        &self,
        model: &mut PolicyValueModel,
        ema_model: Option<&mut PolicyValueModel>,
    ) -> io::Result<()> {
        self.replica.copy_to(model)?;
        if let (Some(ema), Some(ema_model)) = (&self.ema, ema_model) {
            ema.copy_to(ema_model)?;
        }
        Ok(())
    }
}

fn finalize_stats(mut stats: TrainStats) -> TrainStats {
    let count = stats.samples.max(1) as f32;
    stats.policy_loss /= count;
    stats.value_loss /= count;
    stats.loss = stats.policy_loss + stats.value_loss;
    stats
}

struct Replica {
    device: Device,
    input_hidden: Var,
    stone_hidden: Var,
    rank_hidden: Var,
    file_hidden: Var,
    diagonal_hidden: Var,
    anti_diagonal_hidden: Var,
    hidden_bias: Var,
    policy_hidden: Var,
    policy_bias: Var,
    local_axis_embedding: Var,
    policy_local: Var,
    value_head_hidden: Var,
    value_local_output: Var,
    value_head_bias: Var,
    value_head_hidden2: Var,
    value_head_bias2: Var,
    value_head_output: Var,
}
impl Replica {
    fn new(model: &PolicyValueModel, device: &Device) -> io::Result<Self> {
        let h = model.hidden_size;
        Ok(Self {
            device: device.clone(),
            input_hidden: var(&model.input_hidden, (INPUT_SIZE, h), device)?,
            stone_hidden: var(&model.stone_hidden, (STONE_TYPES, h), device)?,
            rank_hidden: var(&model.rank_hidden, (AXIS_FEATURES, h), device)?,
            file_hidden: var(&model.file_hidden, (AXIS_FEATURES, h), device)?,
            diagonal_hidden: var(&model.diagonal_hidden, (DIAGONAL_FEATURES, h), device)?,
            anti_diagonal_hidden: var(&model.anti_diagonal_hidden, (DIAGONAL_FEATURES, h), device)?,
            hidden_bias: var(&model.hidden_bias, (h,), device)?,
            policy_hidden: var(&model.policy_hidden, (CELL_COUNT, h), device)?,
            policy_bias: var(&model.policy_bias, (CELL_COUNT,), device)?,
            local_axis_embedding: var(
                &model.local_axis_embedding,
                (LOCAL_AXIS_PATTERNS, LOCAL_AXIS_FEATURE_SIZE),
                device,
            )?,
            policy_local: var(&model.policy_local, (LOCAL_CANDIDATE_SIZE,), device)?,
            value_head_hidden: var(&model.value_head_hidden, (VALUE_HEAD_SIZE, h), device)?,
            value_local_output: var(
                &model.value_local_output,
                (WDL_SIZE, VALUE_LOCAL_SIZE),
                device,
            )?,
            value_head_bias: var(&model.value_head_bias, (VALUE_HEAD_SIZE,), device)?,
            value_head_hidden2: var(
                &model.value_head_hidden2,
                (VALUE_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            value_head_bias2: var(&model.value_head_bias2, (VALUE_HEAD_SIZE,), device)?,
            value_head_output: var(
                &model.value_head_output,
                (WDL_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
        })
    }
    fn vars(&self) -> Vec<Var> {
        vec![
            self.input_hidden.clone(),
            self.stone_hidden.clone(),
            self.rank_hidden.clone(),
            self.file_hidden.clone(),
            self.diagonal_hidden.clone(),
            self.anti_diagonal_hidden.clone(),
            self.hidden_bias.clone(),
            self.policy_hidden.clone(),
            self.policy_bias.clone(),
            self.local_axis_embedding.clone(),
            self.policy_local.clone(),
            self.value_head_hidden.clone(),
            self.value_local_output.clone(),
            self.value_head_bias.clone(),
            self.value_head_hidden2.clone(),
            self.value_head_bias2.clone(),
            self.value_head_output.clone(),
        ]
    }
    fn backward(&self, samples: &[Sample]) -> io::Result<BatchOutput> {
        let packed = {
            crate::scope_profile!("train.pack");
            pack(samples)
        };
        let b = samples.len();
        crate::scope_profile!("train.tensor_h2d");
        let inputs = Tensor::from_vec(packed.inputs, (b, INPUT_SIZE), &self.device).map_err(err)?;
        let stone_counts =
            Tensor::from_vec(packed.stone_counts, (b, STONE_TYPES), &self.device).map_err(err)?;
        let rank_counts =
            Tensor::from_vec(packed.rank_counts, (b, AXIS_FEATURES), &self.device).map_err(err)?;
        let file_counts =
            Tensor::from_vec(packed.file_counts, (b, AXIS_FEATURES), &self.device).map_err(err)?;
        let diagonal_counts =
            Tensor::from_vec(packed.diagonal_counts, (b, DIAGONAL_FEATURES), &self.device)
                .map_err(err)?;
        let anti_diagonal_counts = Tensor::from_vec(
            packed.anti_diagonal_counts,
            (b, DIAGONAL_FEATURES),
            &self.device,
        )
        .map_err(err)?;
        let targets =
            Tensor::from_vec(packed.policy_targets, (b, CELL_COUNT), &self.device).map_err(err)?;
        let masks =
            Tensor::from_vec(packed.policy_masks, (b, CELL_COUNT), &self.device).map_err(err)?;
        let value_wdl =
            Tensor::from_vec(packed.value_wdl, (b, WDL_SIZE), &self.device).map_err(err)?;
        let local_axis_indices = Tensor::from_vec(
            packed.local_axis_indices,
            (b * CELL_COUNT * LOCAL_AXES,),
            &self.device,
        )
        .map_err(err)?;
        let local_legal_mask =
            Tensor::from_vec(packed.local_legal_mask, (b, CELL_COUNT, 1), &self.device)
                .map_err(err)?;
        let hidden = {
            crate::scope_profile!("train.forward");
            inputs
                .matmul(&self.input_hidden)
                .and_then(|x| x.add(&stone_counts.matmul(&self.stone_hidden)?))
                .and_then(|x| x.add(&rank_counts.matmul(&self.rank_hidden)?))
                .and_then(|x| x.add(&file_counts.matmul(&self.file_hidden)?))
                .and_then(|x| x.add(&diagonal_counts.matmul(&self.diagonal_hidden)?))
                .and_then(|x| x.add(&anti_diagonal_counts.matmul(&self.anti_diagonal_hidden)?))
                .and_then(|x| x.broadcast_add(&self.hidden_bias))
                .and_then(|x| x.relu())
                .map_err(err)?
        };
        let rms = hidden
            .sqr()
            .and_then(|x| x.mean_keepdim(1))
            .and_then(|x| x.affine(1.0, 1.0e-6))
            .and_then(|x| x.sqrt())
            .map_err(err)?;
        let hidden = hidden.broadcast_div(&rms).map_err(err)?;
        let local_axes = self
            .local_axis_embedding
            .as_tensor()
            .index_select(&local_axis_indices, 0)
            .and_then(|x| x.reshape((b, CELL_COUNT, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE)))
            .map_err(err)?;
        let local_mean = local_axes.mean(2).map_err(err)?;
        let local_max = local_axes.max(2).map_err(err)?;
        let local_candidates = Tensor::cat(&[&local_mean, &local_max], 2).map_err(err)?;
        let local_policy_logits = local_candidates
            .reshape((b * CELL_COUNT, LOCAL_CANDIDATE_SIZE))
            .and_then(|x| {
                x.matmul(
                    &self
                        .policy_local
                        .as_tensor()
                        .reshape((LOCAL_CANDIDATE_SIZE, 1))?,
                )
            })
            .and_then(|x| x.reshape((b, CELL_COUNT)))
            .map_err(err)?;
        let logits = hidden
            .matmul(&self.policy_hidden.t().map_err(err)?)
            .and_then(|x| x.broadcast_add(&self.policy_bias))
            .and_then(|x| x.add(&local_policy_logits))
            .and_then(|x| x.add(&masks))
            .map_err(err)?;
        let log_probs = log_softmax(&logits, 1).map_err(err)?;
        let policy_sum_tensor = targets
            .mul(&log_probs)
            .and_then(|x| x.sum_all())
            .and_then(|x| x.affine(-1.0, 0.0))
            .map_err(err)?;
        let legal_counts = local_legal_mask.sum(1).map_err(err)?;
        let masked_local_candidates = local_candidates
            .broadcast_mul(&local_legal_mask)
            .map_err(err)?;
        let local_board_mean = masked_local_candidates
            .sum(1)
            .and_then(|x| x.broadcast_div(&legal_counts))
            .map_err(err)?;
        let local_board_max = local_candidates
            .broadcast_add(&masks.unsqueeze(2).map_err(err)?)
            .and_then(|x| x.max(1))
            .map_err(err)?;
        let local_value = Tensor::cat(&[&local_board_mean, &local_board_max], 1).map_err(err)?;
        let value_features = hidden
            .matmul(&self.value_head_hidden.t().map_err(err)?)
            .and_then(|x| x.broadcast_add(&self.value_head_bias))
            .and_then(|x| x.relu())
            .and_then(|x| x.matmul(&self.value_head_hidden2.t()?))
            .and_then(|x| x.broadcast_add(&self.value_head_bias2))
            .and_then(|x| x.relu())
            .map_err(err)?;
        let value_logits = value_features
            .matmul(&self.value_head_output.t().map_err(err)?)
            .and_then(|x| x.add(&local_value.matmul(&self.value_local_output.t()?)?))
            .map_err(err)?;
        let value_log_probs = log_softmax(&value_logits, 1).map_err(err)?;
        let value_sum_tensor = value_wdl
            .mul(&value_log_probs)
            .and_then(|x| x.sum_all())
            .and_then(|x| x.affine(-1.0, 0.0))
            .map_err(err)?;
        let loss = policy_sum_tensor
            .add(&value_sum_tensor)
            .and_then(|x| x.affine(1.0 / b as f64, 0.0))
            .map_err(err)?;
        let policy_sum = policy_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let value_sum = value_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let grads = {
            crate::scope_profile!("train.backward");
            loss.backward().map_err(err)?
        };
        Ok(BatchOutput {
            grads,
            samples: b,
            policy_sum,
            value_sum,
        })
    }
    fn cpu_values(&self) -> io::Result<Vec<Vec<f32>>> {
        self.vars()
            .iter()
            .map(|v| {
                v.as_tensor()
                    .flatten_all()
                    .and_then(|x| x.to_device(&Device::Cpu))
                    .and_then(|x| x.to_vec1::<f32>())
                    .map_err(err)
            })
            .collect()
    }
    fn update_ema_from(&self, source: &Self, decay: f32) -> io::Result<()> {
        let decay = decay.clamp(0.0, 1.0) as f64;
        for (ema, online) in self.vars().iter().zip(source.vars()) {
            let blended = ema
                .as_tensor()
                .affine(decay, 0.0)
                .and_then(|x| x.add(&online.as_tensor().affine(1.0 - decay, 0.0)?))
                .map_err(err)?;
            ema.set(&blended).map_err(err)?;
        }
        Ok(())
    }
    fn copy_to(&self, m: &mut PolicyValueModel) -> io::Result<()> {
        let v = self.cpu_values()?;
        m.input_hidden = v[0].clone();
        m.stone_hidden = v[1].clone();
        m.rank_hidden = v[2].clone();
        m.file_hidden = v[3].clone();
        m.diagonal_hidden = v[4].clone();
        m.anti_diagonal_hidden = v[5].clone();
        m.hidden_bias = v[6].clone();
        m.policy_hidden = v[7].clone();
        m.policy_bias = v[8].clone();
        m.local_axis_embedding = v[9].clone();
        m.policy_local = v[10].clone();
        m.value_head_hidden = v[11].clone();
        m.value_local_output = v[12].clone();
        m.value_head_bias = v[13].clone();
        m.value_head_hidden2 = v[14].clone();
        m.value_head_bias2 = v[15].clone();
        m.value_head_output = v[16].clone();
        Ok(())
    }
}
struct BatchOutput {
    grads: GradStore,
    samples: usize,
    policy_sum: f32,
    value_sum: f32,
}
fn make_device(requested: usize) -> io::Result<(Device, String)> {
    #[cfg(target_os = "macos")]
    {
        return Device::new_metal(requested)
            .map(|device| (device, format!("metal:{requested}")))
            .map_err(err);
    }
    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    {
        match Device::new_cuda(requested) {
            Ok(device) => return Ok((device, format!("cuda:{requested}"))),
            Err(error) if requested == 0 => {
                eprintln!("train    : CUDA unavailable ({error}), falling back to CPU");
                return Ok((Device::Cpu, "cpu".into()));
            }
            Err(error) => return Err(err(error)),
        }
    }
    #[allow(unreachable_code)]
    {
        if requested != 0 {
            return Err(io::Error::other("当前平台仅支持 gpu_device = 0（CPU）"));
        }
        Ok((Device::Cpu, "cpu".into()))
    }
}

struct Packed {
    inputs: Vec<f32>,
    stone_counts: Vec<f32>,
    rank_counts: Vec<f32>,
    file_counts: Vec<f32>,
    diagonal_counts: Vec<f32>,
    anti_diagonal_counts: Vec<f32>,
    policy_targets: Vec<f32>,
    policy_masks: Vec<f32>,
    local_axis_indices: Vec<u32>,
    local_legal_mask: Vec<f32>,
    value_wdl: Vec<f32>,
}
fn pack(samples: &[Sample]) -> Packed {
    let mut inputs = vec![0.0; samples.len() * INPUT_SIZE];
    let mut stone_counts = vec![0.0; samples.len() * STONE_TYPES];
    let mut rank_counts = vec![0.0; samples.len() * AXIS_FEATURES];
    let mut file_counts = vec![0.0; samples.len() * AXIS_FEATURES];
    let mut diagonal_counts = vec![0.0; samples.len() * DIAGONAL_FEATURES];
    let mut anti_diagonal_counts = vec![0.0; samples.len() * DIAGONAL_FEATURES];
    let mut targets = vec![0.0; samples.len() * CELL_COUNT];
    let mut masks = vec![-1e9; samples.len() * CELL_COUNT];
    let mut local_axis_indices = vec![0_u32; samples.len() * CELL_COUNT * LOCAL_AXES];
    let mut local_legal_mask = vec![0.0; samples.len() * CELL_COUNT];
    let mut value_wdl = Vec::with_capacity(samples.len() * WDL_SIZE);
    for (row, s) in samples.iter().enumerate() {
        let us = s.board.to_move().stone();
        for (sq, &stone) in s.board.cells().iter().enumerate() {
            if stone == us {
                inputs[row * INPUT_SIZE + sq] = 1.0;
                stone_counts[row * STONE_TYPES] += 1.0;
                rank_counts[row * AXIS_FEATURES + sq / 15] += 1.0;
                file_counts[row * AXIS_FEATURES + sq % 15] += 1.0;
                diagonal_counts[row * DIAGONAL_FEATURES + sq / 15 + 14 - sq % 15] += 1.0;
                anti_diagonal_counts[row * DIAGONAL_FEATURES + sq / 15 + sq % 15] += 1.0;
            } else if stone == -us {
                inputs[row * INPUT_SIZE + CELL_COUNT + sq] = 1.0;
                stone_counts[row * STONE_TYPES + 1] += 1.0;
                rank_counts[row * AXIS_FEATURES + 15 + sq / 15] += 1.0;
                file_counts[row * AXIS_FEATURES + 15 + sq % 15] += 1.0;
                diagonal_counts[row * DIAGONAL_FEATURES + 29 + sq / 15 + 14 - sq % 15] += 1.0;
                anti_diagonal_counts[row * DIAGONAL_FEATURES + 29 + sq / 15 + sq % 15] += 1.0;
            }
        }
        inputs[row * INPUT_SIZE + INPUT_SIZE - 1] = s.board.move_count() as f32 / CELL_COUNT as f32;
        for m in s.board.search_candidates() {
            masks[row * CELL_COUNT + m.0] = 0.0;
            local_legal_mask[row * CELL_COUNT + m.0] = 1.0;
            for (axis, (dr, dc)) in [(1, 0), (0, 1), (1, 1), (1, -1)].into_iter().enumerate() {
                let (first, second) = local_ray_codes(&s.board, m, dr, dc);
                let pattern = second * (second + 1) / 2 + first;
                local_axis_indices[(row * CELL_COUNT + m.0) * LOCAL_AXES + axis] = pattern as u32;
            }
        }
        let sum: f32 = s.policy.iter().map(|(_, p)| p.max(0.0)).sum();
        for &(m, p) in &s.policy {
            if m.0 < CELL_COUNT && sum > 1e-12 {
                targets[row * CELL_COUNT + m.0] = p.max(0.0) / sum
            }
        }
        let final_wdl = if s.value > 0.5 {
            [1.0, 0.0, 0.0]
        } else if s.value < -0.5 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        value_wdl.extend_from_slice(&final_wdl);
    }
    Packed {
        inputs,
        stone_counts,
        rank_counts,
        file_counts,
        diagonal_counts,
        anti_diagonal_counts,
        policy_targets: targets,
        policy_masks: masks,
        local_axis_indices,
        local_legal_mask,
        value_wdl,
    }
}
fn var(data: &[f32], shape: impl Into<candle_core::Shape>, device: &Device) -> io::Result<Var> {
    Var::from_slice(data, shape, device).map_err(err)
}
fn err(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Board, Move};

    #[test]
    fn packing_uses_search_candidates_without_changing_rule_legality() {
        let mut board = Board::new();
        let occupied = Move::new(7, 7).unwrap();
        assert!(board.play(occupied));
        let corner = Move::new(0, 0).unwrap();
        let nearby = Move::new(7, 8).unwrap();
        let packed = pack(&[Sample {
            board,
            policy: vec![(nearby, 1.0)],
            value: 0.0,
            generation: 0,
        }]);

        assert_eq!(packed.policy_masks[occupied.0], -1e9);
        assert_eq!(packed.local_legal_mask[occupied.0], 0.0);
        assert_eq!(packed.policy_masks[corner.0], -1e9);
        assert_eq!(packed.local_legal_mask[corner.0], 0.0);
        assert_eq!(packed.policy_masks[nearby.0], 0.0);
        assert_eq!(packed.local_legal_mask[nearby.0], 1.0);
        assert_eq!(packed.policy_targets[nearby.0], 1.0);
    }

    #[test]
    fn trains_policy_and_value_on_available_device() {
        let mut model = PolicyValueModel::random(16, 9);
        let before_local = model.local_axis_embedding.clone();
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        assert!(board.play(Move::new(7, 8).unwrap()));
        let sample = Sample {
            board,
            policy: vec![(Move::new(8, 7).unwrap(), 1.0)],
            value: 1.0,
            generation: 0,
        };
        let stats = train(&mut model, &[sample], 2, 1e-3, 1, 0).unwrap();
        assert_eq!(stats.optimizer_steps, 2);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(model.policy_local.iter().any(|&weight| weight != 0.0));
        assert_ne!(model.local_axis_embedding, before_local);
        let (policy, value) = model.evaluate(&Board::new());
        assert_eq!(policy.len(), 1);
        assert!(
            policy
                .iter()
                .all(|(_, probability)| probability.is_finite())
        );
        assert!(value.is_finite());
    }
}
