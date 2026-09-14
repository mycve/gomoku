use crate::{
    fused_feature_pool::sparse_pool,
    game::{ACTION_COUNT, BOARD_SIZE, CELL_COUNT},
    model::{
        AXIS_FEATURES, DIAGONAL_FEATURES, INPUT_SIZE, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE,
        LOCAL_AXIS_PATTERNS, LOCAL_CANDIDATE_SIZE, MOVE_COUNT_INPUT, POLICY_HEAD_SIZE,
        POLICY_TACTICAL_SIZE, PolicyValueModel, REGION_COUNT, REGION_FEATURE_SIZE,
        REGION_TOTAL_SIZE, ROLE_ADAPTER_RANK, ROLE_COUNT, ROLE_INPUT_START, STONE_TYPES,
        VALUE_HEAD_SIZE, VALUE_LOCAL_SIZE, local_ray_codes, policy_tactical_index,
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

pub fn training_device_name() -> io::Result<String> {
    make_device(0).map(|(_, name)| name)
}

pub fn train(
    model: &mut PolicyValueModel,
    samples: &[Sample],
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
) -> io::Result<TrainStats> {
    let mut session = TrainingSession::new(model, learning_rate)?;
    session.train_controlled(model, samples, epochs, learning_rate, batch_size, None)
}

pub struct TrainingSession {
    replica: Replica,
    optimizer: AdamW,
}

impl TrainingSession {
    pub fn new(model: &PolicyValueModel, learning_rate: f32) -> io::Result<Self> {
        let (device, _) = make_device(0)?;
        let replica = Replica::new(model, &device)?;
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
        Ok(Self { replica, optimizer })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_controlled(
        &mut self,
        model: &mut PolicyValueModel,
        samples: &[Sample],
        epochs: usize,
        learning_rate: f32,
        batch_size: usize,
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
                    self.copy_model(model)?;
                    return Ok(finalize_stats(stats));
                }
                let output = self.train_batch(batch)?;
                stats.samples += output.samples;
                stats.policy_loss += output.policy_sum;
                stats.value_loss += output.value_sum;
                stats.policy_entropy += output.policy_entropy_sum;
                stats.value_entropy += output.value_entropy_sum;
                stats.optimizer_steps += 1;
            }
        }
        self.copy_model(model)?;
        Ok(finalize_stats(stats))
    }

    /// 计算样本损失但不更新参数。当前与训练共用完全相同的前向与标签处理。
    pub fn evaluate(&self, samples: &[Sample], batch_size: usize) -> io::Result<TrainStats> {
        if samples.is_empty() {
            return Ok(TrainStats::default());
        }
        let mut stats = TrainStats::default();
        for batch in samples.chunks(batch_size.max(1)) {
            let output = self.replica.forward(batch, false, batch.len())?;
            stats.samples += output.samples;
            stats.policy_loss += output.policy_sum;
            stats.value_loss += output.value_sum;
            stats.policy_entropy += output.policy_entropy_sum;
            stats.value_entropy += output.value_entropy_sum;
        }
        Ok(finalize_stats(stats))
    }

    fn copy_model(&self, model: &mut PolicyValueModel) -> io::Result<()> {
        self.replica.copy_to(model)?;
        Ok(())
    }

    fn train_batch(&mut self, batch: &[Sample]) -> io::Result<BatchOutput> {
        let output = self.replica.forward(batch, true, batch.len())?;
        crate::scope_profile!("train.optimizer_step");
        self.optimizer
            .step(output.grads.as_ref().expect("训练批次必须包含梯度"))
            .map_err(err)?;
        Ok(output)
    }
}

fn finalize_stats(mut stats: TrainStats) -> TrainStats {
    let count = stats.samples.max(1) as f32;
    stats.policy_loss /= count;
    stats.value_loss /= count;
    stats.policy_entropy /= count;
    stats.value_entropy /= count;
    stats.policy_kl = (stats.policy_loss - stats.policy_entropy).max(0.0);
    stats.value_kl = (stats.value_loss - stats.value_entropy).max(0.0);
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
    role_adapter_down: Var,
    role_adapter_up: Var,
    region_embedding: Var,
    policy_global: Var,
    policy_global_bias: Var,
    policy_dynamic: Var,
    policy_output: Var,
    policy_bias: Var,
    policy_tactical: Var,
    local_axis_embedding: Var,
    local_axis_scale: Var,
    local_axis_bias: Var,
    policy_local: Var,
    value_head_hidden: Var,
    value_region_hidden: Var,
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
            role_adapter_down: var(
                &model.role_adapter_down,
                (ROLE_COUNT * ROLE_ADAPTER_RANK, h),
                device,
            )?,
            role_adapter_up: var(
                &model.role_adapter_up,
                (ROLE_COUNT * h, ROLE_ADAPTER_RANK),
                device,
            )?,
            region_embedding: var(
                &model.region_embedding,
                (REGION_COUNT, STONE_TYPES, REGION_FEATURE_SIZE),
                device,
            )?,
            policy_global: var(&model.policy_global, (POLICY_HEAD_SIZE, h), device)?,
            policy_global_bias: var(&model.policy_global_bias, (POLICY_HEAD_SIZE,), device)?,
            policy_dynamic: var(
                &model.policy_dynamic,
                (LOCAL_CANDIDATE_SIZE, POLICY_HEAD_SIZE),
                device,
            )?,
            policy_output: var(
                &model.policy_output,
                (ACTION_COUNT, POLICY_HEAD_SIZE),
                device,
            )?,
            policy_bias: var(&model.policy_bias, (ACTION_COUNT,), device)?,
            policy_tactical: var(&model.policy_tactical, (POLICY_TACTICAL_SIZE, 1), device)?,
            local_axis_embedding: var(
                &model.local_axis_embedding,
                (LOCAL_AXIS_PATTERNS, LOCAL_AXIS_FEATURE_SIZE),
                device,
            )?,
            local_axis_scale: var(
                &model.local_axis_scale,
                (2, LOCAL_AXIS_FEATURE_SIZE),
                device,
            )?,
            local_axis_bias: var(&model.local_axis_bias, (2, LOCAL_AXIS_FEATURE_SIZE), device)?,
            policy_local: var(&model.policy_local, (LOCAL_CANDIDATE_SIZE,), device)?,
            value_head_hidden: var(&model.value_head_hidden, (VALUE_HEAD_SIZE, h), device)?,
            value_region_hidden: var(
                &model.value_region_hidden,
                (VALUE_HEAD_SIZE, REGION_TOTAL_SIZE),
                device,
            )?,
            value_local_output: var(&model.value_local_output, (1, VALUE_LOCAL_SIZE), device)?,
            value_head_bias: var(&model.value_head_bias, (VALUE_HEAD_SIZE,), device)?,
            value_head_hidden2: var(
                &model.value_head_hidden2,
                (VALUE_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            value_head_bias2: var(&model.value_head_bias2, (VALUE_HEAD_SIZE,), device)?,
            value_head_output: var(&model.value_head_output, (1, VALUE_HEAD_SIZE), device)?,
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
            self.role_adapter_down.clone(),
            self.role_adapter_up.clone(),
            self.region_embedding.clone(),
            self.policy_global.clone(),
            self.policy_global_bias.clone(),
            self.policy_dynamic.clone(),
            self.policy_output.clone(),
            self.policy_bias.clone(),
            self.policy_tactical.clone(),
            self.local_axis_embedding.clone(),
            self.local_axis_scale.clone(),
            self.local_axis_bias.clone(),
            self.policy_local.clone(),
            self.value_head_hidden.clone(),
            self.value_region_hidden.clone(),
            self.value_local_output.clone(),
            self.value_head_bias.clone(),
            self.value_head_hidden2.clone(),
            self.value_head_bias2.clone(),
            self.value_head_output.clone(),
        ]
    }
    fn forward(
        &self,
        samples: &[Sample],
        backward: bool,
        global_batch_size: usize,
    ) -> io::Result<BatchOutput> {
        let packed = {
            crate::scope_profile!("train.pack");
            pack(samples)
        };
        let b = samples.len();
        let h = self.hidden_bias.dim(0).map_err(err)?;
        #[cfg(feature = "profile")]
        let transfer_timer = crate::profile::ScopeTimer::new("train.tensor_h2d");
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
        let roles = Tensor::from_vec(packed.roles, (b, ROLE_COUNT), &self.device).map_err(err)?;
        let region_counts = Tensor::from_vec(
            packed.region_counts,
            (b, REGION_COUNT, STONE_TYPES, 1),
            &self.device,
        )
        .map_err(err)?;
        let targets = Tensor::from_vec(packed.policy_targets, (b, ACTION_COUNT), &self.device)
            .map_err(err)?;
        let masks =
            Tensor::from_vec(packed.policy_masks, (b, ACTION_COUNT), &self.device).map_err(err)?;
        let win_targets =
            Tensor::from_vec(packed.win_targets, (b, 1), &self.device).map_err(err)?;
        let policy_weights =
            Tensor::from_vec(packed.policy_weights, (b,), &self.device).map_err(err)?;
        let value_weights =
            Tensor::from_vec(packed.value_weights, (b,), &self.device).map_err(err)?;
        let local_axis_indices = Tensor::from_vec(
            packed.local_axis_indices,
            (b * ACTION_COUNT * LOCAL_AXES,),
            &self.device,
        )
        .map_err(err)?;
        let policy_tactical_indices = Tensor::from_vec(
            packed.policy_tactical_indices,
            (b * ACTION_COUNT, LOCAL_AXES),
            &self.device,
        )
        .map_err(err)?;
        let local_legal_mask =
            Tensor::from_vec(packed.local_legal_mask, (b, ACTION_COUNT, 1), &self.device)
                .map_err(err)?;
        #[cfg(feature = "profile")]
        drop(transfer_timer);
        #[cfg(feature = "profile")]
        let forward_timer = crate::profile::ScopeTimer::new("train.forward_to_scalar");
        let hidden = {
            crate::scope_profile!("train.trunk_enqueue");
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
        let mut adapted = hidden.clone();
        for role in 0..ROLE_COUNT {
            let down = self
                .role_adapter_down
                .narrow(0, role * ROLE_ADAPTER_RANK, ROLE_ADAPTER_RANK)
                .map_err(err)?;
            let up = self.role_adapter_up.narrow(0, role * h, h).map_err(err)?;
            let mask = roles.narrow(1, role, 1).map_err(err)?;
            let residual = hidden
                .matmul(&down.t().map_err(err)?)
                .and_then(|x| x.relu())
                .and_then(|x| x.matmul(&up.t()?))
                .and_then(|x| x.broadcast_mul(&mask))
                .map_err(err)?;
            adapted = adapted.add(&residual).map_err(err)?;
        }
        let adapted_rms = adapted
            .sqr()
            .and_then(|x| x.mean_keepdim(1))
            .and_then(|x| x.affine(1.0, 1.0e-6))
            .and_then(|x| x.sqrt())
            .map_err(err)?;
        let hidden = adapted.broadcast_div(&adapted_rms).map_err(err)?;
        let region_features = region_counts
            .broadcast_mul(&self.region_embedding.unsqueeze(0).map_err(err)?)
            .and_then(|x| x.sum(2))
            .and_then(|x| x.reshape((b, REGION_TOTAL_SIZE)))
            .map_err(err)?;
        // 与 chineseai 相同：通过前向/反向融合的稀疏池化查表，
        // 不构造巨大中间张量。每行只有一个 item，因此池化等价于 lookup。
        let local_axis_items = local_axis_indices
            .reshape((b * ACTION_COUNT * LOCAL_AXES, 1))
            .map_err(err)?;
        let local_axes = sparse_pool(self.local_axis_embedding.as_tensor(), &local_axis_items)
            .and_then(|x| x.reshape((b, ACTION_COUNT, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE)))
            .map_err(err)?;
        let axis_scale = Tensor::cat(
            &[
                &self
                    .local_axis_scale
                    .as_tensor()
                    .narrow(0, 0, 1)
                    .map_err(err)?,
                &self
                    .local_axis_scale
                    .as_tensor()
                    .narrow(0, 0, 1)
                    .map_err(err)?,
                &self
                    .local_axis_scale
                    .as_tensor()
                    .narrow(0, 1, 1)
                    .map_err(err)?,
                &self
                    .local_axis_scale
                    .as_tensor()
                    .narrow(0, 1, 1)
                    .map_err(err)?,
            ],
            0,
        )
        .and_then(|x| x.reshape((1, 1, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE)))
        .map_err(err)?;
        let axis_bias = Tensor::cat(
            &[
                &self
                    .local_axis_bias
                    .as_tensor()
                    .narrow(0, 0, 1)
                    .map_err(err)?,
                &self
                    .local_axis_bias
                    .as_tensor()
                    .narrow(0, 0, 1)
                    .map_err(err)?,
                &self
                    .local_axis_bias
                    .as_tensor()
                    .narrow(0, 1, 1)
                    .map_err(err)?,
                &self
                    .local_axis_bias
                    .as_tensor()
                    .narrow(0, 1, 1)
                    .map_err(err)?,
            ],
            0,
        )
        .and_then(|x| x.reshape((1, 1, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE)))
        .map_err(err)?;
        let local_axes = local_axes
            .broadcast_mul(&axis_scale)
            .and_then(|x| x.broadcast_add(&axis_bias))
            .map_err(err)?;
        let local_mean = local_axes.mean(2).map_err(err)?;
        let local_max = local_axes.max(2).map_err(err)?;
        let local_candidates = Tensor::cat(&[&local_mean, &local_max], 2).map_err(err)?;
        let policy_global = hidden
            .matmul(&self.policy_global.t().map_err(err)?)
            .and_then(|x| x.broadcast_add(&self.policy_global_bias))
            .and_then(|x| x.relu())
            .map_err(err)?;
        let dynamic_weights = policy_global
            .matmul(&self.policy_dynamic.t().map_err(err)?)
            .and_then(|x| x.broadcast_add(&self.policy_local))
            .and_then(|x| x.unsqueeze(1))
            .map_err(err)?;
        let local_policy_logits = local_candidates
            .broadcast_mul(&dynamic_weights)
            .and_then(|x| x.sum(2))
            .map_err(err)?;
        let tactical_logits =
            sparse_pool(self.policy_tactical.as_tensor(), &policy_tactical_indices)
                .and_then(|x| x.reshape((b, ACTION_COUNT)))
                .map_err(err)?;
        let logits = policy_global
            .matmul(&self.policy_output.t().map_err(err)?)
            .and_then(|x| x.add(&local_policy_logits))
            .and_then(|x| x.broadcast_add(&self.policy_bias))
            .and_then(|x| x.add(&tactical_logits))
            .and_then(|x| x.add(&masks))
            .map_err(err)?;
        let log_probs = log_softmax(&logits, 1).map_err(err)?;
        let policy_sum_tensor = targets
            .mul(&log_probs)
            .and_then(|x| x.sum(1))
            .and_then(|x| x.mul(&policy_weights))
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
            .and_then(|x| x.add(&region_features.matmul(&self.value_region_hidden.t()?)?))
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
        let value_losses = binary_value_losses(&value_logits, &win_targets).map_err(err)?;
        let value_sum_tensor = value_losses
            .sum(1)
            .and_then(|x| x.mul(&value_weights))
            .and_then(|x| x.sum_all())
            .map_err(err)?;
        let loss = policy_sum_tensor
            .add(&value_sum_tensor)
            .and_then(|x| x.affine(1.0 / global_batch_size.max(1) as f64, 0.0))
            .map_err(err)?;
        let policy_sum = policy_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let value_sum = value_sum_tensor.to_scalar::<f32>().map_err(err)?;
        #[cfg(feature = "profile")]
        drop(forward_timer);
        let grads = if backward {
            crate::scope_profile!("train.backward");
            Some(loss.backward().map_err(err)?)
        } else {
            None
        };
        Ok(BatchOutput {
            grads,
            samples: b,
            policy_sum,
            value_sum,
            policy_entropy_sum: packed.policy_entropy_sum,
            value_entropy_sum: packed.value_entropy_sum,
        })
    }
    fn cpu_values(&self) -> io::Result<Vec<Vec<f32>>> {
        crate::scope_profile!("train.model_d2h");
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
    fn copy_to(&self, m: &mut PolicyValueModel) -> io::Result<()> {
        crate::scope_profile!("train.publish_model");
        let v = self.cpu_values()?;
        m.input_hidden = v[0].clone();
        m.stone_hidden = v[1].clone();
        m.rank_hidden = v[2].clone();
        m.file_hidden = v[3].clone();
        m.diagonal_hidden = v[4].clone();
        m.anti_diagonal_hidden = v[5].clone();
        m.hidden_bias = v[6].clone();
        m.role_adapter_down = v[7].clone();
        m.role_adapter_up = v[8].clone();
        m.region_embedding = v[9].clone();
        m.policy_global = v[10].clone();
        m.policy_global_bias = v[11].clone();
        m.policy_dynamic = v[12].clone();
        m.policy_output = v[13].clone();
        m.policy_bias = v[14].clone();
        m.policy_tactical = v[15].clone();
        m.local_axis_embedding = v[16].clone();
        m.local_axis_scale = v[17].clone();
        m.local_axis_bias = v[18].clone();
        m.policy_local = v[19].clone();
        m.value_head_hidden = v[20].clone();
        m.value_region_hidden = v[21].clone();
        m.value_local_output = v[22].clone();
        m.value_head_bias = v[23].clone();
        m.value_head_hidden2 = v[24].clone();
        m.value_head_bias2 = v[25].clone();
        m.value_head_output = v[26].clone();
        m.refresh_local_axis_features();
        Ok(())
    }
}
#[derive(Default)]
struct BatchOutput {
    grads: Option<GradStore>,
    samples: usize,
    policy_sum: f32,
    value_sum: f32,
    policy_entropy_sum: f32,
    value_entropy_sum: f32,
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
            return Err(io::Error::other("当前平台仅支持设备 0（CPU）"));
        }
        Ok((Device::Cpu, "cpu".into()))
    }
}

struct Packed {
    inputs: Vec<f32>,
    roles: Vec<f32>,
    region_counts: Vec<f32>,
    stone_counts: Vec<f32>,
    rank_counts: Vec<f32>,
    file_counts: Vec<f32>,
    diagonal_counts: Vec<f32>,
    anti_diagonal_counts: Vec<f32>,
    policy_targets: Vec<f32>,
    policy_masks: Vec<f32>,
    local_axis_indices: Vec<u32>,
    policy_tactical_indices: Vec<u32>,
    local_legal_mask: Vec<f32>,
    win_targets: Vec<f32>,
    policy_weights: Vec<f32>,
    value_weights: Vec<f32>,
    policy_entropy_sum: f32,
    value_entropy_sum: f32,
}
fn pack(samples: &[Sample]) -> Packed {
    let mut inputs = vec![0.0; samples.len() * INPUT_SIZE];
    let mut roles = vec![0.0; samples.len() * ROLE_COUNT];
    let mut region_counts = vec![0.0; samples.len() * REGION_COUNT * STONE_TYPES];
    let mut stone_counts = vec![0.0; samples.len() * STONE_TYPES];
    let mut rank_counts = vec![0.0; samples.len() * AXIS_FEATURES];
    let mut file_counts = vec![0.0; samples.len() * AXIS_FEATURES];
    let mut diagonal_counts = vec![0.0; samples.len() * DIAGONAL_FEATURES];
    let mut anti_diagonal_counts = vec![0.0; samples.len() * DIAGONAL_FEATURES];
    let mut targets = vec![0.0; samples.len() * ACTION_COUNT];
    let mut masks = vec![-1e9; samples.len() * ACTION_COUNT];
    let mut local_axis_indices = vec![0_u32; samples.len() * ACTION_COUNT * LOCAL_AXES];
    let mut policy_tactical_indices = vec![u32::MAX; samples.len() * ACTION_COUNT * LOCAL_AXES];
    let mut local_legal_mask = vec![0.0; samples.len() * ACTION_COUNT];
    let mut win_targets = Vec::with_capacity(samples.len());
    let mut policy_weights = Vec::with_capacity(samples.len());
    let mut value_weights = Vec::with_capacity(samples.len());
    let mut policy_entropy_sum = 0.0;
    let mut value_entropy_sum = 0.0;
    for (row, s) in samples.iter().enumerate() {
        policy_weights.push(s.policy_weight.max(0.0));
        value_weights.push(s.value_weight.max(0.0));
        let extras = crate::features::encode(&s.board, s.board.to_move());
        inputs[row * INPUT_SIZE + crate::model::GO_INPUT_START..(row + 1) * INPUT_SIZE]
            .copy_from_slice(&extras);
        let us = s.board.to_move().stone();
        for (sq, &stone) in s.board.cells().iter().enumerate() {
            if stone == us {
                inputs[row * INPUT_SIZE + sq] = 1.0;
                stone_counts[row * STONE_TYPES] += 1.0;
                rank_counts[row * AXIS_FEATURES + sq / BOARD_SIZE] += 1.0;
                file_counts[row * AXIS_FEATURES + sq % BOARD_SIZE] += 1.0;
                diagonal_counts[row * DIAGONAL_FEATURES + sq / BOARD_SIZE + BOARD_SIZE
                    - 1
                    - sq % BOARD_SIZE] += 1.0;
                anti_diagonal_counts
                    [row * DIAGONAL_FEATURES + sq / BOARD_SIZE + sq % BOARD_SIZE] += 1.0;
                let region =
                    (sq / BOARD_SIZE / (BOARD_SIZE / 3)) * 3 + (sq % BOARD_SIZE) / (BOARD_SIZE / 3);
                region_counts[(row * REGION_COUNT + region) * STONE_TYPES] += 1.0;
            } else if stone == -us {
                inputs[row * INPUT_SIZE + CELL_COUNT + sq] = 1.0;
                stone_counts[row * STONE_TYPES + 1] += 1.0;
                rank_counts[row * AXIS_FEATURES + BOARD_SIZE + sq / BOARD_SIZE] += 1.0;
                file_counts[row * AXIS_FEATURES + BOARD_SIZE + sq % BOARD_SIZE] += 1.0;
                diagonal_counts[row * DIAGONAL_FEATURES
                    + (BOARD_SIZE * 2 - 1)
                    + sq / BOARD_SIZE
                    + BOARD_SIZE
                    - 1
                    - sq % BOARD_SIZE] += 1.0;
                anti_diagonal_counts[row * DIAGONAL_FEATURES
                    + (BOARD_SIZE * 2 - 1)
                    + sq / BOARD_SIZE
                    + sq % BOARD_SIZE] += 1.0;
                let region =
                    (sq / BOARD_SIZE / (BOARD_SIZE / 3)) * 3 + (sq % BOARD_SIZE) / (BOARD_SIZE / 3);
                region_counts[(row * REGION_COUNT + region) * STONE_TYPES + 1] += 1.0;
            }
        }
        inputs[row * INPUT_SIZE + MOVE_COUNT_INPUT] =
            s.board.move_count() as f32 / CELL_COUNT as f32;
        inputs[row * INPUT_SIZE + crate::model::PASS_INPUT] = s.board.consecutive_passes() as f32;
        let role = usize::from(s.board.to_move() == crate::game::Player::White);
        inputs[row * INPUT_SIZE + ROLE_INPUT_START + role] = 1.0;
        roles[row * ROLE_COUNT + role] = 1.0;
        for m in
            crate::features::moves_from_mask(&s.board, &extras[10 * CELL_COUNT..11 * CELL_COUNT])
        {
            masks[row * ACTION_COUNT + m.0] = 0.0;
            local_legal_mask[row * ACTION_COUNT + m.0] = 1.0;
            for (axis, (dr, dc)) in [(1, 0), (0, 1), (1, 1), (1, -1)].into_iter().enumerate() {
                let (first, second) = local_ray_codes(&s.board, m, dr, dc);
                let pattern = second * (second + 1) / 2 + first;
                local_axis_indices[(row * ACTION_COUNT + m.0) * LOCAL_AXES + axis] = pattern as u32;
                policy_tactical_indices[(row * ACTION_COUNT + m.0) * LOCAL_AXES + axis] =
                    policy_tactical_index(m, axis, pattern) as u32;
            }
        }
        let sum: f32 = s.policy.iter().map(|(_, p)| p.max(0.0)).sum();
        let mut policy_entropy = 0.0;
        for &(m, p) in &s.policy {
            if m.0 < ACTION_COUNT && sum > 1e-12 {
                let probability = p.max(0.0) / sum;
                targets[row * ACTION_COUNT + m.0] = probability;
                if probability > 0.0 {
                    policy_entropy -= probability * probability.ln();
                }
            }
        }
        policy_entropy_sum += policy_entropy * s.policy_weight.max(0.0);
        let target = (s.value + 1.0) * 0.5;
        value_entropy_sum -= s.value_weight.max(0.0)
            * [target, 1.0 - target]
                .into_iter()
                .filter(|&p| p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f32>();
        win_targets.push(target);
    }
    Packed {
        inputs,
        roles,
        region_counts,
        stone_counts,
        rank_counts,
        file_counts,
        diagonal_counts,
        anti_diagonal_counts,
        policy_targets: targets,
        policy_masks: masks,
        local_axis_indices,
        policy_tactical_indices,
        local_legal_mask,
        win_targets,
        policy_weights,
        value_weights,
        policy_entropy_sum,
        value_entropy_sum,
    }
}
// 单个可训练 logit，使用库的稳定 log_softmax 计算加权 BCE。
fn binary_value_losses(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros(logits.shape(), logits.dtype(), logits.device())?;
    let log_probs = log_softmax(&Tensor::cat(&[logits, &zeros], 1)?, 1)?;
    let targets = Tensor::cat(&[targets, &targets.affine(-1.0, 1.0)?], 1)?;
    targets.mul(&log_probs)?.neg()
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
    fn binary_value_loss_is_stable_and_has_correct_gradient() {
        for device in [Device::Cpu, make_device(0).unwrap().0] {
            let logits = Var::from_slice(&[-1000f32, 0.0, 1000.0, 0.0], (4, 1), &device).unwrap();
            let targets = Tensor::from_slice(&[1f32, 1.0, 0.0, 0.0], (4, 1), &device).unwrap();
            let loss = binary_value_losses(&logits, &targets)
                .unwrap()
                .sum_all()
                .unwrap();
            assert!((loss.to_scalar::<f32>().unwrap() - (2000.0 + 2.0 * 2f32.ln())).abs() < 1e-3);
            let grads = loss.backward().unwrap();
            let actual = grads
                .get(&logits)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (actual, expected) in actual.iter().zip([-1.0, -0.5, 1.0, 0.5]) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn packing_masks_only_rule_illegal_moves() {
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
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        }]);

        assert_eq!(packed.policy_masks[occupied.0], -1e9);
        assert_eq!(packed.local_legal_mask[occupied.0], 0.0);
        assert_eq!(packed.policy_masks[corner.0], 0.0);
        assert_eq!(packed.local_legal_mask[corner.0], 1.0);
        assert_eq!(packed.policy_masks[nearby.0], 0.0);
        assert_eq!(packed.local_legal_mask[nearby.0], 1.0);
        assert_eq!(packed.policy_targets[nearby.0], 1.0);
    }

    #[test]
    fn packing_has_explicit_absolute_role() {
        let black = Board::new();
        let mut white = Board::new();
        assert!(white.play(Move::new(7, 7).unwrap()));
        let sample = |board| Sample {
            board,
            policy: Vec::new(),
            value: 0.0,
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        };
        let packed = pack(&[sample(black), sample(white)]);
        assert_eq!(&packed.roles, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(packed.inputs[ROLE_INPUT_START], 1.0);
        assert_eq!(packed.inputs[INPUT_SIZE + ROLE_INPUT_START + 1], 1.0);
        assert_eq!(packed.inputs[MOVE_COUNT_INPUT], 0.0);
        assert_eq!(
            packed.inputs[INPUT_SIZE + MOVE_COUNT_INPUT],
            1.0 / CELL_COUNT as f32
        );
    }

    #[test]
    fn packing_reports_soft_target_entropy() {
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let first = Move::new(7, 8).unwrap();
        let second = Move::new(8, 7).unwrap();
        let policy = [0.25_f32, 0.75];
        let target = 0.35_f32;
        let packed = pack(&[Sample {
            board,
            policy: vec![(first, policy[0]), (second, policy[1])],
            value: 2.0 * target - 1.0,
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        }]);
        let expected_policy = -policy.iter().map(|p| p * p.ln()).sum::<f32>();
        let expected_value = -[target, 1.0 - target]
            .iter()
            .map(|p| p * p.ln())
            .sum::<f32>();
        assert!((packed.policy_entropy_sum - expected_policy).abs() < 1e-6);
        assert!((packed.value_entropy_sum - expected_value).abs() < 1e-6);
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
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        };
        let stats = train(
            &mut model,
            &[sample.clone(), sample.clone(), sample],
            2,
            1e-3,
            3,
        )
        .unwrap();
        assert_eq!(stats.optimizer_steps, 2);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!((stats.loss - stats.policy_loss - stats.value_loss).abs() < 1.0e-5);
        assert!(model.policy_local.iter().any(|&weight| weight != 0.0));
        assert_ne!(model.local_axis_embedding, before_local);
        let (policy, value) = model.evaluate(&Board::new());
        assert_eq!(policy.len(), ACTION_COUNT);
        assert!(
            policy
                .iter()
                .all(|(_, probability)| probability.is_finite())
        );
        assert!(value.is_finite());
    }

    #[test]
    fn pass_training_matches_inference_after_capture_and_pass() {
        let stones = [
            ("a2", crate::game::Player::Black),
            ("b1", crate::game::Player::Black),
            ("c2", crate::game::Player::Black),
            ("b2", crate::game::Player::White),
        ]
        .map(|(s, p)| (Move::parse(s).unwrap(), p));
        let mut board = Board::from_position(&stones, crate::game::Player::Black).unwrap();
        assert!(board.play(Move::parse("b3").unwrap()));
        assert_eq!(board.cells()[Move::parse("b2").unwrap().0], 0);
        assert!(board.play(Move::PASS));
        let sample = Sample {
            board,
            policy: vec![(Move::PASS, 1.0)],
            value: 1.0,
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        };
        let packed = pack(std::slice::from_ref(&sample));
        assert_eq!(packed.policy_targets[Move::PASS.0], 1.0);
        assert_eq!(packed.inputs[crate::model::PASS_INPUT], 1.0);
        let mut model = PolicyValueModel::random(16, 71);
        let before = model.policy_bias[Move::PASS.0];
        train(&mut model, std::slice::from_ref(&sample), 2, 1e-3, 1).unwrap();
        assert!(model.policy_bias[Move::PASS.0] > before);
        let (policy, value) = model.evaluate(&sample.board);
        let probability = policy.iter().find(|(mv, _)| *mv == Move::PASS).unwrap().1;
        let session = TrainingSession::new(&model, 1e-3).unwrap();
        let stats = session.evaluate(std::slice::from_ref(&sample), 1).unwrap();
        assert!((stats.policy_loss + probability.ln()).abs() < 1e-4);
        assert!((stats.value_loss + ((value + 1.0) * 0.5).ln()).abs() < 1e-4);
        let mut loss_sample = sample;
        loss_sample.value = -1.0;
        let stats = session.evaluate(&[loss_sample], 1).unwrap();
        assert!((stats.value_loss + ((1.0 - value) * 0.5).ln()).abs() < 1e-4);
    }
}
