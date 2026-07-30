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
#[cfg(all(target_os = "linux", not(target_env = "musl")))]
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn training_device_names(requested: &[usize]) -> io::Result<Vec<String>> {
    Ok(make_devices(requested)?
        .into_iter()
        .map(|(_, name)| name)
        .collect())
}

pub fn train(
    model: &mut PolicyValueModel,
    samples: &[Sample],
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    requested_devices: &[usize],
) -> io::Result<TrainStats> {
    let mut session = TrainingSession::new(model, None, requested_devices, learning_rate)?;
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
    replicas: Vec<Replica>,
    ema: Option<Replica>,
    optimizer: AdamW,
}

impl TrainingSession {
    pub fn new(
        model: &PolicyValueModel,
        ema_model: Option<&PolicyValueModel>,
        requested_devices: &[usize],
        learning_rate: f32,
    ) -> io::Result<Self> {
        let devices = make_devices(requested_devices)?;
        let replicas = devices
            .iter()
            .map(|(device, _)| Replica::new(model, device))
            .collect::<io::Result<Vec<_>>>()?;
        let ema = ema_model
            .map(|model| Replica::new(model, &devices[0].0))
            .transpose()?;
        let optimizer = AdamW::new(
            replicas[0].vars(),
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
            replicas,
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
                let shard_size = batch.len().div_ceil(self.replicas.len());
                let shards = batch.chunks(shard_size).collect::<Vec<_>>();
                let outputs = std::thread::scope(|scope| {
                    crate::scope_profile!("train.backward_parallel");
                    let handles = shards
                        .iter()
                        .enumerate()
                        .map(|(rank, shard)| {
                            let replica = &self.replicas[rank];
                            scope.spawn(move || {
                                let result = replica.backward(shard, batch.len());
                                crate::profile::flush_thread();
                                result
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|h| h.join().map_err(|_| io::Error::other("GPU 训练线程异常"))?)
                        .collect::<io::Result<Vec<_>>>()
                })?;
                let mut outputs = outputs.into_iter();
                let primary = outputs.next().expect("训练批次至少有一个分片");
                let mut primary_grads = primary.grads;
                stats.samples += primary.samples;
                stats.policy_loss += primary.policy_sum;
                stats.value_loss += primary.value_sum;
                for output in outputs {
                    crate::scope_profile!("train.grad_aggregate");
                    add_cpu_grads(
                        &self.replicas[0].vars(),
                        &mut primary_grads,
                        &output.cpu_grads,
                    )?;
                    stats.samples += output.samples;
                    stats.policy_loss += output.policy_sum;
                    stats.value_loss += output.value_sum;
                }
                {
                    crate::scope_profile!("train.optimizer_step");
                    self.optimizer.step(&primary_grads).map_err(err)?;
                }
                stats.optimizer_steps += 1;
                if let Some(ema) = &self.ema {
                    ema.update_ema_from(&self.replicas[0], ema_decay)?;
                }
                {
                    crate::scope_profile!("train.param_broadcast");
                    broadcast(&self.replicas)?;
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
        self.replicas[0].copy_to(model)?;
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
    fn backward(&self, samples: &[Sample], global_len: usize) -> io::Result<ShardOutput> {
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
            .and_then(|x| x.affine(1.0 / global_len as f64, 0.0))
            .map_err(err)?;
        let policy_sum = policy_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let value_sum = value_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let grads = {
            crate::scope_profile!("train.backward");
            loss.backward().map_err(err)?
        };
        let cpu_grads = self
            .vars()
            .iter()
            .map(|v| {
                grads
                    .get(v)
                    .map(|g| {
                        g.flatten_all()
                            .and_then(|x| x.to_device(&Device::Cpu))
                            .and_then(|x| x.to_vec1::<f32>())
                            .map_err(err)
                    })
                    .transpose()
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(ShardOutput {
            grads,
            cpu_grads,
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
    fn set_values(&self, values: &[Vec<f32>]) -> io::Result<()> {
        for (var, data) in self.vars().iter().zip(values) {
            var.set(
                &Tensor::from_vec(data.clone(), var.shape().clone(), &self.device).map_err(err)?,
            )
            .map_err(err)?
        }
        Ok(())
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
struct ShardOutput {
    grads: GradStore,
    cpu_grads: Vec<Option<Vec<f32>>>,
    samples: usize,
    policy_sum: f32,
    value_sum: f32,
}
fn add_cpu_grads(vars: &[Var], grads: &mut GradStore, cpu: &[Option<Vec<f32>>]) -> io::Result<()> {
    for (var, data) in vars.iter().zip(cpu) {
        if let Some(data) = data {
            let worker =
                Tensor::from_vec(data.clone(), var.shape().clone(), var.device()).map_err(err)?;
            let total = if let Some(current) = grads.get(var) {
                current.add(&worker).map_err(err)?
            } else {
                worker
            };
            grads.insert(var, total);
        }
    }
    Ok(())
}
fn broadcast(replicas: &[Replica]) -> io::Result<()> {
    if replicas.len() < 2 {
        return Ok(());
    }
    let values = replicas[0].cpu_values()?;
    for r in &replicas[1..] {
        r.set_values(&values)?
    }
    Ok(())
}

fn make_devices(requested: &[usize]) -> io::Result<Vec<(Device, String)>> {
    if requested.len() > 1 {
        let unique = requested
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if unique.len() != requested.len() {
            return Err(io::Error::other("gpu_devices 中不能重复指定同一设备"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        let ids = if requested.is_empty() {
            vec![0]
        } else {
            requested.to_vec()
        };
        return ids
            .into_iter()
            .map(|i| {
                Device::new_metal(i)
                    .map(|d| (d, format!("metal:{i}")))
                    .map_err(err)
            })
            .collect();
    }
    #[cfg(all(target_os = "linux", not(target_env = "musl")))]
    {
        let ids = if requested.is_empty() {
            auto_cuda_ids()
        } else {
            requested.to_vec()
        };
        if !ids.is_empty() {
            return ids
                .into_iter()
                .map(|i| {
                    Device::new_cuda(i)
                        .map(|d| (d, format!("cuda:{i}")))
                        .map_err(err)
                })
                .collect();
        }
    }
    #[allow(unreachable_code)]
    Ok(vec![(Device::Cpu, "cpu".into())])
}
#[cfg(all(target_os = "linux", not(target_env = "musl")))]
fn auto_cuda_ids() -> Vec<usize> {
    if let Ok(v) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let n = v.split(',').filter(|x| !x.trim().is_empty()).count();
        if n > 0 {
            return (0..n).collect();
        }
    }
    Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            (0..String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.trim_start().starts_with("GPU "))
                .count())
                .collect()
        })
        .unwrap_or_default()
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
        for m in s.board.legal_moves() {
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
        let stats = train(&mut model, &[sample], 2, 1e-3, 1, &[]).unwrap();
        assert_eq!(stats.optimizer_steps, 2);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(model.policy_local.iter().any(|&weight| weight != 0.0));
        assert_ne!(model.local_axis_embedding, before_local);
    }
}
