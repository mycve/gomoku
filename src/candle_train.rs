use crate::{
    game::CELL_COUNT,
    model::{
        AXIS_FEATURES, DIAGONAL_FEATURES, INPUT_SIZE, LOCAL_AXES, LOCAL_AXIS_FEATURE_SIZE,
        LOCAL_AXIS_PATTERNS, LOCAL_CANDIDATE_SIZE, POLICY_HEAD_SIZE, PolicyValueModel, STONE_TYPES,
        VALUE_HEAD_SIZE, VALUE_LOCAL_SIZE, WDL_SIZE, local_ray_codes,
    },
    replay::Sample,
    selfplay::TrainStats,
};
#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
use candle_core::{
    CudaStorage, Storage,
    cuda_backend::cudarc::nccl::{result as nccl_result, safe as nccl},
    op::BackpropOp,
};
use candle_core::{Device, Tensor, Var, backprop::GradStore};
use candle_nn::{
    ops::log_softmax,
    optim::{AdamW, Optimizer, ParamsAdamW},
};
#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{io, thread};

pub fn training_device_name() -> io::Result<String> {
    let names = training_device_indices()
        .iter()
        .map(|&index| make_device(index).map(|(_, name)| name))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(names.join(","))
}

pub fn train(
    model: &mut PolicyValueModel,
    samples: &[Sample],
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
) -> io::Result<TrainStats> {
    let mut session = TrainingSession::new(model, None, learning_rate)?;
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
    #[cfg(any(
        feature = "nccl-train",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    nccl: Option<NcclAllReduce>,
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
struct NcclAllReduce {
    comms: Vec<nccl::Comm>,
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
unsafe impl Send for NcclAllReduce {}
#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
unsafe impl Sync for NcclAllReduce {}

impl TrainingSession {
    pub fn new(
        model: &PolicyValueModel,
        ema_model: Option<&PolicyValueModel>,
        learning_rate: f32,
    ) -> io::Result<Self> {
        let unique = training_device_indices();
        #[cfg(not(any(
            feature = "nccl-train",
            all(target_os = "linux", not(target_env = "musl"))
        )))]
        if unique.len() > 1 {
            return Err(io::Error::other(
                "多卡训练需要使用 --features nccl-train 编译",
            ));
        }
        let devices = unique
            .iter()
            .map(|&index| make_device(index).map(|(device, _)| device))
            .collect::<io::Result<Vec<_>>>()?;
        let replicas = devices
            .iter()
            .map(|device| Replica::new(model, device))
            .collect::<io::Result<Vec<_>>>()?;
        if replicas.len() > 1 {
            eprintln!(
                "train    : NCCL data parallel devices={:?} global_batch_sharded_each_step",
                unique
            );
        }
        let ema = ema_model
            .map(|model| Replica::new(model, &devices[0]))
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
        #[cfg(any(
            feature = "nccl-train",
            all(target_os = "linux", not(target_env = "musl"))
        ))]
        let nccl = init_nccl_all_reduce(&replicas)?;
        Ok(Self {
            replicas,
            ema,
            optimizer,
            #[cfg(any(
                feature = "nccl-train",
                all(target_os = "linux", not(target_env = "musl"))
            ))]
            nccl,
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
                let output = self.train_batch(batch)?;
                stats.samples += output.samples;
                stats.policy_loss += output.policy_sum;
                stats.value_loss += output.value_sum;
                stats.policy_entropy += output.policy_entropy_sum;
                stats.value_entropy += output.value_entropy_sum;
                stats.optimizer_steps += 1;
                if let Some(ema) = &self.ema {
                    ema.update_ema_from(&self.replicas[0], ema_decay)?;
                }
            }
        }
        self.copy_models(model, ema_model)?;
        Ok(finalize_stats(stats))
    }

    /// 计算样本损失但不更新参数。当前与训练共用完全相同的前向与标签处理。
    pub fn evaluate(&self, samples: &[Sample], batch_size: usize) -> io::Result<TrainStats> {
        if samples.is_empty() {
            return Ok(TrainStats::default());
        }
        let mut stats = TrainStats::default();
        for batch in samples.chunks(batch_size.max(1)) {
            let output = self.replicas[0].forward(batch, false, batch.len())?;
            stats.samples += output.samples;
            stats.policy_loss += output.policy_sum;
            stats.value_loss += output.value_sum;
            stats.policy_entropy += output.policy_entropy_sum;
            stats.value_entropy += output.value_entropy_sum;
        }
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

    fn train_batch(&mut self, batch: &[Sample]) -> io::Result<BatchOutput> {
        let active = self.replicas.len().min(batch.len());
        if active <= 1 {
            let output = self.replicas[0].forward(batch, true, batch.len())?;
            crate::scope_profile!("train.optimizer_step");
            self.optimizer
                .step(output.grads.as_ref().expect("训练批次必须包含梯度"))
                .map_err(err)?;
            #[cfg(any(
                feature = "nccl-train",
                all(target_os = "linux", not(target_env = "musl"))
            ))]
            if self.replicas.len() > 1 {
                self.nccl_broadcast_vars()?;
            }
            return Ok(output);
        }
        let shard_size = batch.len().div_ceil(active);
        #[allow(unused_mut)]
        let mut outputs = thread::scope(|scope| {
            let handles = batch
                .chunks(shard_size)
                .zip(&self.replicas)
                .map(|(shard, replica)| {
                    scope.spawn(move || replica.forward(shard, true, batch.len()))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| io::Error::other("多卡训练线程异常退出"))?
                })
                .collect::<io::Result<Vec<_>>>()
        })?;
        #[cfg(any(
            feature = "nccl-train",
            all(target_os = "linux", not(target_env = "musl"))
        ))]
        {
            self.nccl_all_reduce_grads(&mut outputs)?;
            crate::scope_profile!("train.optimizer_step");
            self.optimizer
                .step(outputs[0].grads.as_ref().expect("NCCL 后主卡梯度缺失"))
                .map_err(err)?;
            self.nccl_broadcast_vars()?;
        }
        let mut total = BatchOutput::default();
        for output in outputs {
            total.add_stats(output);
        }
        Ok(total)
    }

    #[cfg(any(
        feature = "nccl-train",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn nccl_all_reduce_grads(&self, outputs: &mut [BatchOutput]) -> io::Result<()> {
        let vars_by_rank = self.replicas.iter().map(Replica::vars).collect::<Vec<_>>();
        for var_index in 0..vars_by_rank[0].len() {
            let grads = outputs
                .iter()
                .zip(&vars_by_rank)
                .map(|(output, vars)| {
                    output
                        .grads
                        .as_ref()
                        .and_then(|store| store.get(&vars[var_index]))
                })
                .collect::<Vec<_>>();
            if grads.iter().all(|grad| grad.is_none()) {
                continue;
            }
            if grads.iter().any(|grad| grad.is_none()) {
                return Err(io::Error::other(format!(
                    "NCCL 参数 {var_index} 的梯度在部分 GPU 上缺失"
                )));
            }
            let tensors = grads
                .into_iter()
                .map(|grad| grad.expect("已检查梯度").contiguous().map_err(err))
                .collect::<io::Result<Vec<_>>>()?;
            let storages = tensors
                .iter()
                .map(|tensor| {
                    let (storage, layout) = tensor.storage_and_layout();
                    if !layout.is_contiguous() || layout.start_offset() != 0 {
                        return Err(io::Error::other("NCCL 梯度必须连续且偏移为零"));
                    }
                    Ok(storage)
                })
                .collect::<io::Result<Vec<_>>>()?;
            let mut receives = tensors
                .iter()
                .map(|tensor| {
                    tensor
                        .device()
                        .as_cuda_device()
                        .map_err(err)?
                        .cuda_stream()
                        .alloc_zeros::<f32>(tensor.elem_count())
                        .map_err(nccl_cuda_error)
                })
                .collect::<io::Result<Vec<_>>>()?;
            let nccl = self.nccl.as_ref().expect("多卡必须初始化 NCCL");
            nccl::group_start().map_err(nccl_error)?;
            let reduce = (|| -> io::Result<()> {
                for rank in 0..outputs.len() {
                    let send = match &*storages[rank] {
                        Storage::Cuda(storage) => storage.as_cuda_slice::<f32>().map_err(err)?,
                        _ => return Err(io::Error::other("NCCL 梯度不在 CUDA 设备上")),
                    };
                    nccl.comms[rank]
                        .all_reduce(send, &mut receives[rank], &nccl::ReduceOp::Sum)
                        .map_err(nccl_error)?;
                }
                Ok(())
            })();
            let group_end = nccl::group_end().map_err(nccl_error);
            reduce?;
            group_end?;
            for (rank, recv) in receives.into_iter().enumerate() {
                let var = &vars_by_rank[rank][var_index];
                let device = self.replicas[rank]
                    .device
                    .as_cuda_device()
                    .map_err(err)?
                    .clone();
                let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(recv, device));
                let reduced =
                    Tensor::from_storage(storage, var.shape().clone(), BackpropOp::none(), false);
                outputs[rank]
                    .grads
                    .as_mut()
                    .expect("训练输出必须有梯度")
                    .insert(var, reduced);
            }
        }
        Ok(())
    }

    #[cfg(any(
        feature = "nccl-train",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn nccl_broadcast_vars(&self) -> io::Result<()> {
        let vars_by_rank = self.replicas.iter().map(Replica::vars).collect::<Vec<_>>();
        let nccl = self.nccl.as_ref().expect("多卡必须初始化 NCCL");
        for var_index in 0..vars_by_rank[0].len() {
            let root = vars_by_rank[0][var_index]
                .as_detached_tensor()
                .contiguous()
                .map_err(err)?;
            let (root_storage, root_layout) = root.storage_and_layout();
            if !root_layout.is_contiguous() || root_layout.start_offset() != 0 {
                return Err(io::Error::other("NCCL 广播参数必须连续且偏移为零"));
            }
            let mut receives = vars_by_rank
                .iter()
                .map(|vars| {
                    vars[var_index]
                        .device()
                        .as_cuda_device()
                        .map_err(err)?
                        .cuda_stream()
                        .alloc_zeros::<f32>(root.elem_count())
                        .map_err(nccl_cuda_error)
                })
                .collect::<io::Result<Vec<_>>>()?;
            nccl::group_start().map_err(nccl_error)?;
            let broadcast = (|| -> io::Result<()> {
                let send = match &*root_storage {
                    Storage::Cuda(storage) => storage.as_cuda_slice::<f32>().map_err(err)?,
                    _ => return Err(io::Error::other("NCCL 广播源不在 CUDA 设备上")),
                };
                for rank in 0..vars_by_rank.len() {
                    nccl.comms[rank]
                        .broadcast((rank == 0).then_some(send), &mut receives[rank], 0)
                        .map_err(nccl_error)?;
                }
                Ok(())
            })();
            let group_end = nccl::group_end().map_err(nccl_error);
            broadcast?;
            group_end?;
            for (rank, recv) in receives.into_iter().enumerate().skip(1) {
                let var = &vars_by_rank[rank][var_index];
                let device = self.replicas[rank]
                    .device
                    .as_cuda_device()
                    .map_err(err)?
                    .clone();
                let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(recv, device));
                let tensor =
                    Tensor::from_storage(storage, var.shape().clone(), BackpropOp::none(), false);
                var.set(&tensor).map_err(err)?;
            }
        }
        Ok(())
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
    policy_global: Var,
    policy_global_bias: Var,
    policy_gate: Var,
    policy_gate_bias: Var,
    policy_output: Var,
    policy_bias: Var,
    local_axis_embedding: Var,
    local_axis_scale: Var,
    local_axis_bias: Var,
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
            policy_global: var(&model.policy_global, (POLICY_HEAD_SIZE, h), device)?,
            policy_global_bias: var(&model.policy_global_bias, (POLICY_HEAD_SIZE,), device)?,
            policy_gate: var(&model.policy_gate, (POLICY_HEAD_SIZE,), device)?,
            policy_gate_bias: var(&model.policy_gate_bias, (1,), device)?,
            policy_output: var(&model.policy_output, (CELL_COUNT, POLICY_HEAD_SIZE), device)?,
            policy_bias: var(&model.policy_bias, (CELL_COUNT,), device)?,
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
            self.policy_global.clone(),
            self.policy_global_bias.clone(),
            self.policy_gate.clone(),
            self.policy_gate_bias.clone(),
            self.policy_output.clone(),
            self.policy_bias.clone(),
            self.local_axis_embedding.clone(),
            self.local_axis_scale.clone(),
            self.local_axis_bias.clone(),
            self.policy_local.clone(),
            self.value_head_hidden.clone(),
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
        let policy_weights =
            Tensor::from_vec(packed.policy_weights, (b,), &self.device).map_err(err)?;
        let value_weights =
            Tensor::from_vec(packed.value_weights, (b,), &self.device).map_err(err)?;
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
        let policy_global = hidden
            .matmul(&self.policy_global.t().map_err(err)?)
            .and_then(|x| x.broadcast_add(&self.policy_global_bias))
            .and_then(|x| x.relu())
            .map_err(err)?;
        let policy_gate = policy_global
            .matmul(
                &self
                    .policy_gate
                    .reshape((POLICY_HEAD_SIZE, 1))
                    .map_err(err)?,
            )
            .and_then(|x| x.broadcast_add(&self.policy_gate_bias))
            .map_err(err)?;
        let logits = policy_global
            .matmul(&self.policy_output.t().map_err(err)?)
            .and_then(|x| x.add(&local_policy_logits.broadcast_mul(&policy_gate)?))
            .and_then(|x| x.broadcast_add(&self.policy_bias))
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
            .and_then(|x| x.sum(1))
            .and_then(|x| x.mul(&value_weights))
            .and_then(|x| x.sum_all())
            .and_then(|x| x.affine(-1.0, 0.0))
            .map_err(err)?;
        let loss = policy_sum_tensor
            .add(&value_sum_tensor)
            .and_then(|x| x.affine(1.0 / global_batch_size.max(1) as f64, 0.0))
            .map_err(err)?;
        let policy_sum = policy_sum_tensor.to_scalar::<f32>().map_err(err)?;
        let value_sum = value_sum_tensor.to_scalar::<f32>().map_err(err)?;
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
        m.policy_global = v[7].clone();
        m.policy_global_bias = v[8].clone();
        m.policy_gate = v[9].clone();
        m.policy_gate_bias = v[10].clone();
        m.policy_output = v[11].clone();
        m.policy_bias = v[12].clone();
        m.local_axis_embedding = v[13].clone();
        m.local_axis_scale = v[14].clone();
        m.local_axis_bias = v[15].clone();
        m.policy_local = v[16].clone();
        m.value_head_hidden = v[17].clone();
        m.value_local_output = v[18].clone();
        m.value_head_bias = v[19].clone();
        m.value_head_hidden2 = v[20].clone();
        m.value_head_bias2 = v[21].clone();
        m.value_head_output = v[22].clone();
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
impl BatchOutput {
    fn add_stats(&mut self, other: Self) {
        self.samples += other.samples;
        self.policy_sum += other.policy_sum;
        self.value_sum += other.value_sum;
        self.policy_entropy_sum += other.policy_entropy_sum;
        self.value_entropy_sum += other.value_entropy_sum;
    }
}

fn training_device_indices() -> Vec<usize> {
    #[cfg(any(
        feature = "nccl-train",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    {
        let candidates = visible_cuda_device_count()
            .or_else(nvidia_smi_device_count)
            .map(|count| (0..count).collect::<Vec<_>>())
            .unwrap_or_else(|| probe_cuda_devices(64));
        let available = candidates
            .into_iter()
            .filter(|&index| Device::new_cuda(index).is_ok())
            .collect::<Vec<_>>();
        if !available.is_empty() {
            return available;
        }
    }
    vec![0]
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn visible_cuda_device_count() -> Option<usize> {
    let value = std::env::var("CUDA_VISIBLE_DEVICES").ok()?;
    let value = value.trim();
    if value.is_empty() || value == "-1" || value.eq_ignore_ascii_case("NoDevFiles") {
        return None;
    }
    let count = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .count();
    (count > 0).then_some(count)
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn nvidia_smi_device_count() -> Option<usize> {
    let output = Command::new("nvidia-smi").arg("-L").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let count = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.trim_start().starts_with("GPU "))
        .count();
    (count > 0).then_some(count)
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn probe_cuda_devices(limit: usize) -> Vec<usize> {
    let mut devices = Vec::new();
    for index in 0..limit {
        if Device::new_cuda(index).is_ok() {
            devices.push(index);
        } else if !devices.is_empty() {
            break;
        }
    }
    devices
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn init_nccl_all_reduce(replicas: &[Replica]) -> io::Result<Option<NcclAllReduce>> {
    if replicas.len() <= 1 {
        return Ok(None);
    }
    let streams = replicas
        .iter()
        .map(|replica| {
            replica
                .device
                .as_cuda_device()
                .map(|device| device.cuda_stream())
                .map_err(err)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let comms = nccl::Comm::from_devices(streams).map_err(nccl_error)?;
    Ok(Some(NcclAllReduce { comms }))
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn nccl_error(error: nccl_result::NcclError) -> io::Error {
    io::Error::other(format!("NCCL failed: {error:?}"))
}

#[cfg(any(
    feature = "nccl-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn nccl_cuda_error(error: candle_core::cuda_backend::cudarc::driver::DriverError) -> io::Error {
    io::Error::other(format!("NCCL CUDA allocation failed: {error:?}"))
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
    policy_weights: Vec<f32>,
    value_weights: Vec<f32>,
    policy_entropy_sum: f32,
    value_entropy_sum: f32,
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
    let mut policy_weights = Vec::with_capacity(samples.len());
    let mut value_weights = Vec::with_capacity(samples.len());
    let mut policy_entropy_sum = 0.0;
    let mut value_entropy_sum = 0.0;
    for (row, s) in samples.iter().enumerate() {
        policy_weights.push(s.policy_weight.max(0.0));
        value_weights.push(s.value_weight.max(0.0));
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
        let mut policy_entropy = 0.0;
        for &(m, p) in &s.policy {
            if m.0 < CELL_COUNT && sum > 1e-12 {
                let probability = p.max(0.0) / sum;
                targets[row * CELL_COUNT + m.0] = probability;
                if probability > 0.0 {
                    policy_entropy -= probability * probability.ln();
                }
            }
        }
        policy_entropy_sum += policy_entropy * s.policy_weight.max(0.0);
        let final_wdl = s.value_wdl.unwrap_or_else(|| {
            if s.value > 0.5 {
                [1.0, 0.0, 0.0]
            } else if s.value < -0.5 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            }
        });
        value_entropy_sum -= s.value_weight.max(0.0)
            * final_wdl
                .iter()
                .filter(|&&probability| probability > 0.0)
                .map(|&probability| probability * probability.ln())
                .sum::<f32>();
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
        policy_weights,
        value_weights,
        policy_entropy_sum,
        value_entropy_sum,
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
            value_wdl: None,
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
    fn packing_reports_soft_target_entropy() {
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let first = Move::new(7, 8).unwrap();
        let second = Move::new(8, 7).unwrap();
        let policy = [0.25_f32, 0.75];
        let wdl = [0.2_f32, 0.3, 0.5];
        let packed = pack(&[Sample {
            board,
            policy: vec![(first, policy[0]), (second, policy[1])],
            value: -0.3,
            value_wdl: Some(wdl),
            generation: 0,
            policy_weight: 1.0,
            value_weight: 1.0,
            policy_surprise: 0.0,
            value_surprise: 0.0,
            predicted_value: 0.0,
        }]);
        let expected_policy = -policy.iter().map(|p| p * p.ln()).sum::<f32>();
        let expected_value = -wdl.iter().map(|p| p * p.ln()).sum::<f32>();
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
            value_wdl: None,
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
        assert!(model.policy_local.iter().any(|&weight| weight != 0.0));
        assert_ne!(model.local_axis_embedding, before_local);
        let (policy, value) = model.evaluate(&Board::new());
        assert_eq!(policy.len(), CELL_COUNT);
        assert!(
            policy
                .iter()
                .all(|(_, probability)| probability.is_finite())
        );
        assert!(value.is_finite());
    }
}
