use crate::{
    game::{CELL_COUNT, Move},
    model::PolicyValueModel,
    replay::Sample,
    selfplay::TrainStats,
    sparse_transformer::{FF_WIDTH, HEAD_WIDTH, HEADS, TOKEN_WIDTH, cells_aligned},
};
use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use candle_nn::{
    ops::{log_softmax, softmax},
    optim::{AdamW, Optimizer, ParamsAdamW},
};
use std::{
    io,
    sync::atomic::{AtomicBool, Ordering},
};

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
        Self::on_device(model, ema_model, &device, learning_rate)
    }

    fn on_device(
        model: &PolicyValueModel,
        ema_model: Option<&PolicyValueModel>,
        device: &Device,
        learning_rate: f32,
    ) -> io::Result<Self> {
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
                stats.samples += batch.len();
                stats.policy_loss += output.policy_sum;
                stats.value_loss += output.value_sum;
                self.optimizer.step(&output.grads).map_err(err)?;
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
        if let (Some(ema), Some(model)) = (&self.ema, ema_model) {
            ema.copy_to(model)?;
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

struct TrainBlock {
    q: Var,
    k: Var,
    v: Var,
    output: Var,
    ff_up: Var,
    ff_down: Var,
}

struct Replica {
    device: Device,
    stone_embedding: Var,
    position_embedding: Var,
    blocks: Vec<TrainBlock>,
    policy_query: Var,
    policy_key: Var,
    policy_value: Var,
    policy_output: Var,
    policy_bias: Var,
    value_hidden: Var,
    value_output: Var,
}

impl Replica {
    fn new(model: &PolicyValueModel, device: &Device) -> io::Result<Self> {
        let blocks = model
            .blocks
            .iter()
            .map(|block| {
                Ok(TrainBlock {
                    q: var(&block.q, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
                    k: var(&block.k, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
                    v: var(&block.v, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
                    output: var(&block.output, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
                    ff_up: var(&block.ff_up, (TOKEN_WIDTH, FF_WIDTH), device)?,
                    ff_down: var(&block.ff_down, (FF_WIDTH, TOKEN_WIDTH), device)?,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            stone_embedding: var(&model.stone_embedding, (2, TOKEN_WIDTH), device)?,
            position_embedding: var(&model.position_embedding, (CELL_COUNT, TOKEN_WIDTH), device)?,
            blocks,
            policy_query: var(&model.policy_query, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
            policy_key: var(&model.policy_key, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
            policy_value: var(&model.policy_value, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
            policy_output: var(&model.policy_output, (TOKEN_WIDTH, 1), device)?,
            policy_bias: var(&model.policy_bias, (CELL_COUNT,), device)?,
            value_hidden: var(&model.value_hidden, (TOKEN_WIDTH, TOKEN_WIDTH), device)?,
            value_output: var(&model.value_output, (TOKEN_WIDTH, 3), device)?,
        })
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = vec![
            self.stone_embedding.clone(),
            self.position_embedding.clone(),
        ];
        for block in &self.blocks {
            vars.extend([
                block.q.clone(),
                block.k.clone(),
                block.v.clone(),
                block.output.clone(),
                block.ff_up.clone(),
                block.ff_down.clone(),
            ]);
        }
        vars.extend([
            self.policy_query.clone(),
            self.policy_key.clone(),
            self.policy_value.clone(),
            self.policy_output.clone(),
            self.policy_bias.clone(),
            self.value_hidden.clone(),
            self.value_output.clone(),
        ]);
        vars
    }

    fn backward(&self, samples: &[Sample]) -> io::Result<BatchOutput> {
        let mut policy_losses = Vec::with_capacity(samples.len());
        let mut value_losses = Vec::with_capacity(samples.len());
        for sample in samples {
            let (policy_loss, value_loss) = self.forward_sample(sample)?;
            policy_losses.push(policy_loss);
            value_losses.push(value_loss);
        }
        let policy_sum = Tensor::stack(&policy_losses, 0)
            .and_then(|tensor| tensor.sum_all())
            .map_err(err)?;
        let value_sum = Tensor::stack(&value_losses, 0)
            .and_then(|tensor| tensor.sum_all())
            .map_err(err)?;
        let loss = policy_sum
            .add(&value_sum)
            .and_then(|tensor| tensor.affine(1.0 / samples.len() as f64, 0.0))
            .map_err(err)?;
        let grads = loss.backward().map_err(err)?;
        Ok(BatchOutput {
            grads,
            policy_sum: policy_sum.to_scalar::<f32>().map_err(err)?,
            value_sum: value_sum.to_scalar::<f32>().map_err(err)?,
        })
    }

    fn forward_sample(&self, sample: &Sample) -> io::Result<(Tensor, Tensor)> {
        let board = &sample.board;
        let us = board.to_move().stone();
        let moves = board
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, &stone)| (stone != 0).then_some(Move(index)))
            .collect::<Vec<_>>();
        let positions = moves.iter().map(|mv| mv.0 as u32).collect::<Vec<_>>();
        let sides = moves
            .iter()
            .map(|mv| u32::from(board.cells()[mv.0] != us))
            .collect::<Vec<_>>();
        let mut tokens = if moves.is_empty() {
            Tensor::zeros((0, TOKEN_WIDTH), DType::F32, &self.device).map_err(err)?
        } else {
            let positions = Tensor::from_vec(positions, moves.len(), &self.device).map_err(err)?;
            let sides = Tensor::from_vec(sides, moves.len(), &self.device).map_err(err)?;
            self.position_embedding
                .as_tensor()
                .index_select(&positions, 0)
                .and_then(|tensor| {
                    tensor.add(&self.stone_embedding.as_tensor().index_select(&sides, 0)?)
                })
                .map_err(err)?
        };
        if !moves.is_empty() {
            let mask = self_attention_mask(&moves);
            let mask =
                Tensor::from_vec(mask, (moves.len(), moves.len()), &self.device).map_err(err)?;
            for block in &self.blocks {
                tokens = transformer_block(&tokens, &mask, block)?;
            }
        }

        let legal = board.search_candidates();
        let legal_indices = Tensor::from_vec(
            legal.iter().map(|mv| mv.0 as u32).collect::<Vec<_>>(),
            legal.len(),
            &self.device,
        )
        .map_err(err)?;
        let positions = self
            .position_embedding
            .as_tensor()
            .index_select(&legal_indices, 0)
            .map_err(err)?;
        let context = if moves.is_empty() {
            Tensor::zeros((legal.len(), TOKEN_WIDTH), DType::F32, &self.device).map_err(err)?
        } else {
            let (cross_mask, query_valid) = cross_attention_mask(&legal, &moves);
            let cross_mask = Tensor::from_vec(cross_mask, (legal.len(), moves.len()), &self.device)
                .map_err(err)?;
            let query_valid =
                Tensor::from_vec(query_valid, (legal.len(), 1), &self.device).map_err(err)?;
            cross_attention(
                &positions,
                &tokens,
                &cross_mask,
                &query_valid,
                &self.policy_query,
                &self.policy_key,
                &self.policy_value,
            )?
        };
        let logits = context
            .add(&positions)
            .and_then(|tensor| tensor.matmul(self.policy_output.as_tensor()))
            .and_then(|tensor| tensor.reshape(legal.len()))
            .and_then(|tensor| {
                tensor.add(
                    &self
                        .policy_bias
                        .as_tensor()
                        .index_select(&legal_indices, 0)?,
                )
            })
            .map_err(err)?;
        let targets = normalized_policy_target(&legal, &sample.policy);
        let targets = Tensor::from_vec(targets, legal.len(), &self.device).map_err(err)?;
        let policy_loss = log_softmax(&logits, 0)
            .and_then(|log_probs| targets.mul(&log_probs))
            .and_then(|tensor| tensor.sum_all())
            .and_then(|tensor| tensor.affine(-1.0, 0.0))
            .map_err(err)?;

        let value_logits = if moves.is_empty() {
            Tensor::zeros(3, DType::F32, &self.device).map_err(err)?
        } else {
            tokens
                .mean(0)
                .and_then(|tensor| tensor.reshape((1, TOKEN_WIDTH)))
                .and_then(|tensor| tensor.matmul(self.value_hidden.as_tensor()))
                .and_then(|tensor| tensor.relu())
                .and_then(|tensor| tensor.matmul(self.value_output.as_tensor()))
                .and_then(|tensor| tensor.reshape(3))
                .map_err(err)?
        };
        let target_wdl = if sample.value > 0.5 {
            vec![1.0_f32, 0.0, 0.0]
        } else if sample.value < -0.5 {
            vec![0.0_f32, 0.0, 1.0]
        } else {
            vec![0.0_f32, 1.0, 0.0]
        };
        let target_wdl = Tensor::from_vec(target_wdl, 3, &self.device).map_err(err)?;
        let value_loss = log_softmax(&value_logits, 0)
            .and_then(|log_probs| target_wdl.mul(&log_probs))
            .and_then(|tensor| tensor.sum_all())
            .and_then(|tensor| tensor.affine(-1.0, 0.0))
            .map_err(err)?;
        Ok((policy_loss, value_loss))
    }

    fn update_ema_from(&self, source: &Self, decay: f32) -> io::Result<()> {
        let decay = decay.clamp(0.0, 1.0) as f64;
        for (ema, online) in self.vars().iter().zip(source.vars()) {
            let blended = ema
                .as_tensor()
                .affine(decay, 0.0)
                .and_then(|tensor| tensor.add(&online.as_tensor().affine(1.0 - decay, 0.0)?))
                .map_err(err)?;
            ema.set(&blended).map_err(err)?;
        }
        Ok(())
    }

    fn copy_to(&self, model: &mut PolicyValueModel) -> io::Result<()> {
        let values = self
            .vars()
            .iter()
            .map(cpu_values)
            .collect::<io::Result<Vec<_>>>()?;
        let mut index = 0;
        model.stone_embedding = values[index].clone();
        index += 1;
        model.position_embedding = values[index].clone();
        index += 1;
        for block in &mut model.blocks {
            block.q = values[index].clone();
            block.k = values[index + 1].clone();
            block.v = values[index + 2].clone();
            block.output = values[index + 3].clone();
            block.ff_up = values[index + 4].clone();
            block.ff_down = values[index + 5].clone();
            index += 6;
        }
        model.policy_query = values[index].clone();
        model.policy_key = values[index + 1].clone();
        model.policy_value = values[index + 2].clone();
        model.policy_output = values[index + 3].clone();
        model.policy_bias = values[index + 4].clone();
        model.value_hidden = values[index + 5].clone();
        model.value_output = values[index + 6].clone();
        model.refresh_runtime_caches();
        Ok(())
    }
}

fn transformer_block(tokens: &Tensor, mask: &Tensor, block: &TrainBlock) -> io::Result<Tensor> {
    let normalized = rms_norm(tokens)?;
    let q = heads(&normalized.matmul(block.q.as_tensor()).map_err(err)?)?;
    let k = heads(&normalized.matmul(block.k.as_tensor()).map_err(err)?)?;
    let v = heads(&normalized.matmul(block.v.as_tensor()).map_err(err)?)?;
    let k_t = k
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .map_err(err)?;
    let scores = q
        .matmul(&k_t)
        .and_then(|tensor| tensor.affine((HEAD_WIDTH as f64).sqrt().recip(), 0.0))
        .and_then(|tensor| tensor.broadcast_add(mask))
        .map_err(err)?;
    let context = softmax(&scores, 2)
        .and_then(|attention| attention.matmul(&v))
        .and_then(|tensor| tensor.transpose(0, 1))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.reshape((tokens.dim(0)?, TOKEN_WIDTH)))
        .map_err(err)?;
    let residual = tokens
        .add(&context.matmul(block.output.as_tensor()).map_err(err)?)
        .map_err(err)?;
    let ff = rms_norm(&residual)?
        .matmul(block.ff_up.as_tensor())
        .and_then(|tensor| tensor.relu())
        .and_then(|tensor| tensor.matmul(block.ff_down.as_tensor()))
        .map_err(err)?;
    residual.add(&ff).map_err(err)
}

fn cross_attention(
    positions: &Tensor,
    tokens: &Tensor,
    mask: &Tensor,
    query_valid: &Tensor,
    query: &Var,
    key: &Var,
    value: &Var,
) -> io::Result<Tensor> {
    let q = heads(&positions.matmul(query.as_tensor()).map_err(err)?)?;
    let k = heads(&tokens.matmul(key.as_tensor()).map_err(err)?)?;
    let v = heads(&tokens.matmul(value.as_tensor()).map_err(err)?)?;
    let k_t = k
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .map_err(err)?;
    let scores = q
        .matmul(&k_t)
        .and_then(|tensor| tensor.affine((HEAD_WIDTH as f64).sqrt().recip(), 0.0))
        .and_then(|tensor| tensor.broadcast_add(mask))
        .map_err(err)?;
    softmax(&scores, 2)
        .and_then(|attention| attention.broadcast_mul(query_valid))
        .and_then(|attention| attention.matmul(&v))
        .and_then(|tensor| tensor.transpose(0, 1))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.reshape((positions.dim(0)?, TOKEN_WIDTH)))
        .map_err(err)
}

fn heads(tensor: &Tensor) -> io::Result<Tensor> {
    tensor
        .reshape((tensor.dim(0).map_err(err)?, HEADS, HEAD_WIDTH))
        .and_then(|tensor| tensor.transpose(0, 1))
        .and_then(|tensor| tensor.contiguous())
        .map_err(err)
}

fn rms_norm(tensor: &Tensor) -> io::Result<Tensor> {
    let rms = tensor
        .sqr()
        .and_then(|tensor| tensor.mean_keepdim(1))
        .and_then(|tensor| tensor.affine(1.0, 1e-6))
        .and_then(|tensor| tensor.sqrt())
        .map_err(err)?;
    tensor.broadcast_div(&rms).map_err(err)
}

fn self_attention_mask(moves: &[Move]) -> Vec<f32> {
    moves
        .iter()
        .flat_map(|&left| {
            moves.iter().map(move |&right| {
                if cells_aligned(left, right) {
                    0.0
                } else {
                    -1e9
                }
            })
        })
        .collect()
}

fn cross_attention_mask(legal: &[Move], stones: &[Move]) -> (Vec<f32>, Vec<f32>) {
    let mut mask = Vec::with_capacity(legal.len() * stones.len());
    let mut valid = Vec::with_capacity(legal.len());
    for &candidate in legal {
        let mut any = false;
        for &stone in stones {
            let aligned = cells_aligned(candidate, stone);
            any |= aligned;
            mask.push(if aligned { 0.0 } else { -1e9 });
        }
        valid.push(f32::from(any));
    }
    (mask, valid)
}

fn normalized_policy_target(legal: &[Move], policy: &[(Move, f32)]) -> Vec<f32> {
    let sum = policy
        .iter()
        .map(|(_, probability)| probability.max(0.0))
        .sum::<f32>()
        .max(1e-12);
    legal
        .iter()
        .map(|mv| {
            policy
                .iter()
                .find_map(|(target, probability)| (*target == *mv).then_some(probability.max(0.0)))
                .unwrap_or(0.0)
                / sum
        })
        .collect()
}

fn var(data: &[f32], shape: impl Into<candle_core::Shape>, device: &Device) -> io::Result<Var> {
    Var::from_slice(data, shape, device).map_err(err)
}

fn cpu_values(var: &Var) -> io::Result<Vec<f32>> {
    var.as_tensor()
        .flatten_all()
        .and_then(|tensor| tensor.to_device(&Device::Cpu))
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(err)
}

fn make_device(requested: usize) -> io::Result<(Device, String)> {
    #[cfg(target_os = "macos")]
    {
        return Device::new_metal(requested)
            .map(|device| (device, format!("metal:{requested}")))
            .map_err(err);
    }
    #[cfg(all(target_os = "linux", not(target_env = "musl")))]
    {
        return Device::new_cuda(requested)
            .map(|device| (device, format!("cuda:{requested}")))
            .map_err(err);
    }
    #[allow(unreachable_code)]
    {
        if requested != 0 {
            return Err(io::Error::other("当前平台仅支持 gpu_device = 0（CPU）"));
        }
        Ok((Device::Cpu, "cpu".into()))
    }
}

struct BatchOutput {
    grads: GradStore,
    policy_sum: f32,
    value_sum: f32,
}

fn err(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Board;

    #[test]
    fn gpu_graph_updates_transformer_body_and_heads() {
        let mut model = PolicyValueModel::random(TOKEN_WIDTH, 9);
        let before_q = model.blocks[0].q.clone();
        let before_k = model.blocks[0].k.clone();
        let before_ff = model.blocks[0].ff_up.clone();
        let before_position = model.position_embedding.clone();
        let before_policy = model.policy_output.clone();
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        assert!(board.play(Move::new(7, 8).unwrap()));
        let sample = Sample {
            board,
            policy: vec![(Move::new(8, 7).unwrap(), 1.0)],
            value: 1.0,
            generation: 0,
        };
        #[cfg(target_os = "macos")]
        let stats = train(&mut model, &[sample], 1, 1e-3, 1, 0).unwrap();
        #[cfg(not(target_os = "macos"))]
        let stats = TrainingSession::on_device(&model, None, &Device::Cpu, 1e-3)
            .and_then(|mut session| {
                session.train_controlled(&mut model, None, &[sample], 1, 1e-3, 1, 1.0, None)
            })
            .unwrap();
        assert_eq!(stats.optimizer_steps, 1);
        assert!(stats.loss.is_finite());
        assert_ne!(model.blocks[0].q, before_q);
        assert_ne!(model.blocks[0].k, before_k);
        assert_ne!(model.blocks[0].ff_up, before_ff);
        assert_ne!(model.position_embedding, before_position);
        assert_ne!(model.policy_output, before_policy);
    }

    #[test]
    fn training_policy_matches_cpu_for_query_without_attention_edges() {
        let model = PolicyValueModel::random(TOKEN_WIDTH, 19);
        let mut board = Board::new();
        assert!(board.play(Move::new(7, 7).unwrap()));
        let target = Move::new(0, 1).unwrap();
        let probability = model
            .evaluate(&board)
            .0
            .into_iter()
            .find_map(|(mv, probability)| (mv == target).then_some(probability))
            .unwrap();
        let sample = Sample {
            board,
            policy: vec![(target, 1.0)],
            value: 0.0,
            generation: 0,
        };
        let replica = Replica::new(&model, &Device::Cpu).unwrap();
        let (policy_loss, _) = replica.forward_sample(&sample).unwrap();
        let policy_loss = policy_loss.to_scalar::<f32>().unwrap();
        assert!((policy_loss + probability.ln()).abs() < 1e-5);
    }
}
