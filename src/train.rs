use crate::{
    model::{EvalScratch, PolicyValueModel},
    replay::Sample,
    selfplay::TrainStats,
};
use std::{
    io,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn training_device_name(requested: usize) -> io::Result<String> {
    if requested == 0 {
        Ok("cpu-sparse".into())
    } else {
        Err(io::Error::other(
            "CPU 稀疏 Transformer 训练仅支持 gpu_device = 0",
        ))
    }
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
    online: PolicyValueModel,
    ema: Option<PolicyValueModel>,
}

impl TrainingSession {
    pub fn new(
        model: &PolicyValueModel,
        ema_model: Option<&PolicyValueModel>,
        requested_device: usize,
        _learning_rate: f32,
    ) -> io::Result<Self> {
        training_device_name(requested_device)?;
        Ok(Self {
            online: model.clone(),
            ema: ema_model.cloned(),
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
        let mut stats = TrainStats::default();
        let mut scratch = EvalScratch::new();
        'epochs: for _ in 0..epochs {
            for batch in samples.chunks(batch_size.max(1)) {
                if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                    break 'epochs;
                }
                for sample in batch {
                    let (policy_loss, value_loss) = self.online.train_heads(
                        &sample.board,
                        &sample.policy,
                        sample.value,
                        learning_rate,
                        &mut scratch,
                    );
                    stats.samples += 1;
                    stats.policy_loss += policy_loss;
                    stats.value_loss += value_loss;
                }
                stats.optimizer_steps += 1;
                if let Some(ema) = &mut self.ema {
                    ema.update_ema(&self.online, ema_decay);
                }
            }
        }
        *model = self.online.clone();
        if let (Some(source), Some(target)) = (&self.ema, ema_model) {
            *target = source.clone();
        }
        let count = stats.samples.max(1) as f32;
        stats.policy_loss /= count;
        stats.value_loss /= count;
        stats.loss = stats.policy_loss + stats.value_loss;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Board, CELL_COUNT, Move};

    #[test]
    fn training_updates_sparse_output_heads() {
        let mut model = PolicyValueModel::random(32, 9);
        let before = model.evaluate(&Board::new()).0;
        let sample = Sample {
            board: Board::new(),
            policy: vec![(Move::new(7, 7).unwrap(), 1.0)],
            value: 1.0,
            generation: 0,
        };
        let stats = train(&mut model, &[sample], 2, 1e-3, 1, 0).unwrap();
        let after = model.evaluate(&Board::new()).0;
        assert_eq!(after.len(), CELL_COUNT);
        assert_eq!(stats.optimizer_steps, 2);
        assert!(stats.loss.is_finite());
        assert_ne!(before, after);
    }
}
