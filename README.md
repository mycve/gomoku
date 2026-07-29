# GomokuAI

从 ChineseAI 的“环境 → 搜索 → 自博弈 → 回放池 → 训练 → 检查点”框架思路派生的五子棋自博弈系统。

当前规则为 15×15 自由五子棋：黑先，任一方横、竖或斜线连续五子（含长连）获胜，不含 Renju 禁手。

## 快速开始

```bash
cargo test
cargo run -- az-init model.safetensors 192
cargo run -- az-loop                 # 首次生成 gomoku.azloop.toml
cargo run -- az-loop --target-update 10
cargo run -- az-search model.safetensors 400 1.5 h8 h9 i8
cargo run -- az-bench model.safetensors 400 20
cargo run -- az-train-bench
cargo run -- az-eval-best best.safetensors 800 --human-side black
cargo run -- az-arena-best model.safetensors best.safetensors 40 400
cargo run -- play
```

人工评估默认在终端原位刷新，不会重复堆叠棋盘。对局中输入 `info` 查看 Best 上次
搜索的候选着，输入 `help` 查看命令，输入 `quit` 退出；不支持 ANSI 的控制台可增加
`--no-clear`。

运行机制与 ChineseAI 的 AlphaZero 主循环保持一致：首次执行 `az-loop` 自动生成
`gomoku.azloop.toml`，检查参数后再次执行即可持续训练。进度记录在
`data/azloop-progress.json`，中断后会从绝对更新编号继续。

每次更新依次执行并行自博弈、回放池裁剪、策略价值训练、模型与进度保存。
配置的更新间隔到达后保存轮转检查点，并让候选模型与当前最佳模型交换先后手进行
竞技场比赛；达到晋级分数才覆盖 `best.safetensors`。按 Ctrl+C 会在安全边界保存并退出。
训练损失、学习率、自博弈速度、回放池大小和竞技场得分会写入 `runs/gomoku`，
可用 `tensorboard --logdir runs/gomoku` 查看。训练使用 Candle 自动微分和 AdamW，
macOS 优先使用 Metal，其他环境在加速设备不可用时回退 CPU；批大小由配置文件控制。
Linux 多 GPU 可在配置中设置 `gpu_devices = [0, 1, 2, 3]`；留空时自动读取
`CUDA_VISIBLE_DEVICES` 或 `nvidia-smi`。全局批次会在各卡间分片，梯度汇总到主卡
执行一次 AdamW 更新，再把参数广播到全部模型副本。

自博弈使用常驻异步 Worker。`selfplay_workers = 0` 表示使用可用 CPU 线程数，
`selfplay_queue_capacity = 0` 表示使用自动队列容量。Worker 在
每盘开始时获取模型快照；训练、检查点和竞技场运行期间继续生产，队列满时自动背压。

流水线与 ChineseAI 一致，分为常驻 Actor、独立 Collector 和独立 Trainer 三段。
默认 Actor 队列容量为 `max(workers × 8, 32)`；Collector 持续排空 Actor 结果并只
缓存一个完整训练批次，Trainer 独占回放池和 GPU 模型，主线程仅发布新模型、保存
检查点、运行竞技场和输出日志。

控制台每次更新分行报告黑白胜负、平均手数、平均搜索模拟数、根策略熵、访问动作数、
策略 Top-1/Top-2 集中度、温度采样命中率、Worker 活跃数、模型版本滞后、生产吞吐、
回放池利用率，以及训练设备、损失和样本吞吐；相应核心指标也写入 TensorBoard。

## 目录

- `game.rs`：棋盘、坐标、胜负与合法着
- `mcts.rs`：PUCT 蒙特卡洛树搜索
- `model.rs`：可保存、可训练的策略价值模型
- `selfplay.rs`：并行自博弈与训练
- `replay.rs`：JSONL 回放数据
- `az_loop_config.rs`：训练配置及默认值
- `az_loop.rs`：可续跑训练循环、检查点与竞技场晋级
- `main.rs`：与 ChineseAI 对齐的 `az-*` 命令及人机对战入口

当前模型沿用 ChineseAI 的 AZ-NNUE 方向：默认 192 宽共享隐藏层、ReLU 后
RMSNorm，以及 `96 → 96 → WDL(3)` 价值头。MCTS 标量价值由
`P(win) - P(loss)` 得到；初始 WDL 输出层为零，避免随机模型产生虚假胜率。
