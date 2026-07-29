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

日常开发和训练部署可使用与 ChineseAI 相同的快速优化编译模式：

```bash
cargo build --profile fast
./target/fast/gomoku az-loop
# 或直接运行
cargo run --profile fast -- az-loop
```

`fast` 继承 Release 优化，但使用 Thin LTO、16 个 codegen unit 和增量编译，明显缩短
重复编译时间；最终性能测量或正式长期训练仍可使用 `--release` 的 Fat LTO 单元构建。

需要定位搜索和训练热点时，启用与 ChineseAI 相同的轻量性能分析 feature：

```bash
cargo run --profile fast --features profile -- az-bench model.safetensors 400 50
cargo run --profile fast --features profile -- az-train-bench model.safetensors data/replay.jsonl
```

程序结束时会按总耗时排序输出调用次数、总耗时、平均耗时和最大耗时。正常长期训练不要
启用 `profile`，避免计时器带来额外开销。

人工评估默认在终端原位刷新，不会重复堆叠棋盘。对局中输入 `info` 查看 Best 上次
搜索的候选着，输入 `help` 查看命令，输入 `quit` 退出；不支持 ANSI 的控制台可增加
`--no-clear`。

运行机制与 ChineseAI 的 AlphaZero 主循环保持一致：首次执行 `az-loop` 自动生成
`gomoku.azloop.toml`，检查参数后再次执行即可持续训练。进度记录在
`data/azloop-progress.json`，中断后会从绝对更新编号继续。

每次更新依次执行并行自博弈、回放池裁剪、策略价值训练、模型与进度保存。
配置的更新间隔到达后保存轮转检查点，并让候选模型与当前最佳模型交换先后手进行
竞技场比赛；达到晋级分数才覆盖 `best.safetensors`。按 Ctrl+C 会在安全边界保存并退出。
周期性 Arena 默认先生成 2 ply 随机开局，再由候选 EMA 与 Best 接管；相邻两盘复用
同一开局并交换黑白。自博弈 Actor 始终持续获取最新 EMA 网络；Arena 只维护评估
基准，候选得分率达到 `arena_promotion_rate` 时才覆盖 `best.safetensors`，不会阻塞
最新训练网络发布。
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

经验池采用与 ChineseAI 同类的固定训练量混合采样：每次更新不再遍历整个经验池，
而是默认有放回采样 50,000 条，其中 40% 强制来自最近 5 个模型版本，其余从完整
500,000 条窗口均匀采样，并在训练前执行确定性洗牌。控制台和 TensorBoard 分别报告
近期配额和实际近期样本占比。队列饱和时 Actor 使用非阻塞投递并丢弃结果；模型发布
后 Collector 与 Trainer 会丢弃旧版本数据，避免 Worker 背压和旧数据持续排队。

探索超参数与 ChineseAI 保持同一套语义：网络策略 logits 在 MCTS 前按
`policy_softmax_temp` 软化；自博弈根节点混合 Dirichlet 噪声；落子按访问次数和
逐手退火温度采样，并支持胜率差过滤与访问次数偏移。五子棋默认退火区间缩短为
12+24 手，适配五子棋较短的有效对局长度；竞技场、人工评估和普通搜索自动关闭
根噪声与随机采样。

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
价值塔同时连接 `96 → 1` 剩余手数辅助头，以“距终局手数 / 225”为目标进行 MSE
训练，默认损失权重为 0.1。在线模型执行梯度更新，EMA 模型默认按每个优化器 step
折算 `ema_decay = 0.999`，并专门提供给自博弈 Actor、检查点和 Arena；Best 仍只由
Arena 晋级替换。
输入侧将精确棋子格点嵌入与己/彼棋子类型、15 行和 15 列结构嵌入相加；CPU
推理使用一次稀疏棋盘遍历完成全部累加，训练侧使用等价的 Candle 批量矩阵运算。
MCTS 节点还缓存黑白双视角的增量累加器，扩展子节点时只加入新落子和手数特征，
避免每次叶子求值重新扫描整盘。

模型格式现为 v5，价值塔采用适合 NEON/AVX2/FMA 点积的输出优先布局。现有 v4 模型
加载时会自动转置迁移，并在下次保存时写成 v5，无需重新训练；v3 及更早模型仍不能
直接加载。
