# GomokuAI

基于 AlphaZero 方法实现的五子棋自博弈、训练、评估与协议引擎系统。

当前规则为 15×15 自由五子棋：黑先，任一方横、竖或斜线连续五子（含长连）获胜，不含 Renju 禁手。

## 快速开始

```bash
cargo test
cargo run -- az-init model.safetensors 192
cargo run -- az-loop                 # 首次生成 gomoku.azloop.toml
cargo run -- az-loop --target-update 10
cargo run -- az-search model.safetensors 3000 1.5 h8 h9 i8
cargo run -- az-bench model.safetensors 3000 20
cargo run -- az-train-bench
cargo run -- az-eval-best best.safetensors 3000 --human-side black
cargo run -- az-arena-best model.safetensors best.safetensors 100 3000
cargo run --profile fast --bin pbrain-gomoku
cargo run -- play
```

独立的 `pbrain-gomoku` 可执行文件通过纯标准输入/输出实现 Gomocup/Piskvork 协议。Piskvork
以必需的 `pbrain-` 文件名前缀识别标准输入/输出新协议。引擎支持 `START`、
`BEGIN`、`TURN`、`BOARD`、`INFO timeout_turn`、`TAKEBACK`、`RESTART`、`ABOUT`
和 `END`。协议坐标为零起始的 `x,y`；正常对局只按 `timeout_turn` 与 `time_left`
确定搜索截止时间，不再设置固定模拟次数上限。`timeout_turn = 0` 按协议要求尽快
落子。默认优先加载 `model.safetensors`，不存在时自动加载同目录的
`best.safetensors`。
当前规则为 15×15 freestyle Gomoku，不支持 Renju 禁手及非 15×15 棋盘。

交付引擎时只需编译并复制独立二进制与模型，不包含训练命令入口：

```bash
cargo build --profile fast --bin pbrain-gomoku
cp target/fast/pbrain-gomoku ./dist/
cp model.safetensors ./dist/       # 优先加载
# 或复制 best.safetensors，model 不存在时自动回退
```

GUI 中的引擎命令设置为：

```bash
./pbrain-gomoku
```

日常开发和训练部署可使用快速优化编译模式：

```bash
cargo build --profile fast
./target/fast/gomoku az-loop
# 或直接运行
cargo run --profile fast -- az-loop
```

`fast` 继承 Release 优化，但使用 Thin LTO、16 个 codegen unit 和增量编译，明显缩短
重复编译时间；最终性能测量或正式长期训练仍可使用 `--release` 的 Fat LTO 单元构建。

需要定位搜索和训练热点时，可启用轻量性能分析 feature：

```bash
cargo run --profile fast --features profile -- az-bench model.safetensors 3000 50
cargo run --profile fast --features profile -- az-train-bench model.safetensors data/replay.jsonl
```

程序结束时会按总耗时排序输出调用次数、总耗时、平均耗时和最大耗时。正常长期训练不要
启用 `profile`，避免计时器带来额外开销。

人工评估默认在终端原位刷新，不会重复堆叠棋盘。对局中输入 `info` 查看 Best 上次
搜索的候选着，输入 `help` 查看命令，输入 `quit` 退出；不支持 ANSI 的控制台可增加
`--no-clear`。

AlphaZero 训练循环首次执行 `az-loop` 时会自动生成
`gomoku.azloop.toml`，检查参数后再次执行即可持续训练。进度记录在
`data/azloop-progress.json`，中断后会从绝对更新编号继续。

默认每个训练周期生成至少 50,000 个新局面，每步搜索 3,000 次；Collector 只按
`selfplay_samples_per_update` 判断是否触发训练，不再按对局数控制。短局时自动生成
更多对局，长局时减少对局，使每轮新增样本量与 50,000 条训练抽样量大致匹配，避免
对局长度改变训练强度。每次更新依次执行并行自博弈、回放池裁剪、策略价值训练、
模型与进度保存。
每盘自博弈以 `selfplay_random_opening_probability = 0.25` 的概率启用随机开局：先在
棋盘全局随机选择一个 3×3、4×4 或 5×5 子区域，由黑白在其中各随机落一手，再交给
MCTS。两手开局不写入训练样本；设为 `0.0` 可关闭。Arena 的成对开局逻辑保持独立。
冷启动默认先累计 `replay_warmup_samples = 100000` 个局面，再执行第一次训练；之后
按 `selfplay_samples_per_update = 50000` 触发更新。中断快照中已恢复的经验会计入
预热数量，且预热目标不能超过经验池容量。
配置的更新间隔到达后保存轮转检查点，并让候选模型与当前最佳模型交换先后手进行
竞技场比赛；候选得分达到晋级线且按成对开局计算的置信下界超过 50% 时，才覆盖
`best.safetensors`。按 Ctrl+C 会在安全边界保存并退出。
周期性 Arena 默认先生成 2 ply 随机开局，再由候选 EMA 与 Best 接管；相邻两盘复用
同一开局并交换黑白。自博弈 Actor 始终持续获取最新 EMA 网络；Arena 只维护评估
基准，候选得分率达到 `arena_promotion_rate` 且置信下界通过门槛时才覆盖
`best.safetensors`，不会阻塞
最新训练网络发布。
训练损失、学习率、自博弈速度、回放池大小和竞技场得分会写入 `runs/gomoku`，
可用 `tensorboard --logdir runs/gomoku` 查看。训练使用 Candle 自动微分和 AdamW，
优化器动量在同一次运行的各次 update 之间持续保留，EMA 在每个 optimizer step 后于
训练设备更新。训练固定使用单个设备，由配置项 `gpu_device = 0` 选择；macOS 使用对应
Metal 设备，Linux 使用对应 CUDA 设备（0 号 CUDA 不可用时回退 CPU）。在线模型、
AdamW 状态与 EMA 均位于同一设备，不再进行跨卡批次分片、梯度回传或参数广播。
批大小由配置文件控制。

自博弈使用常驻异步 Worker，配置值超过机器可用 CPU 核心数时会自动限制到核心数。
`selfplay_workers = 0` 表示使用可用 CPU 线程数，
`selfplay_queue_capacity = 0` 表示使用自动队列容量。Worker 在
每盘开始时获取模型快照；训练、检查点和竞技场运行期间继续生产。队列满时采用
非阻塞投递并丢弃已完成对局，避免 Actor 被消费者阻塞；丢弃总数写入 TensorBoard。

训练流水线分为常驻 Actor、独立 Collector 和独立 Trainer 三段。
默认 Actor 队列容量为 `max(workers × 8, 32)`；Collector 持续排空 Actor 结果并只
缓存一个完整训练批次，Trainer 独占回放池和 GPU 模型，主线程仅发布新模型、保存
检查点、运行竞技场和输出日志。

经验池采用固定训练量分层采样：每次更新不再遍历整个经验池，
而是默认有放回采样 50,000 条，其中 40% 来自最近 5 个模型版本，其余 60% 只从更早的
历史样本抽取，并在训练前执行确定性洗牌；历史区为空时才回退到全近期样本。控制台和
TensorBoard 分别报告近期配额和实际近期样本占比。队列饱和时 Actor 使用非阻塞投递并丢弃结果；模型发布
后 Collector 与 Trainer 会丢弃旧版本数据，避免 Worker 背压和旧数据持续排队。

探索过程使用可配置的温度与噪声参数：网络策略 logits 在 MCTS 前按
`policy_softmax_temp` 软化；自博弈根节点混合 Dirichlet 噪声；落子按访问次数和
逐手退火温度采样，并支持胜率差过滤与访问次数偏移。五子棋默认退火区间缩短为
12+24 手，适配五子棋较短的有效对局长度；竞技场、人工评估和普通搜索自动关闭
根噪声与随机采样。

控制台每次更新分行报告黑白胜负、平均手数、平均搜索模拟数、根策略熵、访问动作数、
策略 Top-1/Top-2 集中度、温度采样命中率、Worker 活跃数、模型版本滞后、生产吞吐、
回放池利用率，以及训练设备、损失和样本吞吐；相应核心指标也写入 TensorBoard。
TensorBoard 进一步记录学习率、优化步数、各损失、训练耗时与吞吐，自博弈胜率、
平均手数、策略质量，Actor 活跃率、队列积压、累计丢弃和版本延迟，经验池填充率与
采样组成，以及 Arena 的得分、置信下界、Elo、分颜色得分、耗时和晋级事件；每个
update 都主动刷新日志，便于实时查看。

## 目录

- `game.rs`：棋盘、坐标、胜负与合法着
- `mcts.rs`：PUCT 蒙特卡洛树搜索
- `model.rs`：可保存、可训练的策略价值模型
- `selfplay.rs`：并行自博弈与训练
- `replay.rs`：LZ4 压缩经验池中断快照与混合采样
- `az_loop_config.rs`：训练配置及默认值
- `az_loop.rs`：可续跑训练循环、检查点与竞技场晋级
- `main.rs`：`az-*` 训练、评估命令及人机对战入口
- `gomocup.rs`：Gomocup/Piskvork 协议状态机与限时搜索
- `bin/gomoku-engine.rs`：不含训练命令入口的独立交付引擎

当前模型采用面向高速增量推理的 AZ-NNUE 结构：默认 192 宽共享隐藏层、ReLU 后
RMSNorm，以及 `96 → 96 → WDL(3)` 价值头。MCTS 标量价值由
`P(win) - P(loss)` 得到；初始 WDL 输出层为零，避免随机模型产生虚假胜率。
在线模型执行梯度更新，EMA 模型默认按每个优化器 step 折算 `ema_decay = 0.999`，
并专门提供给自博弈 Actor、检查点和 Arena；Best 仍只由 Arena 晋级替换。
新训练没有 EMA 检查点时，第一次完整更新会先把在线模型完整复制到 EMA，后续才启用
指数平滑；恢复已有 `ema.safetensors` 时直接延续历史 EMA，不会重新覆盖。
输入侧将精确棋子格点嵌入与己/彼棋子类型、15 行、15 列及两个方向各 29 条对角线
结构嵌入相加；CPU 推理使用一次稀疏棋盘遍历完成全部累加，训练侧使用等价的
Candle 批量矩阵运算。经验池抽样时会随机应用正方形的 8 种旋转/镜像对称变换，
棋盘与策略标签同步变换，从而提高等价棋形的样本利用率而不增加推理开销。
Policy 和 Value 均只从可训练的棋盘结构编码中学习，不注入成五、阻五、开放三等
手工战术标签；Policy 的位置 bias 也从全零开始训练。每个候选点额外读取水平、垂直
和两条对角线共 4 条轴线，每条轴线同时包含前后两个射线各 4 格。空、己、敌、边界
四态形成有限的轴棋形类别，由四轴共享的可训练查表产生两个统计量；轴内左右射线按
无序对编码，因此镜像棋形严格共享表示。候选特征通过均值与最大值残差接入 Policy，
并再次跨候选池化后直接校准 WDL logits。局部 Policy 和 Value 输出均从零初始化，
不会在未训练时形成随机战术偏见。Value 只使用完整对局的最终胜、和、负作为 WDL
监督，不混入当前网络搜索产生的 Q 值。
MCTS 节点还缓存黑白双视角的增量累加器，扩展子节点时只加入新落子和手数特征，
避免每次叶子求值重新扫描整盘。
模型格式为 v10。`rule_legal_moves()` 只返回规则允许的全部空点；搜索和训练只调用
`search_candidates()`，使用显式半径 3 候选生成器。空棋盘从中心开始；非终局且棋盘
未满时若候选为空会立即触发断言暴露错误，不会静默扩大搜索空间。价值塔采用适合
NEON/AVX2/FMA 点积的输出优先布局。旧模型不再
迁移；升级后请清理旧模型、进度和经验池，再用 `az-init` 开始全新训练。

正常训练时经验池只保存在 Trainer 内存中，不再
每个更新周期重写 50 万条 JSONL；仅在 Ctrl+C 中断时原子写入 LZ4 压缩快照。下次
启动会加载该快照并立即删除已消费的快照文件，避免旧快照被重复加载；正常达到目标
更新退出时不会保留中断快照。

开始全新 v10 训练时，应先停止旧进程，再只删除该实验对应文件和旧配置：

```bash
rm -f model.safetensors ema.safetensors best.safetensors
rm -f data/azloop-progress.json data/replay.jsonl data/replay.jsonl.tmp
rm -f gomoku.azloop.toml
rm -rf checkpoints runs/gomoku
cargo run --profile fast -- az-init model.safetensors 192
```
