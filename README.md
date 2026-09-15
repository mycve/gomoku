# Go19

`codex/go19` 分支上的 19×19 围棋 AlphaZero 实验；9 路版本保留在 `codex/go`，原五子棋版本保留在 `main`。
支持自博弈、CUDA 训练、Arena 评估、终端对战和 GTP 2。

## 19 路改造

361 个交点加停着；棋块及气使用 384 位固定集合（6 个 u64），不再受 u128 限制。
输入按 361 点扩展，九个区域按比例划分为每轴 7/6/6 点，训练与推理共享同一映射。
模型格式 v32；旧 9 路模型与回放拒绝载入，不做兼容或迁移。
正式配置保持原有 400 模拟、128 actors 和训练超参数；19 路棋局、历史与输入更大，时间和内存开销需要重新评估。

## 19 路搜索优化

历史与超级劫索引采用持久化共享结构；候选哈希命中后仍核对精确盘面。搜索缓存终局结果，预合并模型落子权重，并优化位集合、棋块构建及输入投影。
本地 16 线程固定局面测试，800 次模拟下中位吞吐提升约 4.6 倍，进程峰值工作集减少约 70%；不代表服务器整条流水线的加速保证。
模型仍为 v32，现有 19 路模型与数据无需迁移或重新训练。验证方法和限制见 [性能报告](PERFORMANCE_GO19.md)。

## v32 价值目标

主价值头只有一个胜率 logit，使用稳定二元交叉熵训练；搜索使用 `v=2p-1`。
每个样本直接使用最终胜负（当前行棋方胜 +1、负 -1），取消 TD(λ)、WDL 三分类及短期自举辅助头。
默认 7.5 贴目下没有计分平局。自定义整数贴目若平分，期望结果记为 0（BCE 目标 0.5），不额外设置和棋输出。
中断对局仍不产生训练标签；终局死活估分的局限仍然存在。

**不兼容旧格式**：9 路模型和旧 TD 回放会被拒绝，既不自动迁移，也不覆盖。
v32 使用独立模型、回放和进度路径，需要重新开始训练；回放终局结果字段为必填 `mc_value`。

## 启动训练

```powershell
cargo test --profile fast
cargo run --profile fast -- az-loop   # 首次生成 go19-v32.azloop.toml
cargo run --profile fast -- az-loop --target-update 10
cargo run --profile fast -- play
```

默认配置：隐藏宽度 128，每步 400 次模拟，128 个自博弈线程，每次收集至少 81920 条样本并训练 81920 条，batch=256，回放容量 500000。
更新编号是绝对值，运行会读取已有进度。模型、Best、回放、日志分别使用
`go19-v32-model.safetensors`、`go19-v32-best.safetensors`、`data/go19-v32/`、`runs/go19-v32/`。

小规模端到端验证：

```powershell
cargo run --profile fast -- az-loop --config go19.smoke.toml --target-update 2
```

小配置使用独立的 `data/go19-smoke-v32/` 与 `runs/go19-smoke-v32/`，每次更新进行 4 局 Arena。
小配置用于验证 19 路自博弈、CUDA 训练、模型与回放格式以及恢复训练。
少量更新只能验证运行流程，不能证明棋力提升。

## 回放抽样

抽样先选择并打乱索引，再构建样本；对称增强共享原始历史，通过坐标映射保持精确超级劫规则，避免逐样本重建整局历史。随机结果、近期配额和模型/回放格式不变。
本地扩展池 246145 条抽取 81920 条约 0.164 秒；真实服务器负载下耗时需复测，详见 [性能报告](PERFORMANCE_GO19.md)。

## 训练打包预取

默认使用 4 个 CPU 打包线程，为 GPU 有序预取 batch；每个线程的发送队列容量为 1，另有至多一批正在打包/等待发送。
保持原样本顺序、batch 大小和优化器步数。环境变量 `GO19_TRAIN_PACK_WORKERS=0` 可关闭预取，1/2/4 用于对照测试。
此优化不改变模型或回放格式，也没有增加特征缓存。

此前 9 路固定模型测试：独立训练三次中位吞吐提高约 9%；128 个固定局面搜索线程施压时，单 epoch、81920 样本提高约 58%，搜索吞吐基本持平。这些数字不代表 19 路性能。
这是压力基准，不等于正式自博弈流水线加速比。大量小 epoch 下线程反复创建会影响收益，测试细节见 `PERFORMANCE.md`。

## 围棋规则与自动计分

- 19×19、黑先、默认白贴 7.5 目，提子、禁自杀、位置超级劫。
- 362 个动作：361 个交点和 `pass`。连续两次停一手结束。
- Benson 算法证明无条件活棋；只在确定活棋围成的至多 8 点区域内做战术搜索，
  自动移除即使防守方先走也无法救出的死子。每个棋块最多 4000 个搜索节点；预算耗尽为未定。
- 能识别两个棋块共用两气、任一方填气会被提且没有简单倒扑的双活。
- 计分使用棋子加单方围住的空域，双方接触的空域中立；双活棋子保留。
- **复杂死活、劫争和双活不保证全部判定。未定棋子保留，自博弈的两次停一手按双方接受当前盘面处理，
  使用移除确定死子后的面积估分。该标签仍有近似性，不能当作死活求解器的真值。**
- 1444 手上限只中止对局：不计作和棋、不产生训练样本，Arena 不使用中止对局晋级。

终端终局显示确认的死棋、双活和未定棋块数量。方向键移动，Enter 落子，P 停一手，Q 退出。
摆局搜索还支持 Backspace 撤销、R 清盘。坐标 A–H、J–T（跳过 I），从底部 1 到顶部 19。

## 围棋专用网络输入

除棋子位置、手数、停一手和执棋方外，新增 18 个输入平面与贴目标量：

| 平面 | 内容 |
| --- | --- |
| 0–5 | 双方棋块气数：1 气、2 气、3 气以上 |
| 6–7 | 双方棋块大小 / 361 |
| 8–9 | 双方局部眼形 |
| 10 | 当前方合法落点，含超级劫约束 |
| 11 | 当前方该处落子可提子数 / 361 |
| 12 | 当前方落子后是否只有一气 |
| 13–14 | 双方 Benson 确定活棋 |
| 15–16 | 上一手盘面上的双方棋子 |
| 17 | 上一手改变的交点，包含提子 |

训练和推理共用 `features::encode`。八向对称同步变换盘面、上一手与超级劫历史。
提子后重建搜索节点累加器。贴目按相对视角输入网络。

**模型格式升级为 30、配置格式升级为 22，需要重新训练。**
旧模型、回放和配置不能混用；旧文件保留，新默认路径独立。

## GTP 2 接口

先训练或初始化当前格式模型，然后在围棋 GUI 中配置可执行文件与参数：

```powershell
cargo build --profile fast
.\target\fast\go19.exe gtp --model go19-v32-model.safetensors --simulations 256
```

支持 `protocol_version`、`name`、`version`、`known_command`、`list_commands`、`quit`、
`boardsize`（仅 19）、`clear_board`、`komi`、`play`、`genmove`、`reg_genmove`、`undo`、
`showboard`、`fixed_handicap`（2–5 子）、`set_free_handicap`、`time_settings`、`time_left`、
`final_score`、`final_status_list`。

- 标准输入接收命令，标准输出只有 GTP 响应；支持命令编号、注释与空行。
- `play` 接受指定颜色并维护超级劫历史；`undo` 恢复盘面、历史与停一手状态。
- `komi` 接受有限的小数（精度 0.001），同时影响网络输入与计分。
- 时间控制支持主时间及加拿大读秒，搜索同时受模拟次数与时间预算限制。
- **`final_score` 在仍有未定棋块时返回 `? cannot score`**，不把估分冒充确定结果。
  可用扩展命令 `estimate_score` 查看移除已确认死子后的面积估计，
  `final_status_list unsettled` 列出需要继续对弈或人工裁决的棋块。
- `final_status_list dead` 返回已证明的死子；不把所有未证明活的棋子报告为死子。
- 尚不支持 SGF 导入或 19×19 棋盘，不对外宣称支持这些命令。

协议参考：[GTP 2 规范](https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html)。
保守无条件活棋与封闭区域分析参考：[Solving Go on Small Boards，章节 5](https://project.dke.maastrichtuniversity.nl/games/files/phd/Van%20der%20Werf_thesis.pdf)。

## 棋力评估

```powershell
cargo run --profile fast -- az-init
cargo run --profile fast -- az-search go19-v32-model.safetensors 256 1.5 e5 e6 f5
cargo run --profile fast -- az-bench go19-v32-model.safetensors 256 5
cargo run --profile fast -- az-arena-best go19-v32-model.safetensors go19-v32-best.safetensors 100 256
```

观察训练 loss、自博弈平均手数和中止数，再与冻结的早期模型在相同预算下交替执黑执白评估。
仅 loss 下降不代表棋力提升；终局存在未定棋块的对局应人工复核或与成熟围棋引擎交叉评估。
日常使用 `--profile fast`。Windows/Linux 沿用 Candle CUDA，macOS 使用 Metal/Accelerate。
