# 多模型延迟评估

## 多run汇总与训练曲线

一个STUDY对应一种算法，MODELS中的各条目是该算法独立训练的run，
id用于标识run；不同算法分别建立研究目录，不在此处自动混合。
原始评估数据和summary.csv仍按run保存。跨run统计先计算每个run的指标，
再等权计算均值、样本标准差（ddof=1）和有效run数；不按episode或决策数加权。
只有一个有效run时标准差为空、图中不画波动带，缺失值不补零。

训练日志默认读取各条目config所在目录的metrics.json。
可在模型条目添加tensorboard="results/tb_logs/<run></run>"指定事件文件目录。
Sacred优先，TensorBoard只补充Sacred未记录的指标，不拼接同指标的两份序列。
训练日志的step按本项目日志约定为环境交互步数；导入外部算法需先确认其含义。
重复step保留最后一条；不同run按共同覆盖区间的记录步数并集线性插值，不外推。
灰线保留各run原始轨迹，均值和标准差带仅位于共同区间；默认不平滑。
缺少某指标的run不参与该指标统计，来源和缺失情况写入training_sources.json。
这些曲线来自训练日志，与训练完成后的延迟扫描结果分开。

新增表格：

- run_summary.csv：各延迟条件的跨run均值、标准差和有效run数。
- quality_per_run.csv、run_quality.csv：重建/动作分桶的单run与跨run统计。
- dynamic_per_run.csv、run_dynamic.csv：连续环境时间步的单run与跨run统计。
- run_efficiency.csv：按硬件分组的效率统计。
- training_per_run.csv、training_summary.csv：原始及对齐后的训练曲线。

运行draw_all.py会更新上述表格，并输出到figures/evaluation和figures/training。
热力图显示平均胜率及标准差（百分点），曲线阴影/误差棒为跨run标准差。
原summary.csv中的Wilson/bootstrap置信区间仍是单run局内不确定性，
不替代跨训练run的标准差。动态图后期与误差分桶的有效run数可能减少，
可由相应表格的*_n列检查。训练日志读取需要numpy、pandas；配置TensorBoard
来源时还需tensorboard，绘图需要matplotlib。
旧figures根目录中的单run图不会自动删除；新图以两个子目录为准。

## 运行

正式脚本`delay_study.py`维护评估逻辑，实验实例的`.local.py`只配置参数并调用它。
不要复制整份正式脚本，否则正式逻辑更新后，本地副本仍可能执行旧流程。

### 1. 创建本地参数文件

在`scripts/eval_scripts/`下新建`glide12_mmm2.local.py`。
下面是四个GLIDE1.2训练run的示例；将RUNS中的占位名称替换为实际目录名。
示例假定Sacred、模型和TensorBoard使用相同的run目录名；若不同，直接在
MODELS中分别填写对应路径即可。

```python
"""Evaluate four GLIDE1.2 training runs on MMM2."""
import delay_study as study


RUNS = [
    ("seed1", "<seed1训练目录名>"),
    ("seed2", "<seed2训练目录名>"),
    ("seed3", "<seed3训练目录名>"),
    ("seed4", "<seed4训练目录名>"),
]

study.STUDY = "GLIDE1.2_MMM2_best_4runs"
study.MODELS = [
    {
        "id": run_id,
        "config": f"results/sacred/{run}/1/config.json",
        "checkpoint": f"results/models/{run}/best_model",
        "tensorboard": f"results/tensorboard_logs/{run}",
    }
    for run_id, run in RUNS
]

# Gaussian and uniform share raw means and standard deviations.
study.MEANS = [-2, -1, 0, 1, 2]
study.STDS = [0, 0.5, 1, 1.5, 2]
study.CAP = 8
study.EPISODES = 64       # Per run and delay condition.
study.PARALLEL = 4
study.SEED = 101          # Evaluation seed, not the training seed.
study.MASK_INTERVENTION = False
study.FEATURE_GROUPS = {}


if __name__ == "__main__":
    study.main()
```

必须使用`study.MEANS`等模块属性赋值，单独写`MEANS = ...`不会修改正式脚本参数。
把本地文件放在正式脚本同目录可直接import，且必须保留主入口判断，
避免Windows环境子进程重复启动评估。

路径相对项目根目录，也可以是绝对路径。`id` 在研究内唯一。模型在各自训练地图上测试；
不尝试跨不同输入/动作/智能体维度加载。当前入口支持现行 BCRBC/GLIDE
checkpoint，不是任意算法或历史架构的通用加载器。
checkpoint指向包含`agent.th`的目录；示例选取每个run的`best_model`。
正式实验应统一checkpoint选择标准，不能把不同训练预算的结果视为同预算比较。
Sacred配置不在`1/`时填写实际编号；没有TensorBoard日志时删除该字段。

旧配置若缺少 `bcrbc_time_block_every`，核对原训练源码后在模型条目中显式
填写 `legacy_time_block_every=4`（或实际值）。不自动用当前默认配置补齐
网络，也不改变 horizon、solver steps、模型宽度或深度。

### 2. 运行评估

在项目根目录执行：

```bash
conda activate marl_stable
python -u scripts/eval_scripts/glide12_mmm2.local.py
```

示例共49个条件、每run每条件64局、4个run，合计12544局。
正式脚本默认并行数为8；示例覆盖为4，可按显存调整。
模型、条件和评估批次顺序执行；不启动多个GPU评估进程争资源。
末批可以少于配置的并行数，因此总局数严格等于配置值。
未填写模型清单不会启动任务。

### 3. 汇总与绘图

```bash
python scripts/eval_scripts/draw_all.py results/evaluate/GLIDE1.2_MMM2_best_4runs
```

该命令只读取已有结果和训练日志，不启动环境。输出位于研究目录下的
`tables/`、`figures/evaluation/`和`figures/training/`。
绘图环境需要numpy、pandas、matplotlib；配置了TensorBoard来源时还需tensorboard。

中断后可用相同命令续评，已完成批次会跳过。修改模型、网格、局数、并行数或
评估源码后，应使用新STUDY名称，避免混合实验记录。只重画图不需要重跑评估。
已有包含完整执行逻辑的`.local.py`应改为上面的参数入口，保留原参数值。

### 环境复用与随机性

相同地图、环境初始化配置和并行数共用SMAC进程。切换模型或延迟条件不重启
SC2，只更新延迟采样器并reset；模型历史、KV缓存和诊断状态每批重建。
不同地图或初始化配置需新建进程；不足整批的末批单独复用较小的环境组。
执行顺序按环境配置分组，不保证全局模型顺序。

延迟与模型随机数按每批seed初始化，但SMAC随机状态随reset继续推进。
不同条件不承诺相同初始战局；中断后跳过完成批次，也不承诺复现不中断时的
环境随机轨迹。每批result记录environment_seed、environment_batch_index和
worker_pids。当前协议为v4，勿混入旧版研究目录。
根目录session.json记录本次待运行批次，session.log保存整体进度和进程输出；
批次run.log保存该批Python输出。

每个条件沿用相同的延迟采样seed序列，独立的 CPU torch Generator 采样延迟。
模型生成噪声不会推进延迟随机流。策略改变仍然会导致环境轨迹分叉。

## 条件

- 高斯与连续均匀分布共用 μ = -2,-1,0,1,2；σ = 0,0.5,1,1.5,2。
  在delay_study.py的MEANS、STDS中统一配置。
  均匀分布全宽w=√12 σ，边界为[μ−√3 σ, μ+√3 σ]；匹配的是处理前的均值和标准差。
- 实际延迟为 `min(8, ceil(max(0, sample)))`，坐标不是离散后的实际均值。
- 两个网格的零方差格点共享固定延迟结果；非正均值对应固定0。
- 额外固定4、8。原离散均匀条件已被连续均匀网格替代，旧结果不重新解释。
- 混合：50% N(0,1)+50% N(2,1)；90% N(0,1)+10% N(4,1)。
- 动态：N(0,1)/N(2,1)每16步交替；两状态Markov保持概率0.9。
- 混合/动态的状态在单个环境的智能体之间共享，条件高斯噪声独立。
  balanced mixture与动态对照具有相同状态组成，但时间相关性不同；有限
  episode的实际频率不必相同。周期模式从低延迟开始，Markov从均匀状态开始。
- 始终保持cap=8，不用均匀分布上界改变控制器可修正的历史范围。
- 两个网格各25格，按零方差条件去重后总共49个评估条件，64局时每模型3136局。

分布只决定新数据包到达时间，不重新抽取旧包。训练仍走原来零延迟逻辑。
这是一组观测延迟评估，不应标记为通信延迟实验。环境state和合法动作沿用
现有wrapper语义（即时），论文需说明这个信息条件。

## 流程与诊断

实际执行始终采用checkpoint对应的决策流程。真实完整本地观测只进入影子
Reference，使用实际执行动作历史；Reference不是最优策略或性能上界。
诊断不更新模型、不改变执行缓存，也不消费策略随机流。

`MASK_INTERVENTION=False`为默认。启用后额外记录关闭生成的影子动作，
字段名为`intervention_*`、`correction`、`damage`，只解释同模型的推理干预，
不是独立训练的“仅编码基线”，也不产生这种基线的胜率。

重建诊断始终可以比较执行z与MASK编码z；这不需要额外执行一个MASK策略。
Reference z来自完整本地历史的encoder，生成z来自实际决策前向，绝不重新
采样。Decoder比较如下：

- Reference observation：完整z与完整decoder历史。
- MASK observation：当前已修正MASK历史的decoder输出。
- Generated observation：用同一个MASK decoder历史条件解码实际生成z。

默认输出latent/observation MSE、MAE、cosine及补全前后latent MSE差值。
latent只建议同模型内部对照。cosine对零向量采用PyTorch的epsilon约定。
`FEATURE_GROUPS={map_name: {group: [start, end]}}`可按真实观测布局增加分组MSE；
不能盲目复用不同地图的切片。不配置时只报告整体误差，不伪造语义分组。

动作指标：greedy agreement、Q-softmax KL（temperature=1）、参考Q差距。
KL不是epsilon-greedy行为分布的KL，参考Q差距也不是环境regret。
动作及重建主汇总仅统计“当前缺失且至少两个合法动作”的位置，保存分子分母；
计数为0时CSV留空。完整观测自身的`reference_obs_all_mse`另按全部位置统计，
无延迟也可检查tokenizer重建水平。

数据还包括当前缺失率、从未收到率、最新观测年龄、连续缺失长度、真实采样
延迟直方图、P95、截顶比例及动态状态/状态持续步数。`age=null`表示从未
收到；分桶表编码成-1，绘图单独标记，不接入普通age曲线。

胜率区间采用95% Wilson；诊断比值区间按episode重采样1000次（加权分子/
分母）。这些区间不是训练seed方差。误差分桶与动态图是描述性均值，不是
因果效应或显著性检验。动态图按环境时间步连续汇总，不按状态切换重置横轴。
下方显示每步仍在运行的对局中处于高延迟状态的比例；Markov各局切换时刻不同，
该比例不是单条轨迹的状态。误差与一致率仍只统计有效缺失决策，无样本处留空，
不插值。后期只包含尚未结束的对局，不能将曲线变化单独解释为适应或恢复。

动作选择时间在CUDA同步后计时，不包含评分分支；它是整个环境batch的延迟，
不是单智能体推理时间。吞吐与峰值显存包含诊断，不能作为部署成本。
诊断decoder会重新计算已修正历史，有额外开销，不要将其算入训练性能。

## 数据与复跑

```text
results/evaluate/<STUDY>/
  manifest.json
  runs/<model_id>/<condition_id>/batch_<episode_offset>/
    job.json, effective_config.json, run.log
    episodes.jsonl, decisions.jsonl.gz, result.json
  tables/episodes.csv, summary.csv, quality_buckets.csv
  figures/evaluation/*.png, *.pdf
  figures/training/*.png, *.pdf
```

每个decision保存标量误差和动作，未默认保存高维向量。终局支持的dead_allies/
dead_enemies写入episode；环境不提供时为空。模型配置/checkpoint哈希、评估
源码哈希及git提交记录在清单中。结果不是只有TensorBoard均值。

仅`result.json`完成的批次参与汇总；中断批次重新执行，完成批次跳过。
同STUDY不允许混入改变后的协议或checkpoint；修改模型、条件、并行数或
局数请使用新STUDY名称。数据不会自动删除或合并成无法追溯的跨版本结果。

```bash
python scripts/eval_scripts/summarize_study.py results/evaluate/<STUDY>
```

## 独立画图

汇总/绘图离线运行，只需numpy、pandas、matplotlib，不需要SC2或PyTorch。
评估端使用训练环境；绘图端可用单独环境。每个脚本都接受研究根目录：

```bash
python scripts/eval_scripts/draw_all.py results/evaluate/<STUDY>
# 同时绘制指定episode的病例图：
python scripts/eval_scripts/draw_all.py results/evaluate/<STUDY> <model_id> <condition_id> <episode> <agent>
```

地图分开画；胜率、动作一致率和其他比例统一显示0–100%；未评估单元为空。
同一研究内，重建图按z/obs分别统一纵轴；动态图的latent MSE统一纵轴；
策略图的KL与Q差距各自统一纵轴并包含标准差范围。不同单位不强行共用范围。
绘图实现统一放在`plots/`子目录，仍可单独运行其中的脚本。
`draw_all.py`先更新汇总表，再生成七类评估图与训练过程曲线，保存PNG和PDF。
胜率热力图分别输出高斯和均匀网格，共用均值/标准差坐标与0–100%色阶。
旧研究没有uniform_cells时只画已有高斯网格，不从旧离散均匀点推造新网格。
重建曲线保留全部数据点，但长横轴只显示少量整数刻度。
模型不同训练结构与seed的解释来自manifest，不把测试episodes当成训练seed。
病例应预先指定或明确选择规则，不能只挑成功案例。
