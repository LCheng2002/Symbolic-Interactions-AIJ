# Fig 5 调查记录（2026-08-15）

针对 LLM 行"纺锤形（spindle）随 n 加重"现象的三轮调查结论。分析脚本：
`source/check_spindle_vs_n.py`（分解柱高 = 活跃子集数 × 单子集平均强度）。

## 1. 纺锤随 n 加重的机制：C(n,m) 组合计数效应

柱高 S_k 是对该阶全部 C(n,k) 个子集求和。分解 S_k = A_k × M_k（A_k=阈值过滤后活跃子集数，M_k=单个子集平均强度）：

- 所有模型、所有 n 下，**单子集强度 p_k = S_k/C(n,k) 都随阶数单调衰减**——LLM 并非"高阶交互单个更强"。
- 但 LLM 的 p_k 衰减太慢（Llama n=10：k=2→5 仅掉 ~3 倍），抵消不了 C(n,k) 中间阶的爆炸增长（C(n,n/2)/C(n,1)：n=6→3.3，n=8→8.75，n=10→25.2）→ 中间阶柱高追上 k=1，形成纺锤。n=6 时 C 峰值只有 3.3 倍，补不上 p_k 的 ~9 倍跌落 → 单调衰减，无纺锤。
- 小模型（ResNet-20@MNIST）p_k 衰减 ~100 倍且高阶子集大量被阈值清零（k=8 仅剩 2% 活跃），双保险压住组合增长。

## 2. 稀疏化不是优化失误

- LLM 与小模型用完全相同参数（lr=1e-7, momentum=0.999, niters=20000），loss 均收敛到 L1 地板（重建误差≈0）：LLM 49.7→3.33（L1=3.29）；ResNet 79.4→24.29（L1=24.29）。
- tau 阈值过滤后 LLM 单样本同样稀疏：活跃子集 3.5–7.6%（ResNet 8.3–14.5%），正常样本 k=8 活跃数为 0。之前观察到的"90–100% 活跃"是**跨 20 样本取并集**的假象（各样本稀疏活跃集位置不同）。
- 纺锤在**稀疏化之前的原始 dividend** 中就存在：LLM 原始单子集幅度跨阶几乎平坦（~0.4–0.5），求和即 ∝ C(n,k)。文本语义存在于短语/子句级 token 组合，单 player 不主导——这是数据本征性质。

## 3. 视觉误导的成因与修复（已实施）

现象：Llama 行 all 面板纺锤随 n 加重，但 gen/non-gen 面板形状看似不随 n 变。

- 无 bug：`all = gen + non_gen` 恒等式精确成立（误差 ≤3.5e-18）。
- 真相：gen（1 阶衰减形）与 non-gen（纺锤形）**形状各自稳定，但相对权重随 n 剧变**。Llama non-gen 峰值 0.40(n=6)→0.91(n=8)→2.60(n=10)，涨 6.5 倍；n=6 时 non-gen 仅 gen 的 ~12%（all≈gen），n=10 时中间阶 non-gen 是 gen 的 2.4 倍（all 呈纺锤）。
- 视觉问题根源：原代码 non-gen 组**每个面板独立 y 缩放**（n=6 面板 ylim ±0.65，n=10 ±4.16），幅度增长完全被隐藏。
- **修复**：`source/render_with_llm_cache.py` 中 non-gen 组改为**行内跨 n 共享 y 轴**；`outputs/main_paper_robustness_with_llm.{pdf,png}` 已重新生成。
- **待办**：`final/AIJ-Fig5-robustness.pdf` 由 `source/AIJ-Fig5-robustness.pptx` 手工排版，需重新导入新面板导出。
- 附带注意：non-gen 相对权重随 n 增长部分来自"shared"判据的统计效应（C(n,m) 越大，baseline 稀疏活跃集覆盖率越低，target 中阶交互更难被判为共享；Llama gen 占比 k=4 从 0.57→0.30）。形状结论不受影响，但"non-gen 占比随 n 上升"不宜过度解读。

## 4. 模型间差异（Llama/Phi vs Qwen/GLM）的主因：n=10 缓存稀疏化配置不一致

all 面板纺锤强度（max S_中阶/S_1）：

| 模型 | n=6 | n=8 | n=10 | n=10 的 q_bound 配置 |
|---|---|---|---|---|
| Qwen2.5-7B | 0.45 | 0.46 | 0.49 | **全局常数 0.1446**（≈0.025×5.78） |
| GLM-Z1-9B | 0.34 | 0.42 | 0.48 | **全局常数 0.1366**（≈0.025×5.46） |
| Llama-3-8B | 0.37 | 0.55 | 0.92 | per-sample 0.025×range（20 个不同值） |
| Phi-4-14B | 0.29 | 0.41 | 0.62 | per-sample 0.025×range（20 个不同值） |

关键事实：

- **n=6、n=8 四个模型配置完全一致**（q_bound 均 = 0.025 × 各样本自己的 output range，qb/range 恒为 0.025）；此时无清晰的两对分组（n=8：Llama 0.55 > Qwen 0.46 > GLM 0.42 ≈ Phi 0.41）。模型对分裂几乎只出现在 n=10。
- n=10 缓存拷贝自 `interactions/llm/generalization`（Fig 3/6 同源，**Fig 3/6 的 LLM 面板同样受影响**）。
- 机制：稀疏化（`AndOrHarsanyiSparsifier`）对每个子集 reward 加有界扰动 q（|q|≤q_bound，每步 SGD 前 clamp），再做精确 Harsanyi 变换并最小化 huber-L1。q_bound 是"抹平交互"的预算：Qwen/GLM 在 n=10 对典型样本的相对预算约 4.7%/3.1%，是 Llama/Phi（2.5%）的 1.2–1.9 倍 → 中阶 dividend 被抵消得更多 → 纺锤更弱。四个模型的原始 dividend 几乎相同（中阶单子集平均幅度均 ~0.4–0.5），差异完全产生在稀疏化保留哪些子集这一步。
- 次要成分：Phi-4 纺锤强度与 1/range 强相关（corr=0.79），其最强纺锤样本 range 仅 0.27/0.28——低 range 样本 figure 阈值（tau×range）塌缩的叠加效应。Llama 在相同预算下仍强于 Phi（0.92 vs 0.62），高 range 样本（Sample_18，range=7.4）也呈纺锤——可能含少量本征成分，但非主因。
- 另有少数**退化样本**（Qwen Sample_13/15，range≈0.013）阈值塌缩到 ~4e-4 导致 ~77% 子集幸存；对柱高贡献极小，剔除后纺锤形态不变。

## 5. 建议待办

1. ~~用统一配置重跑 n=10 的 Qwen/GLM 稀疏化~~ → **已定方案（2026-08-15）：统一方向为全局常数，尺度口径 = 缓存实际均值**。所有模型、所有 n 都使用 `q_bound = 0.025 × scale[model]`，其中 `scale[model] = 该模型全部缓存样本的 mean|v(N)−v(∅)|`，由 `resparsify_global_qbound.py` 的 `compute_scales()` 在运行时从旧缓存（`llm_target_single_cache_merged`，3 个 n 共用同 20 句 finance 样本，per-n 均值完全相同）现算，**不使用** `interaction_demo_with_anchor.py` 的历史手工表（该表已被否决：来源不可复现，且缺 gemma 条目）。实测 scale：Qwen 3.837、Llama 4.568、GLM 5.273、Phi-4 4.569、gemma 5.416、deepseek 8.309。重跑脚本：`source/resparsify_global_qbound.py`（从已保存的 rewards/masks 重跑，无需 LLM 推理；lr=1e-7, momentum=0.999, niters=20000, p/q 从零初始化，末尾 clamp 到 q_bound；每个样本的 `sparsify_config.json` 记录 scale 及 `scale_provenance`）。输出到新缓存 `interactions/robustness_n/llm_target_single_cache_globalq/`，旧缓存保留不动。渲染脚本支持环境变量 `FIG05_LLM_ROOT` 切换缓存根（默认仍指向旧缓存，验证后再切换）。
2. ~~评估 `llm/generalization` 缓存重跑对 Fig 3/6 的影响~~ → 已排查，见 §7：**Fig 3、Fig 6 的 LLM 面板需要替换**。
3. 考虑对 figure 阈值加绝对下限，或剔除 range 过小的退化样本。
4. 可选：画 S_k/C(n,m)（每子集平均）版本——所有模型所有 n 均为单调衰减，可彻底消除组合伪影。

## 6. 重跑结果

批次完成后在此记录：四个模型的纺锤强度对比（新缓存 vs 旧缓存）、渲染图差异、以及是否将 `llm_target_single_cache_globalq` 设为默认。

### 6.1 批次执行记录（2026-08-15）

- **范围**：6 模型（Qwen2.5-7B、Llama-3-8B、GLM-Z1-9B、Phi-4、gemma-3-12b-pt、DeepSeek-R1-Distill-Llama-8B）× n∈{6,8,10} × 20 样本 = 360 个稀疏化任务。gemma/deepseek 不是图的行，但作为 baseline 参与 gen/non-gen 的 shared 判定，必须齐全。
- **第一次启动失败**：脚本头部改为运行时 `compute_scales()` 后，main() 残留两处对已删除常量 `MEAN_ABS_VN_V0_FINANCE` 的引用（argparse 默认值、jobs 过滤），6 个进程启动即 NameError 退出，无产出。修复：`resparsify_sample()` 改为传入 `scale` 参数；`--models` 默认改为 `MODELS`；新增 `--shard/--num-shards` 分片；启动时打印各模型 scale 及 per-n 分解。
- **冒烟测试通过**（Llama n=10 Sample_18，~55s/样本）：scale=4.5675（三个 n 的 per-n 均值完全相同，印证 v(N)−v(∅) 不随 player 划分变化），effective_q_bound=0.114188=0.025×scale，输出文件（I_and/I_or/p/q/q_bound/before_sparsify/sparsify_config）齐全。旧批次（anchor 表口径，跑 19/300 后中止）的残留输出已整目录删除。
- **正式批次**：6 路并行（3 张 A800 各 2 进程），每路 60 任务；日志 `/tmp/resparsify_shard{0..5}.log`。批次幂等（输出覆盖写），中断后按 shard 重跑缺失部分即可。
- **实际完成（2026-08-15）**：360/360 全部成功，单 shard 耗时 58–76 分钟。

### 6.2 校验结果

- **齐全性**：6 模型 × 3 个 n × 20 样本的 `sparsify_config.json` / `I_and.npy` 全部到位（各 360 个）。
- **q_bound 统一性**：每个模型跨全部 60 个样本只有一个 q_bound 值（Qwen 0.095924、Llama 0.114188、GLM 0.131823、Phi 0.114215、gemma 0.135412、deepseek 0.207729），qb/range 不再恒为 0.025（范围 0.006–6.9，6.9 来自 Qwen Sample_13/15 等 range≈0.01 的退化样本，见 §4）。
- **收敛**：与旧缓存同模式。同一旧样本对比（Llama n=10 Sample_18）：旧（per-sample 预算 0.1858）loss 500.2→102.9；新（全局预算 0.1142）loss 500.2→122.5——预算收紧则终值略高，符合预期；Harsanyi 求和性质保持（sum(I)=7.414 vs v(N)−v(∅)=7.433，残差来自有界扰动 q，量级与旧缓存一致）。

### 6.3 纺锤强度对比（all 面板 max S_中阶/S_1，同口径重算新旧）

| 模型 | n=6 旧→新 | n=8 旧→新 | n=10 旧→新 |
|---|---|---|---|
| Qwen2.5-7B | 0.65→0.64 | 0.55→0.55 | 0.64→0.66 |
| Llama-3-8B | 0.44→0.46 | 0.55→0.48 | **0.92→0.75** |
| GLM-Z1-9B | 0.37→0.36 | 0.42→0.38 | 0.49→0.49 |
| Phi-4-14B | 0.51→0.52 | 0.41→0.37 | **0.61→0.51** |

- Qwen/GLM 基本不变（n=10 本来就是全局常数；Qwen 新 scale 3.84 < 旧 5.79，预算收紧但纺锤几乎不动 0.64→0.66，说明在该预算区间内配置差异对 Qwen 影响很小）。
- Llama/Phi 的 n=10 纺锤明显减弱（0.92→0.75、0.61→0.51），n=8 也略降——证实 §4 的主因判断：**两对模型的分裂主要由缓存配置不一致造成**，统一后 n=10 四模型强度收敛到 0.49–0.75。
- 残余差异（Llama 0.75 仍最高、GLM 0.49 最低）为配置无关的本征差异（各模型 p_k 跨阶衰减速度不同），不再是伪影。
- 渲染：`FIG05_LLM_ROOT=.../llm_target_single_cache_globalq python source/render_with_llm_cache.py` 已重渲 `outputs/main_paper_robustness_with_llm.{pdf,png}`，旧缓存渲染归档在 `outputs/archive_merged_cache_20260815/`。目视确认 Llama/Phi 的 n=10 all 面板纺锤柱明显变矮，Qwen/GLM 行不变。
- **待确认**：render 脚本默认 `LLM_SINGLE_ROOT` 是否切换为 `globalq`（当前默认仍指向旧缓存）；`final/AIJ-Fig5-robustness.pdf` 仍需在 pptx 里手动换面板；`llm/generalization`（Fig 3/6 用）缓存的配置不统一问题本次未处理（§5 第 2 条）。

## 7. 其他图受 LLM 统一重跑影响的排查（2026-08-15）

判据：图的 LLM 数据是否来自"跨模型 q_bound 口径不一致"的缓存。逐缓存用 `q_bound.npy`（或 meta 中的 `mean_vn_v0`）核验 qb/range 是否恒为 per-sample 0.025。

| 图 | LLM 数据源 | 口径核验 | 是否需要替换 |
|---|---|---|---|
| Fig 5 robustness | `robustness_n/llm_target_single_cache_merged` | 不一致（§4） | **已重跑**（globalq，§6） |
| **Fig 3 distributions** | `llm/generalization` | **不一致**：Qwen/GLM 20 样本仅 2 个 q_bound 取值（全局常数 0.1446/0.1366 + 个别值），Llama/Phi/gemma/deepseek 为 per-sample 0.025×range | **需要** |
| **Fig 6 fit** | `llm/generalization`（同上） | 同上 | **需要** |
| Fig 7 noise | `llm/noise` | 一致：4 模型（Qwen/Llama/GLM/Phi）均逐样本记录 `mean_vn_v0`（每样本值不同），统一 per-sample 口径 | 否 |
| Fig 9 correctness（additional） | `llm/misclassification` | 一致：GLM+Phi 均严格 per-sample（qb/range 恒 0.025） | 否 |
| Fig A3–A6 examples | `llm/examples` | 一致：Qwen+deepseek 均 per-sample（qb/range 恒 0.02） | 否 |

**Fig 3/6 的替换路径（无需重新稀疏化）**：`llm/generalization/finance` 与 `robustness_n/llm_target_single_cache_merged/n=10/finance` 的 before_sparsify rewards **120/120 逐字节相同**（6 模型 × 20 样本，cmp 核验）——即 Fig 5 的 n=10 本就拷贝自该缓存。因此 `llm_target_single_cache_globalq/n=10` 的输出就是 `llm/generalization` 的统一口径重稀疏化版本，直接将 Fig 3/6 的渲染输入指向 globalq 的 n=10（或拷贝为新缓存）即可，只涉及重渲染。

### 7.1 Fig 3 / Fig 6 新口径重渲（2026-08-15，旧图全部保留）

渲染方式：无需重新稀疏化，直接把 LLM 输入指向 `robustness_n/llm_target_single_cache_globalq/n=10`（目录布局与 `llm/generalization` 相同）。

- **Fig 3**：新脚本 `Fig_03_distributions/source/render_with_llm_cache_globalq.py`（原脚本仅改缓存根 + 输出名）。新图 `outputs/main_with_llm_single_cache_globalq.{pdf,png}`；旧图 `outputs/main_with_llm_single_cache.{pdf,png}`（2026-08-02，与现行代码同版式）未动。
- **Fig 6**：`main_w_LLM.ipynb` 转脚本的两份变体——`source/render_globalq.py`（新数据 → `outputs_globalq/`）与 `source/render_oldcache.py`（旧数据 + 当前版式 → `outputs_oldcache/`）。注意旧存档 `outputs/main.png` 是 2026-07-13 由旧版式渲染的，对比应以 `outputs_oldcache/` 为准。
- 数值对比（Fig 3 柱总量 |old|→|new|）：GLM 几乎不变（all_pos 9.89→9.93）；Qwen 变大（8.52→9.48，non-gen 4.38→5.21，因新 scale 3.84<旧 5.79 预算收紧）；Llama/Phi 变小（all_pos 11.42→9.74、10.52→8.53；non-gen 6.73→5.13、5.60→3.78）。gen 组全部基本稳定，变化集中在 non-gen 组。
- Fig 6 SIM（旧→新）：Qwen 0.801→0.802，**Llama 0.754→0.799**（+0.045，纺锤减弱后拟合显著提升），GLM 0.887→0.874，Phi 0.758→0.753；5 个小模型行 bit 级不变（ sanity check 通过）。Llama 行 all 面板中阶鼓包明显变矮。
- **待用户确认**后：替换 pptx 面板（Fig3 `source/AIJ-Fig3-distributions_LLM.pptx`、Fig6 `source/AIJ-Fig6-fit_LLM.pptx`）、重导 final PDF、更新 MANIFEST/CHECKSUMS。

### 7.2 Fig 6 predict_tau 重扫（2026-08-15，新缓存数据）

脚本 `Fig_06_fit/source/scan_predict_tau_globalq.py`，网格与选取规则同旧扫描（predict_tau ∈ {0.03, 0.05, 0.08, 0.10}，full SIM 取 argmax），结果 `Fig_06_fit/outputs_globalq/llm_sim_predict_tau_scan.csv`：

| 模型 | 0.03 | 0.05 | 0.08 | 0.10 | 旧选取 | 新 argmax |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.7245 | 0.7886 | **0.8049** | 0.8020 | 0.10 | **0.08**（+0.003） |
| Llama-3-8B | 0.7283 | **0.7987** | 0.7369 | 0.7723 | 0.05 | 0.05（不变） |
| GLM-Z1-9B | 0.7615 | 0.8201 | **0.8741** | 0.8501 | 0.08 | 0.08（不变） |
| Phi-4 | 0.7431 | **0.7530** | 0.7345 | 0.6624 | 0.05 | 0.05（不变） |

仅 Qwen 的 argmax 从 0.10 移到 0.08（收益仅 +0.003，属边界移动）。已将 `render_globalq.py` 中 Qwen 的 `set_predict_tau` 改为 0.08 并重渲；旧 tau 版备份为 `outputs_globalq/main_oldtau_qwen0.10.{pdf,png}`。

## 8. 定稿与清理（2026-08-15，用户确认采用新口径）

**Fig 3/5/6 全部定稿在 globalq 缓存上，旧缓存与对比产物已归档/移除。**

- Fig 5：`render_with_llm_cache.py` 默认 `LLM_SINGLE_ROOT` 切到 `llm_target_single_cache_globalq`（`FIG05_LLM_ROOT` 保留为覆盖项）；重渲与审查版逐字节一致（md5 相同）。`main_w_LLM.ipynb` 同步 patch。
- Fig 3：`render_with_llm_cache.py` 与 `main_with_llm_cache.ipynb` 的 LLM 根切到 `globalq/n=10`；正稿 `outputs/main_with_llm_single_cache.{pdf,png}` 由新数据生成（与审查版 md5 相同）。
- Fig 6：`main_w_LLM.ipynb` patch（LLM 根 → `globalq/n=10`，Qwen `set_predict_tau` 0.10→0.08，依据 §7.2 重扫）并就地执行；`outputs/main.{pdf,png}` 与审查版 md5 相同，SIM：Qwen 0.805 / Llama 0.799 / GLM 0.874 / Phi 0.753。
- **已删除**：`interactions/llm/generalization/`（13M，原始数据仍在 `AIJ_fig/interaction_target_single_cache`，且 rewards 已逐字节保存在 globalq 的 before_sparsify 中）与 `interactions/robustness_n/llm_target_single_cache_merged/`（28M，全部被 globalq 取代）。
- **归档**：`archive_globalq_migration_20260815/`（Fig 3/5 旧渲染、Fig 6 旧版式导出、新旧对照渲染、tau 重扫 CSV、一次性变体脚本；README 说明各目录来历）。
- 文档同步：`MANIFEST.tsv`、`PATH_MAPPING.tsv`、`interactions/README.md`、Fig 3/5/6 README、`Fig_06_fit/source/PREDICT_TAU_SELECTION.md`（重扫附录）；`CHECKSUMS.sha256` 全量重生成（214021 条，旧缓存 0 引用）。
- **遗留人工步骤**：三个图对应的 pptx 面板替换与 `final/*.pdf` 重导需手动完成（Fig3 `source/AIJ-Fig3-distributions_LLM.pptx`、Fig5 `source/AIJ-Fig5-robustness.pptx`、Fig6 `source/AIJ-Fig6-fit_LLM.pptx`），面板取自各 `outputs/` 正稿。

## 9. 定稿与清理（2026-08-20，用户确认）

- **y 缩放口径变更**：non-gen 组不再行内共享 y 轴。数值仍行级归一化，但每面板独立适配自身数据，ylim = max × 1.5（柱子约占面板 2/3 高，`NON_GEN_FILL_INV`）。§3 的共享 y 轴修复被此口径取代。
- **定稿**：正稿面板为 `outputs/main_paper_robustness.{pdf,png}`（由 `render_with_llm_cache.py` 生成），`final/AIJ-Fig5-robustness.{pdf,png}` 由 `compose_submission_figure.py` 直接合成，pptx 手工排版路径废弃（pptx 已删除）。
- **清理**：删除 `render_with_llm_cache_normalized.py`、`main.ipynb`、`main_w_LLM.ipynb`、`check_spindle_vs_n.py`、`verify_llm_consistency.py`、`resparsify_global_qbound.py`、`__pycache__/` 及旧命名输出（`main_paper_robustness_n6_8_10*`、`*_five_models_normalized_non_gen*`）。本文件中对这些脚本路径的引用为历史记录。
- 文档同步：`README.md`（重建命令与缩放口径）、`MANIFEST.tsv`（Figure 5 行）、`CHECKSUMS.sha256`（Fig_05 段与 MANIFEST 条目已更新，`sha256sum -c` 通过）。注意 PDF 含生成时间戳，重渲染后 PDF 哈希会变、PNG 位级稳定。
