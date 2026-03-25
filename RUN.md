# 运行说明（仅 Conda 环境）

项目统一用 **Conda 环境** `nba-predictor` 运行（已包含 llvm-openmp，避免 macOS 上 XGBoost 的 libomp 报错）。

## 1. 首次：创建并激活环境

在项目目录执行：

```bash
conda env create -f environment.yml
conda activate nba-predictor
```

## 2. 启动项目

**预测 UI（Streamlit）：**

```bash
conda activate nba-predictor
streamlit run app.py
```

浏览器打开：http://localhost:8501

**训练 / 跑完整 Agent：**

```bash
conda activate nba-predictor
python agent.py
```

会生成/更新 `models/` 与 `outputs/` 下的模型和报告。

## 3. 之后每次运行

```bash
# 进入项目根目录（克隆后的仓库目录）
# 例如：cd nba-game-predictor
conda activate nba-predictor
streamlit run app.py
```

（或 `python agent.py` 做训练）
