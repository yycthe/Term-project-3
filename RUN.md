# Run Guide (Conda Only)

This project is designed to run in the **Conda** environment `nba-predictor` (includes `llvm-openmp` to avoid XGBoost `libomp` issues on macOS).

## 1. First-time setup: create and activate environment

From the project directory, run:

```bash
conda env create -f environment.yml
conda activate nba-predictor
```

## 2. Start the project

**Prediction UI (Streamlit):**

```bash
conda activate nba-predictor
streamlit run app.py
```

Open in browser: http://localhost:8501

**Train / run the full Agent:**

```bash
conda activate nba-predictor
python agent.py
```

This generates/updates model artifacts and reports under `models/` and `outputs/`.

## 3. For subsequent runs

```bash
# Enter project root directory (your cloned repo)
# Example: cd nba-game-predictor
conda activate nba-predictor
streamlit run app.py
```

(Or run `python agent.py` for training.)
