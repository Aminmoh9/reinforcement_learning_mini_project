# ❄️ FrozenLake Q-Learning Mini-Project

## 🚀 Overview
**FrozenLake Q-Learning** applies reinforcement learning to the classic `FrozenLake-v1` environment from OpenAI Gymnasium. It demonstrates how Q-Learning can solve navigation tasks in both deterministic and slippery (stochastic) settings, while investigating common pitfalls such as the **"Left-only" policy problem**.

---

## 🧠 Features
- 🤖 **Q-Learning agent** with epsilon-greedy exploration
- 🧊 Supports both deterministic (`is_slippery=False`) and slippery (`is_slippery=True`) environments
- ⚠️ **Negative rewards** for falling into holes to encourage safer policies
- 🎲 **Action selection noise** to break ties and improve exploration
- 📊 Rich visualizations: learning curves, policy maps, Q-table heatmaps, action distributions
- 🎬 Optional animated demonstration of learned policy

---

## 📁 Files
- `main.py` – Main script for training, analysis, and visualization  
- `main.ipynb` – Jupyter notebook version (if available)  
- `frozen_lake_results.png` – Learning curve visualization  
- `frozenlake_extended_analysis.png` – Key plots 
- `frozenlake_extended_comparison-analysis.png` – Key plots for comparing slippery and non-slippery environments


## ▶️ How to Run ##
- Clone the repository or copy the files to your workspace.
Run main.py to train the agent and generate visualizations.
Review the console output and generated plots for analysis.

---

## How to Run ##
- Clone the repository or copy the files to your workspace.
Run main.py to train the agent and generate visualizations.
Review the console output and generated plots for analysis.

---

## 🔍 Key Insights
- Penalizing holes and adding noise to action selection improves policy diversity and success rate in slippery environments.
- The "Left-only" problem is caused by lack of penalty for holes and insufficient exploration.
- The same hyperparameters can yield different results depending on environment stochasticity and reward structure.

---
## 📈 Results ##
- **Deterministic environment:** High success rate (≈93.5%), efficient policy.
- **Slippery environment:** Lower success rate (≈50.8%), but improved action diversity and robustness.
- **Visualizations** show learning progress, policy maps, and Q-table values.

---

## 📚 Resources ##
- Online tutorials
- Matplotlib and Seaborn for visualization

## 🎤 Presentation Link
[Check out the full presentation](https://docs.google.com/presentation/d/162wKMr2CtW1aMuWJoyaSJ0sdbp5E307RBWrbrfXdF90/edit?usp=sharing)

---

## 🛠️ Requirements
- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`

Install dependencies with:
```bash
pip install gymnasium numpy matplotlib seaborn tqdm




