# RL2025-SVayej

# Reinforcement Learning Project 

This repository contains source code for experiments with reinforcement learning algorithms on the Crafter environment. The project uses Proximal Policy Optimisation (PPO) and Actor Critic architectures.

## Directory Structure

```
C:\Users\DELL\Reinforcement-Learning-Project-2026-Crafter\
│
├── ppobase.py    # Base PPO implementation
├── ppo1.py       # PPO Improvement 1
├── ppo2.py       # PPO Improvement 2
├── ppolstm.py    # PPO with LSTM 
├── ppomet.py     # PPO with 1.5m parameters
│
├── actbase.py    # Base Actor Critic implementation
├── act1.py       # Actor Critic Improvement 1 
├── act2.py       # Actor Critic Improvement 2 
├── actmet.py     # Actor Critic with 1.5m parameters
│
├── run.sh        # Script to run experiments
├── README.md     # Project documentation and instructions
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vize786/RL2025-SVayej.git


cd Reinforcement-Learning-Project-2025-Crafter
```

### 2. Set Up the Environment


```bash
conda activate crafter-env
```


### 3. Running Experiments


```bash
bash run.sh
```

Alternatively, run individual Python files directly:

```bash
python ppobase.py
```

## File Overview

- **ppobase.py:** Base PPO implementation.
- **ppo1.py, ppo2.py:** PPO first and second improvemnets.
- **ppolstm.py:** PPO with LSTM.
- **ppomet.py:** Final PPO with 1.5m parameters.
- **actbase.py:** Base Actor Critic implementation.
- **act1.py, act2.py:** Actor Critic first and second improvements.
- **actmet.py:** Final Actor Critic with 1.5m parameters.
- **run.sh:** Shell script to automate running experiments.

## Notes

- Be sure to activate the `crafter-env` Conda environment before running code.
- Modify `run.sh` as needed for your experiments.
