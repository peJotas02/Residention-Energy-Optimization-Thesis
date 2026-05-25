# Reinforcement Learning and MILP for Residential Energy Management

Source code accompanying the MSc thesis:

**"Reinforcement Learning for Residential Energy Optimization"**

Author: Pedro Jorge  
Institution: Lund University  
Year: 2026

## Overview

This repository contains the implementation developed for the thesis project, comparing Mixed-Integer Linear Programming (MILP) and Reinforcement Learning (RL) approaches for residential energy management.

The project considers household energy optimization under dynamic electricity prices, photovoltaic generation, battery storage, and electric vehicle integration.

---

## Repository Structure

```text
MILP/
    ├── ...
    └── ...

RL/
    ├── Battery_Only/
    │   ├── Agent.py
    │   ├── DQN.py
    │   ├── HEMS.py
    │   ├── PlayLoop.py
    │   └── Models/
    │
    └── Battery_and_EV/
        ├── Agent.py
        ├── DQN.py
        ├── HEMS.py
        ├── PlayLoop.py
        └── Models/
```

---

## Main Components

### RL implementation

- **HEMS.py** — Gymnasium environment representing the residential energy management system.
- **Agent.py** — DQN agent implementation.
- **DQN.py** — Neural network architecture.
- **ReplayBuffer.py** — Experience replay mechanism.
- **PlayLoop.py** — Training and evaluation loop.

Two RL configurations are included:

- Battery-only household system
- Battery + Electric Vehicle household system

### MILP implementation

Contains the optimization formulation used for comparison with RL-based control.

---

## Requirements

Python dependencies:

```bash
numpy
pandas
torch
gymnasium
matplotlib
pyomo
```

---

## Data

Datasets used in this work may be omitted due to licensing, privacy, or project restrictions.

---

## Citation

If using this repository, please cite the associated thesis.
