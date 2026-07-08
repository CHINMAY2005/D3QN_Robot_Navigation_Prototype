# D3QN Robot Navigation Prototype

An interactive 2D simulation environment designed for testing, visualizing, and researching robot navigation using Reinforcement Learning (RL) techniques (inspired by D3QN / Double DQN architectures) alongside traditional pathfinding and steering algorithms.

The project features a **FastAPI backend** that simulates robot motion kinematics and calculates reward functions, paired with a **React (Vite) frontend** providing real-time canvas rendering, telemetry feedback, and metrics charts.

---

## Architecture Overview

```mermaid
graph LR
    subgraph Frontend (React + Vite + Canvas)
        UI[Mission Control Dashboard]
        Canvas[Simulation Canvas]
        Charts[Performance Charts]
        Controller[Navigation Planner]
    end
    
    subgraph Backend (FastAPI)
        API[Step Simulation API]
        Kinematics[Unicycle Robot Model]
        RewardCalc[Multiplicative Reward Engine]
    end
    
    UI -->|1. Request Action step| API
    API --> Kinematics
    Kinematics --> RewardCalc
    RewardCalc -->|2. Return Next State & Reward| UI
```

---

## Core Algorithms & Navigation Modes

The prototype contains three distinct navigation/steering policies selectable in the frontend dashboard:

### 1. Greedy Steering Policy (Target Bearings)
A simple, direct navigation policy where the robot computes the bearing to the goal at each timestep and selects the discrete angular velocity that minimizes the heading error:
$$\theta_{\text{error}} = \text{Normalize}(\theta_{\text{goal}} - \theta_{\text{robot}})$$

* **Pros**: Minimal computation, works perfectly in obstacle-free environments.
* **Cons**: No obstacle avoidance capabilities; will collide with any block in its path.

### 3. Lookahead Policy (MPC-inspired Tree Search)
A short-horizon predictive control policy. At each step, it performs a depth-first search (up to depth 8) simulating future states using the unicycle robot kinematics:
* Simulates candidate action trajectories.
* Evaluates trajectories against collision with rectangular obstacles.
* Penalizes collisions ($R = -1000$) and slight turning actions (steering smoothness penalty).
* Selects the immediate action leading to the path that ends closest to the goal while guaranteeing safety.

### 4. A* Global Pathfinding + Pure Pursuit Steering
A hybrid policy combining global path planning with local trajectory following:
* **Global Planner**: Uses the **A* Search Algorithm** on a grid overlaid on the environment canvas to compute a collision-free path from the robot's start position to the goal.
* **Local Controller**: Uses a lookahead target point along the global A* path to guide the robot's heading, adjusting steering dynamically as the robot progresses.

---

## Reinforcement Learning Environment Design

The backend serves as a step-based OpenAI Gym-style environment (`POST /step`) to simulate and train Reinforcement Learning agents (specifically suited for **D3QN - Dueling Double Deep Q-Networks**).

### State Space
At each step, the environment processes and returns:
* Robot Position: $(x, y)$
* Heading Angle: $\theta$ (in radians)
* Distance to Goal: $d$ (current) and $d_{\text{prev}}$ (previous)
* Target Bearing Error: $\theta_{\text{error}}$
* Obstacle Locations & Boundaries

### Action Space
A discrete action space consisting of **5 angular velocities** (rad/s) with a fixed forward linear velocity of $15.0 \text{ units/s}$:
| Action Index | Angular Velocity ($\omega$) | Description |
| :---: | :---: | :---: |
| **0** | $-1.5$ rad/s | Hard Left Turn |
| **1** | $-0.75$ rad/s | Soft Left Turn |
| **2** | $0.0$ rad/s | Move Straight |
| **3** | $0.75$ rad/s | Soft Right Turn |
| **4** | $1.5$ rad/s | Hard Right Turn |

### Reward Formulation
To guide RL agent convergence, the environment employs a **composite multiplicative reward** coupled with terminal rewards:

$$R = R_d \times R_{\theta}$$

* **Distance Reward ($R_d$)**: Encourages closing the distance to the target.
  $$R_d = 2.0 \times e^{-\frac{\text{distance}_{\text{current}}}{\text{distance}_{\text{previous}}}}$$
* **Heading Alignment Reward ($R_{\theta}$)**: Rewards aligning heading directly with the bearing to the goal.
  $$R_{\theta} = 5.0 - \cos(\theta_{\text{error}})$$
* **Terminal States**:
  * **Goal Reached**: $+500.0$ reward
  * **Collision / Out of Bounds**: $-100.0$ reward

---

## Installation & Getting Started

### Prerequisites
* Python 3.9+
* Node.js 18+

### Setup the Backend (FastAPI)
1. Navigate to the `Backend` directory:
   ```bash
   cd Backend
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic
   ```
3. Run the development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The backend API will run on [http://localhost:8000](http://localhost:8000).

### Setup the Frontend (React + Vite)
1. Navigate to the `Frontend` directory:
   ```bash
   cd Frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open the application in your browser at [http://localhost:5173](http://localhost:5173).

---

## Features
* **Interactive Canvas**: Drag and place obstacles or reposition the robot/goal dynamically.
* **Policy Selection**: Switch instantly between Greedy, Lookahead (MPC), and A* Navigation.
* **Real-time Telemetry**: Monitor step count, total reward, current coordinates, heading, and efficiency score.
* **Analytics Dashboard**: Live charts visualizing reward accumulation and path efficiency metrics.
