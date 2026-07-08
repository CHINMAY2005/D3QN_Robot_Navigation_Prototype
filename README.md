# Interactive D3QN Robot Navigation Sandbox

An interactive 2D simulation environment designed for testing, visualizing, and researching robot navigation using Reinforcement Learning (RL) techniques (Double Dueling Deep Q-Networks - D3QN) alongside traditional pathfinding (A*) and steering algorithms.

The sandbox features a **FastAPI backend** that simulates unicycle kinematics and dynamically calculates request-driven reward values, paired with a feature-rich **React (Vite) frontend** providing side-by-side comparative views, active LiDAR ray-casting, live neural network decomposition streams, and training progress visualizations.

---

## Sandbox Dashboard Architecture

```mermaid
graph TD
    subgraph Frontend Client (React + Canvas + ChartJS)
        UI[Dual Agent Arena Dashboard]
        Canvas1[Advanced D3QN Canvas]
        Canvas2[Baseline DQN Canvas]
        LidarCalc[8-Directional LiDAR Solver]
        Decomp[Dueling DQN V/A Breakdown Visualizer]
        TrainSim[Offline Training Plots Simulator]
        History[State history snap stack]
    end
    
    subgraph Backend Server (FastAPI)
        API[Post /step Environment Endpoint]
        Kinematics[Unicycle Kinematics Integrator]
        RewardCalc[Reward Engine: Multiplicative vs Additive]
    end
    
    Canvas1 -->|Raycasting| LidarCalc
    Canvas2 -->|Raycasting| LidarCalc
    UI -->|1. Post State + Selected Action| API
    API --> Kinematics
    Kinematics --> RewardCalc
    RewardCalc -->|2. Return Next Coordinates & Step Reward| UI
    UI -->|Save State| History
    UI -->|Decompose State| Decomp
```

---

## Advanced Sandbox Features

### 1. Dual-Agent Real-time Benchmarking Arena
The dashboard renders two split-screen canvas views side-by-side:
* **Advanced Agent (Neon Cyan)**: Navigates using Lookahead (MPC) or Dynamic A* steering, evaluating under the **Multiplicative Reward** formulation.
* **Baseline Agent (Neon Pink)**: Simulates a shortsighted standard DQN agent (depth 2 search horizon, exploration noise, and no turning smoothness penalty) evaluating under the **Additive Reward** formulation.
* **Synchronized Environment**: Shared obstacles and goal target positions ensure a completely fair comparative sandbox. Obstacles drawn or randomized in one canvas update the other concurrently.

### 2. Multi-Robot Parallel Benchmarking
* Select **1 to 4 robots** using the dropdown control panel. Both canvases spawn the chosen number of robots concurrently.
* Compare how the D3QN lookahead and dynamic A* paths scale across multiple concurrent starting positions versus the baseline DQN model in real-time.

### 3. Drag-and-Drop Coordinate Positioning
* Reposition the start coordinates of any robot by clicking and dragging them directly on either canvas. The new starting points sync automatically across both environments.

### 4. Interactive Step-by-Step Navigation
* **Next Step**: Advances both simulation environments by a single step manually.
* **Prev Step**: Rewinds the state snapshot history (coordinates, status, reward history, step counts) using a local state history stack, enabling frame-by-frame analysis of pathing decisions.

### 5. 8-Directional LiDAR Sensor Raycasting
* Projects 8 rangefinder beams from the robot's center at $45^\circ$ offsets relative to its heading.
* Dynamically calculates intersections with obstacle boundaries and canvas boundary walls:
  * **Safe (Distance > 80)**: Semi-transparent green ray.
  * **Warning (Distance 40-80)**: Semi-transparent orange ray.
  * **Immediate Hazard (Distance < 40)**: Semi-transparent red ray.
* Displays LiDAR beams for the primary selected robot to show how the state space is modeled.

---

## Navigation & Steering Policies

### 1. Lookahead Policy (MPC-inspired Tree Search)
A short-horizon predictive control policy. At each step, it performs a depth-first search (depth 8) simulating future states using unicycle kinematics:
* Simulates candidate action trajectories.
* Evaluates trajectories against collision with rectangular obstacles.
* **Goal-Seeking Optimization (BUG FIX)**: Stops path search and returns a maximum score (`5000 - depth`) immediately if any simulated node reaches within the goal threshold. This prevents erratic oscillations and meandering near the target.
* Penalizes collisions ($R = -1000$) and turning changes (steering smoothness penalty).
* Selects the immediate action leading to the path that ends closest to the goal while guaranteeing safety.

### 2. Dynamic A* Pathfinding + Local Pure Pursuit (BUG FIX)
* **Dynamic Re-planning**: To prevent the robot from circling loops at the starting coordinate, the algorithm re-plans the shortest grid path using **A* search** from the robot's *current coordinates* at every step.
* Steers directly towards the immediate next node (`path[1]`) on the newly generated A* path, resolving the circling loops bug completely.

---

## Reinforcement Learning Environment Design

### Action Space
A discrete action space consisting of **5 angular velocities** (rad/s) with a fixed forward linear velocity of $15.0 \text{ units/s}$:
| Action Index | Angular Velocity ($\omega$) | Description |
| :---: | :---: | :---: |
| **0** | $-1.5$ rad/s | Hard Left Turn |
| **1** | $-0.75$ rad/s | Soft Left Turn |
| **2** | $0.0$ rad/s | Move Straight |
| **3** | $0.75$ rad/s | Soft Right Turn |
| **4** | $1.5$ rad/s | Hard Right Turn |

### Reward Formulations Comparison
* **Multiplicative Reward (D3QN Agent)**:
  $$R = R_d \times R_{\theta}$$
  * $R_d = 2.0 \times e^{-\frac{\text{distance}_{\text{current}}}{\text{distance}_{\text{previous}}}}$ (Progress Reward)
  * $R_{\theta} = 5.0 - \cos(\theta_{\text{error}})$ (Alignment Reward)
  * *Characteristics*: Deviating bearing errors collapse the step reward to near-zero, forcing the robot to steer directly toward the goal bearing.
* **Additive Reward (Baseline Agent)**:
  $$R = R_d + R_{\theta} - 2.0$$
  * *Characteristics*: Weakly guides the agent because heading errors do not collapse the progress reward, allowing the robot to drift or meander.

---

## Neural Network Architecture Breakdowns

### Dueling DQN Value/Advantage Decomposition
Exposes a real-time progress bar panel visualizer decomposing Q-values into State Value $V(s)$ and Action Advantages $A(s, a_i)$:
$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')\right)$$
* **State Value $V(s)$**: Represents baseline state utility (drops as the robot approaches obstacles and rises near the goal).
* **Action Advantage $A(s, a)$**: Evaluates the relative value of selecting a specific turning action versus alternative options.
* *Contrast*: Baseline DQN displays only flat $Q(s, a)$ values with no decomposition.

### Offline Training Performance Simulator Tab
Runs an animated offline training process logging learning metrics over 100 episodes:
1. **Cumulative Reward**: D3QN converges faster and higher than DQN.
2. **Success Rate**: D3QN reaches ~98% success, while standard DQN plateaus near ~72%.
3. **Average Steps to Goal**: D3QN path optimization reduces average steps significantly compared to DQN's meandering.
4. **Q-value Overestimation Bias Analysis**: Demonstrates Double DQN's stable Q-estimates tracking true returns, contrasted with Standard DQN's severe overestimation bias.

---

## Installation & Getting Started

### Backend Setup (FastAPI)
1. Navigate to the `Backend` directory:
   ```bash
   cd Backend
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install uvicorn and FastAPI dependencies:
   ```bash
   pip install fastapi uvicorn pydantic
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup (React + Vite)
1. Navigate to the `Frontend` directory:
   ```bash
   cd Frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Access the interface at [http://localhost:5173](http://localhost:5173).
