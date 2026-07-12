# Interactive D3QN Robot Navigation Sandbox

An interactive 2D simulation environment designed for testing, visualizing, and researching robot navigation using Reinforcement Learning (RL) techniques (Double Dueling Deep Q-Networks - D3QN) alongside traditional pathfinding (A*) and predictive lookahead steering.

### 🌐 Live Production Deployments
* **Frontend Dashboard**: [https://d3qn.vercel.app/](https://d3qn.vercel.app/) (Hosted on Vercel)
* **FastAPI Kinematics Backend**: [https://d3qn-robot-navigation-backend.onrender.com](https://d3qn-robot-navigation-backend.onrender.com) (Hosted on Render)

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
        TrainSim[Real-time Training Monitor & Log Console]
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

## Key Features

### 1. Minimalistic SVG Steering Simulator Landing Page
A dark-themed engineering landing page welcoming users with:
* Key project performance metrics panel.
* Dynamic SVG local interactive steering model simulating local collision avoidance and LiDAR scanning.
* Deep-dive visual mapping of the system architecture pipeline.

### 2. Dual-Agent Real-time Benchmarking Arena
A side-by-side split-screen canvas comparator view contrasting:
* **Advanced Agent (Neon Cyan)**: Navigates using Lookahead (MPC) or Dynamic A* steering, evaluating under the **Multiplicative Reward** formulation.
* **Baseline Agent (Neon Pink)**: Simulates a standard DQN agent, evaluating under the **Additive Reward** formulation.
* **Parallel Multi-Robot Benchmarking**: Benchmark **1 to 4 robots** simultaneously with interactive drag-and-drop start coordinates.
* **Interactive Steps Navigation**: Rewind or fast-forward simulation frames manually using local state history snaps.

### 3. Decision Field & Trajectory Footprint Visualizers
* **Live Vector Field Flow**: Toggleable background overlay showing net force directions (goal attraction + obstacle repulsion) driving the robot paths.
* **Dwell Heatmap Overlay**: Traces trajectory density footprint cells dynamically, grading cell colors from cool cyan (low occupancy) to hot red (high dwell time).
* **Collision Hotspots**: Places flashing target ring indicators at coordinates where robots collided.

### 4. Dynamic Moving Obstacles
* Obstacles slide vertically with custom speed vectors and bounce boundaries. Testing the robot's real-time predictive path-planning and collision avoidance under moving constraints.

### 5. Steering Jerk vs Steps Pareto Plot
* Tracks steering heading jerk (rad) of all paths and maps completed runs onto a **Pareto Frontier Scatter Plot**, visually proving D3QN's superior steering smoothness and path efficiency versus flat DQN.

### 6. Real-time Training Convergence Monitor
* Upgraded offline training curves simulation showing success rates, steps-to-goal, and Q-value overestimation bias.
* Includes a **scrolling virtual terminal log console** streaming metrics ( Bellman loss, exploration rate, episode returns, and collision/success statuses) in real time.

---

## Installation & Getting Started

### Local Backend Setup (FastAPI)
1. Navigate to the `Backend` directory:
   ```bash
   cd Backend
   ```
2. Set up and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Local Frontend Setup (React + Vite)
1. Navigate to the `Frontend` directory:
   ```bash
   cd Frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Create a `.env.local` file to set your local backend API URL:
   ```env
   VITE_API_URL=http://localhost:8000/step
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Access the interface at [http://localhost:5173](http://localhost:5173).

---

## Cloud Deployment

### Backend Deployment (Render)
This project is configured with a Render blueprint (`render.yaml`) for one-click setup. When deploying to **Render**:
1. Select **Blueprint** in the Render Dashboard.
2. Connect this repository. Render will declare:
   * Build command: `pip install -r requirements.txt`
   * Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   * Root Directory: `Backend`

### Frontend Deployment (Vercel)
This project uses a root-level build delegation script (`package.json`) and SPA router configuration (`vercel.json`) allowing immediate monorepo build sync. When deploying to **Vercel**:
1. Connect your repository.
2. In Vercel environment variables, set:
   * **Key**: `VITE_API_URL`
   * **Value**: `https://your-deployed-backend-url.com/step`
3. Click deploy. Vercel will build out the frontend automatically.
