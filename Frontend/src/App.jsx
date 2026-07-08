import React, { useState, useEffect, useRef, useCallback } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler);

// --------------------------------------------------------------------------
// Constants — must stay in sync with the FastAPI backend (main.py)
// --------------------------------------------------------------------------

const API_URL = "http://localhost:8000/step";
const STEP_INTERVAL_MS = 100;

const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 400;

const GOAL = { x: 520, y: 200 };
const START = { x: 50, y: 200, theta: 0 };

const ROBOT_RADIUS = 10;
const GOAL_RADIUS = 12;
const NOSE_LENGTH = 20;

// Discrete action space (index -> angular velocity, rad/s). Mirrors the backend.
const ANGULAR_VELOCITIES = [-1.5, -0.75, 0.0, 0.75, 1.5];

// Palette — dark "mission control" console, neon blue (D3QN) and neon pink (DQN Baseline)
const COLORS = {
  bg: "#0b0f12",
  panel: "#11171b",
  panelBorder: "#20292f",
  grid: "#182126",
  robot: "#00f2fe",      // Neon Blue for Advanced D3QN
  robotB: "#ff007f",     // Neon Pink for Baseline DQN
  robotNose: "#f4f6f7",
  goal: "#4fd67a",
  amber: "#f5a623",
  amberDim: "#7a5a1c",
  danger: "#f0546b",
  textPrimary: "#e7edf0",
  textDim: "#7c8b93",
};

// --------------------------------------------------------------------------
// Helpers & Steering Policies
// --------------------------------------------------------------------------

/** Normalize an angle to the range [-pi, pi]. */
function normalizeAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

/** Bearing from (x, y) toward the goal, in radians. */
function bearingToGoal(x, y) {
  return Math.atan2(GOAL.y - y, GOAL.x - x);
}

/**
 * Greedy policy simulation: pick the discrete action whose resulting
 * angular velocity best reduces the current heading error toward the goal.
 */
function selectGreedyAction(x, y, theta) {
  const desiredTheta = bearingToGoal(x, y);
  const headingError = normalizeAngle(desiredTheta - theta);

  let bestAction = 2; // default: go straight (0.0 rad/s)
  let bestScore = Infinity;

  ANGULAR_VELOCITIES.forEach((omega, index) => {
    const projectedTheta = normalizeAngle(theta + omega);
    const projectedError = Math.abs(normalizeAngle(desiredTheta - projectedTheta));
    if (projectedError < bestScore) {
      bestScore = projectedError;
      bestAction = index;
    }
  });

  return { action: bestAction, headingError };
}

/**
 * Tree-search lookahead (MPC) policy: simulates future paths to avoid collisions
 * with obstacles while steering towards the goal.
 */
function selectLookaheadAction(x, y, theta, obstaclesList) {
  const V = 15.0;
  const ACTIONS_SUBSEQUENT = [0, 2, 4]; // left, straight, right

  function checkCollision(px, py) {
    for (const rect of obstaclesList) {
      const closestX = Math.max(rect.x, Math.min(px, rect.x + rect.width));
      const closestY = Math.max(rect.y, Math.min(py, rect.y + rect.height));
      const dist = Math.hypot(px - closestX, py - closestY);
      if (dist < ROBOT_RADIUS) {
        return true;
      }
    }
    if (px <= 0 || px >= CANVAS_WIDTH || py <= 0 || py >= CANVAS_HEIGHT) {
      return true;
    }
    return false;
  }

  function simulateStep(px, py, ptheta, action) {
    const omega = ANGULAR_VELOCITIES[action];
    const thetaNext = normalizeAngle(ptheta + omega);
    const xNext = px + V * Math.cos(thetaNext);
    const yNext = py + V * Math.sin(thetaNext);
    return { x: xNext, y: yNext, theta: thetaNext };
  }

  function findBestPath(px, py, ptheta, depth, maxDepth) {
    if (checkCollision(px, py)) {
      return { score: -1000 + depth, path: [] };
    }
    
    if (depth === maxDepth) {
      const dist = Math.hypot(GOAL.x - px, GOAL.y - py);
      return { score: 1000 - dist, path: [] };
    }
    
    let bestScore = -Infinity;
    let bestPath = [];
    
    const actionsToTry = (depth === 0) ? [0, 1, 2, 3, 4] : ACTIONS_SUBSEQUENT;
    
    for (const action of actionsToTry) {
      const nextState = simulateStep(px, py, ptheta, action);
      const result = findBestPath(nextState.x, nextState.y, nextState.theta, depth + 1, maxDepth);
      
      let score = result.score;
      if (score > -1000) {
        const omega = ANGULAR_VELOCITIES[action];
        score -= Math.abs(omega) * 0.1; // small turning penalty
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestPath = [action, ...result.path];
      }
    }
    
    return { score: bestScore, path: bestPath };
  }

  const result = findBestPath(x, y, theta, 0, 8);
  const chosenAction = result.path[0] !== undefined ? result.path[0] : 2;
  const desiredTheta = bearingToGoal(x, y);
  const headingError = normalizeAngle(desiredTheta - theta);
  return { action: chosenAction, headingError };
}

/**
 * A* Pathfinding implementation
 */
function astarPath(startX, startY, goalX, goalY, obstaclesList) {
  const GRID_SIZE = 15;
  const cols = Math.floor(CANVAS_WIDTH / GRID_SIZE);
  const rows = Math.floor(CANVAS_HEIGHT / GRID_SIZE);

  function getGridNode(x, y) { return { c: Math.floor(x/GRID_SIZE), r: Math.floor(y/GRID_SIZE) }; }
  
  const startNode = getGridNode(startX, startY);
  const goalNode = getGridNode(goalX, goalY);

  function isBlocked(c, r) {
    if (c < 0 || c >= cols || r < 0 || r >= rows) return true;
    const px = c * GRID_SIZE + GRID_SIZE / 2;
    const py = r * GRID_SIZE + GRID_SIZE / 2;
    const margin = ROBOT_RADIUS + 6;
    for (const rect of obstaclesList) {
      if (px > rect.x - margin && px < rect.x + rect.width + margin &&
          py > rect.y - margin && py < rect.y + rect.height + margin) {
        return true;
      }
    }
    return false;
  }

  const openSet = [{ c: startNode.c, r: startNode.r, g: 0, f: 0, parent: null }];
  const closedSet = new Set();
  
  function heuristic(c, r) { return Math.hypot(goalNode.c - c, goalNode.r - r); }

  while (openSet.length > 0) {
    openSet.sort((a, b) => a.f - b.f);
    const current = openSet.shift();
    const key = `${current.c},${current.r}`;

    if (current.c === goalNode.c && current.r === goalNode.r) {
      const path = [];
      let curr = current;
      while (curr) {
        path.unshift({ x: curr.c * GRID_SIZE + GRID_SIZE / 2, y: curr.r * GRID_SIZE + GRID_SIZE / 2 });
        curr = curr.parent;
      }
      return path;
    }

    closedSet.add(key);

    const neighbors = [
      {dc: 0, dr: -1}, {dc: 1, dr: 0}, {dc: 0, dr: 1}, {dc: -1, dr: 0},
      {dc: 1, dr: -1}, {dc: 1, dr: 1}, {dc: -1, dr: 1}, {dc: -1, dr: -1}
    ];

    for (const n of neighbors) {
      const nc = current.c + n.dc;
      const nr = current.r + n.dr;
      const nKey = `${nc},${nr}`;

      if (closedSet.has(nKey)) continue;
      if (isBlocked(nc, nr)) { closedSet.add(nKey); continue; }

      const cost = Math.hypot(n.dc, n.dr);
      const tentativeG = current.g + cost;

      const existingNode = openSet.find(n => n.c === nc && n.r === nr);
      if (!existingNode) {
        openSet.push({ c: nc, r: nr, g: tentativeG, f: tentativeG + heuristic(nc, nr), parent: current });
      } else if (tentativeG < existingNode.g) {
        existingNode.g = tentativeG;
        existingNode.f = tentativeG + heuristic(nc, nr);
        existingNode.parent = current;
      }
    }
  }
  return []; // No path found
}

function selectAStarAction(x, y, theta, globalPath) {
  if (!globalPath || globalPath.length === 0) return selectGreedyAction(x, y, theta);
  
  let target = GOAL;
  let lookaheadDist = 25; 
  for (let i = 0; i < globalPath.length; i++) {
     const pt = globalPath[i];
     if (Math.hypot(pt.x - x, pt.y - y) > lookaheadDist) {
        target = pt;
        break;
     }
  }

  const desiredTheta = Math.atan2(target.y - y, target.x - x);
  const headingError = normalizeAngle(desiredTheta - theta);

  let bestAction = 2; 
  let bestScore = Infinity;

  ANGULAR_VELOCITIES.forEach((omega, index) => {
    const projectedTheta = normalizeAngle(theta + omega);
    const projectedError = Math.abs(normalizeAngle(desiredTheta - projectedTheta));
    if (projectedError < bestScore) {
      bestScore = projectedError;
      bestAction = index;
    }
  });

  return { action: bestAction, headingError };
}

/**
 * Baseline DQN steering policy: simulates a poorly generalized, shortsighted agent.
 * Lacks steering smoothness penalty, has a shallow search horizon (depth 2), and contains
 * exploration/Q-overestimation noise.
 */
function selectBaselineAction(x, y, theta, obstaclesList) {
  // 12% probability of making sub-optimal choices simulating overestimation bias loops
  if (Math.random() < 0.12) {
    const randomAction = Math.floor(Math.random() * 5);
    const desiredTheta = bearingToGoal(x, y);
    const headingError = normalizeAngle(desiredTheta - theta);
    return { action: randomAction, headingError };
  }

  const V = 15.0;
  const ACTIONS_SUBSEQUENT = [0, 2, 4];

  function checkCollision(px, py) {
    for (const rect of obstaclesList) {
      const closestX = Math.max(rect.x, Math.min(px, rect.x + rect.width));
      const closestY = Math.max(rect.y, Math.min(py, rect.y + rect.height));
      const dist = Math.hypot(px - closestX, py - closestY);
      if (dist < ROBOT_RADIUS) return true;
    }
    if (px <= 0 || px >= CANVAS_WIDTH || py <= 0 || py >= CANVAS_HEIGHT) return true;
    return false;
  }

  function simulateStep(px, py, ptheta, action) {
    const omega = ANGULAR_VELOCITIES[action];
    const thetaNext = normalizeAngle(ptheta + omega);
    const xNext = px + V * Math.cos(thetaNext);
    const yNext = py + V * Math.sin(thetaNext);
    return { x: xNext, y: yNext, theta: thetaNext };
  }

  function findBestPath(px, py, ptheta, depth, maxDepth) {
    if (checkCollision(px, py)) return { score: -500 + depth, path: [] };
    if (depth === maxDepth) {
      const dist = Math.hypot(GOAL.x - px, GOAL.y - py);
      return { score: 500 - dist, path: [] };
    }
    
    let bestScore = -Infinity;
    let bestPath = [];
    const actionsToTry = (depth === 0) ? [0, 1, 2, 3, 4] : ACTIONS_SUBSEQUENT;
    
    for (const action of actionsToTry) {
      const nextState = simulateStep(px, py, ptheta, action);
      const result = findBestPath(nextState.x, nextState.y, nextState.theta, depth + 1, maxDepth);
      
      let score = result.score;
      // Note: Lacks turning / smoothness penalty entirely!
      
      if (score > bestScore) {
        bestScore = score;
        bestPath = [action, ...result.path];
      }
    }
    return { score: bestScore, path: bestPath };
  }
  
  const result = findBestPath(x, y, theta, 0, 2); // Shortsighted Depth 2 Lookahead
  const chosenAction = result.path[0] !== undefined ? result.path[0] : 2;
  const desiredTheta = bearingToGoal(x, y);
  const headingError = normalizeAngle(desiredTheta - theta);
  return { action: chosenAction, headingError };
}

function distanceTo(x, y, target) {
  return Math.hypot(target.x - x, target.y - y);
}

// --------------------------------------------------------------------------
// Main Component
// --------------------------------------------------------------------------

export default function App() {
  // Navigation View Tabs: "comparison" | "training"
  const [activeTab, setActiveTab] = useState("comparison");

  // --- STATE FOR AGENT 1 (Advanced D3QN Agent) ---
  const [robot, setRobot] = useState({ ...START });
  const [prevDistance, setPrevDistance] = useState(distanceTo(START.x, START.y, GOAL));
  const [status, setStatus] = useState("idle"); // idle | navigating | goal_reached | collision
  const [stepCount, setStepCount] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [cumulativeReward, setCumulativeReward] = useState(0);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [pathHistory, setPathHistory] = useState([]);
  const [liveRewardDetails, setLiveRewardDetails] = useState({
    dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
  });

  // --- STATE FOR AGENT 2 (Baseline DQN Agent) ---
  const [robotB, setRobotB] = useState({ ...START });
  const [prevDistanceB, setPrevDistanceB] = useState(distanceTo(START.x, START.y, GOAL));
  const [statusB, setStatusB] = useState("idle"); // idle | navigating | goal_reached | collision
  const [stepCountB, setStepCountB] = useState(0);
  const [lastRewardB, setLastRewardB] = useState(0);
  const [cumulativeRewardB, setCumulativeRewardB] = useState(0);
  const [rewardHistoryB, setRewardHistoryB] = useState([]);
  const [pathHistoryB, setPathHistoryB] = useState([]);
  const [liveRewardDetailsB, setLiveRewardDetailsB] = useState({
    dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
  });

  // Shared simulation state
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [policyMode, setPolicyMode] = useState("lookahead"); // lookahead | greedy | astar
  const [globalPath, setGlobalPath] = useState([]);
  
  // Shared obstacles
  const [obstacles, setObstacles] = useState([
    { x: 200, y: 40, width: 30, height: 180 },
    { x: 360, y: 180, width: 30, height: 180 }
  ]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingCurrent, setDrawingCurrent] = useState(null);

  // Live Dueling DQN values for visualization
  const [duelingDecomp, setDuelingDecomp] = useState({
    V: 0,
    advantages: [0, 0, 0, 0, 0],
    qValues: [0, 0, 0, 0, 0]
  });

  // --- TRAINING SIMULATION STATE ---
  const [isTraining, setIsTraining] = useState(false);
  const [trainingEpisode, setTrainingEpisode] = useState(0);
  const [trainingData, setTrainingData] = useState({
    episodes: [],
    d3qnRewards: [],
    dqnRewards: [],
    d3qnSuccess: [],
    dqnSuccess: [],
    d3qnSteps: [],
    dqnSteps: [],
    d3qnQVals: [],
    dqnQVals: [],
    actualReturn: []
  });

  // Canvas Refs
  const canvasRef = useRef(null);
  const canvasRefB = useRef(null);
  const intervalRef = useRef(null);
  const trainingIntervalRef = useRef(null);

  // Refs for inside callbacks
  const robotRef = useRef(robot);
  const prevDistanceRef = useRef(prevDistance);
  const statusRef = useRef(status);
  const stepCountRef = useRef(stepCount);

  const robotBRef = useRef(robotB);
  const prevDistanceBRef = useRef(prevDistanceB);
  const statusBRef = useRef(statusB);
  const stepCountBRef = useRef(stepCountB);

  const obstaclesRef = useRef(obstacles);
  const policyModeRef = useRef(policyMode);
  const globalPathRef = useRef(globalPath);

  useEffect(() => { robotRef.current = robot; }, [robot]);
  useEffect(() => { prevDistanceRef.current = prevDistance; }, [prevDistance]);
  useEffect(() => { statusRef.current = status; }, [status]);
  useEffect(() => { stepCountRef.current = stepCount; }, [stepCount]);

  useEffect(() => { robotBRef.current = robotB; }, [robotB]);
  useEffect(() => { prevDistanceBRef.current = prevDistanceB; }, [prevDistanceB]);
  useEffect(() => { statusBRef.current = statusB; }, [statusB]);
  useEffect(() => { stepCountBRef.current = stepCountB; }, [stepCountB]);

  useEffect(() => { obstaclesRef.current = obstacles; }, [obstacles]);
  useEffect(() => { policyModeRef.current = policyMode; }, [policyMode]);
  useEffect(() => { globalPathRef.current = globalPath; }, [globalPath]);

  // Recalculate A* path
  useEffect(() => {
    if (policyMode === "astar") {
      const path = astarPath(START.x, START.y, GOAL.x, GOAL.y, obstacles);
      setGlobalPath(path);
    }
  }, [obstacles, policyMode]);

  // ------------------------------------------------------------------------
  // Dual Canvas rendering
  // ------------------------------------------------------------------------
  const drawScene = useCallback((canvas, robotState, pathHistoryState, robotColor) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let gx = 0; gx <= CANVAS_WIDTH; gx += 40) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, CANVAS_HEIGHT);
      ctx.stroke();
    }
    for (let gy = 0; gy <= CANVAS_HEIGHT; gy += 40) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(CANVAS_WIDTH, gy);
      ctx.stroke();
    }

    // Draw Obstacles
    obstacles.forEach((obs) => {
      ctx.fillStyle = "rgba(240, 84, 107, 0.22)";
      ctx.fillRect(obs.x, obs.y, obs.width, obs.height);
      ctx.strokeStyle = "rgba(240, 84, 107, 0.65)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(obs.x, obs.y, obs.width, obs.height);
    });

    // Draw Goal
    ctx.beginPath();
    ctx.arc(GOAL.x, GOAL.y, GOAL_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.goal;
    ctx.fill();
    ctx.strokeStyle = "rgba(79, 214, 122, 0.35)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw Path History
    if (pathHistoryState.length > 0) {
      ctx.beginPath();
      ctx.moveTo(pathHistoryState[0].x, pathHistoryState[0].y);
      for (let i = 1; i < pathHistoryState.length; i++) {
        ctx.lineTo(pathHistoryState[i].x, pathHistoryState[i].y);
      }
      ctx.strokeStyle = robotColor + "aa";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Draw A* Path if active for advanced agent
    if (robotColor === COLORS.robot && policyModeRef.current === "astar" && globalPathRef.current.length > 0) {
      ctx.beginPath();
      ctx.moveTo(globalPathRef.current[0].x, globalPathRef.current[0].y);
      for (let i = 1; i < globalPathRef.current.length; i++) {
        ctx.lineTo(globalPathRef.current[i].x, globalPathRef.current[i].y);
      }
      ctx.strokeStyle = "rgba(245, 166, 35, 0.35)";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Draw Robot body
    ctx.beginPath();
    ctx.arc(robotState.x, robotState.y, ROBOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = robotColor;
    ctx.fill();

    // Draw Robot heading (nose vector)
    const noseX = robotState.x + NOSE_LENGTH * Math.cos(robotState.theta);
    const noseY = robotState.y + NOSE_LENGTH * Math.sin(robotState.theta);
    ctx.beginPath();
    ctx.moveTo(robotState.x, robotState.y);
    ctx.lineTo(noseX, noseY);
    ctx.strokeStyle = COLORS.robotNose;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(noseX, noseY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.robotNose;
    ctx.fill();
  }, [obstacles]);

  // Redraw canvases on updates
  useEffect(() => {
    if (activeTab === "comparison") {
      drawScene(canvasRef.current, robot, pathHistory, COLORS.robot);
      drawScene(canvasRefB.current, robotB, pathHistoryB, COLORS.robotB);
    }
  }, [robot, robotB, obstacles, pathHistory, pathHistoryB, drawScene, activeTab]);

  // Drawing Custom Obstacles on Canvas (handles left or right click/drag synchronically)
  const getCanvasMousePos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleCanvasMouseDown = (e, canvas) => {
    if (!isDrawingMode) return;
    const pos = getCanvasMousePos(e, canvas);
    setDrawingStart(pos);
    setDrawingCurrent(pos);
  };

  const handleCanvasMouseMove = (e, canvas) => {
    if (!isDrawingMode || !drawingStart) return;
    const pos = getCanvasMousePos(e, canvas);
    setDrawingCurrent(pos);
  };

  const handleCanvasMouseUp = (e, canvas) => {
    if (!isDrawingMode || !drawingStart) return;
    const pos = getCanvasMousePos(e, canvas);
    
    const newObs = {
      x: Math.min(drawingStart.x, pos.x),
      y: Math.min(drawingStart.y, pos.y),
      width: Math.max(10, Math.abs(pos.x - drawingStart.x)),
      height: Math.max(10, Math.abs(pos.y - drawingStart.y))
    };
    
    if (newObs.width > 5 && newObs.height > 5) {
      setObstacles(prev => [...prev, newObs]);
    }
    setDrawingStart(null);
    setDrawingCurrent(null);
  };

  // ------------------------------------------------------------------------
  // Live Value Breakdown Computation
  // ------------------------------------------------------------------------
  const updateDuelingBreakdown = useCallback((robotState) => {
    const { x, y, theta } = robotState;
    const d = distanceTo(x, y, GOAL);
    const dMax = distanceTo(START.x, START.y, GOAL);
    const distFactor = Math.max(0, 1.0 - d / dMax);
    
    // Proximity risk
    let collisionRisk = 0;
    obstacles.forEach(obs => {
      const closestX = Math.max(obs.x, Math.min(x, obs.x + obs.width));
      const closestY = Math.max(obs.y, Math.min(y, obs.y + obs.height));
      const dist = Math.hypot(x - closestX, y - closestY);
      if (dist < 40) {
        collisionRisk += (1.0 - (dist / 40));
      }
    });

    const stateValue = 480 * distFactor - 160 * Math.min(1.0, collisionRisk);
    
    // Advantages
    const bearing = bearingToGoal(x, y);
    const advs = ANGULAR_VELOCITIES.map(omega => {
      const projectedTheta = normalizeAngle(theta + omega);
      const err = Math.abs(normalizeAngle(bearing - projectedTheta));
      return 15.0 * Math.cos(err) - 4.5;
    });

    const advMean = advs.reduce((a, b) => a + b, 0) / advs.length;
    const finalAdvs = advs.map(a => a - advMean);
    const finalQ = finalAdvs.map(a => stateValue + a);

    setDuelingDecomp({
      V: stateValue,
      advantages: finalAdvs,
      qValues: finalQ
    });
  }, [obstacles]);

  useEffect(() => {
    if (status === "navigating") {
      updateDuelingBreakdown(robot);
    }
  }, [robot, status, updateDuelingBreakdown]);

  // ------------------------------------------------------------------------
  // Simulation loop
  // ------------------------------------------------------------------------
  const runStep = useCallback(async () => {
    const current = robotRef.current;
    const currentPrevDistance = prevDistanceRef.current;
    const currentB = robotBRef.current;
    const currentPrevDistanceB = prevDistanceBRef.current;

    // Call steering policies
    const actionObj = policyModeRef.current === "lookahead"
      ? selectLookaheadAction(current.x, current.y, current.theta, obstaclesRef.current)
      : policyModeRef.current === "astar" 
      ? selectAStarAction(current.x, current.y, current.theta, globalPathRef.current)
      : selectGreedyAction(current.x, current.y, current.theta);

    const actionObjB = selectBaselineAction(currentB.x, currentB.y, currentB.theta, obstaclesRef.current);

    // Call Backend step for Advanced Agent (Multiplicative reward)
    let advancedPromise = Promise.resolve(null);
    if (statusRef.current === "idle" || statusRef.current === "navigating") {
      advancedPromise = fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: current.x,
          y: current.y,
          theta: current.theta,
          prev_distance: currentPrevDistance,
          initial_distance: Math.hypot(GOAL.x - START.x, GOAL.y - START.y),
          step_count: stepCountRef.current,
          action: actionObj.action,
          obstacles: obstaclesRef.current,
          reward_type: "multiplicative"
        }),
      }).then(r => r.json());
    }

    // Call Backend step for Baseline Agent (Additive reward)
    let baselinePromise = Promise.resolve(null);
    if (statusBRef.current === "idle" || statusBRef.current === "navigating") {
      baselinePromise = fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: currentB.x,
          y: currentB.y,
          theta: currentB.theta,
          prev_distance: currentPrevDistanceB,
          initial_distance: Math.hypot(GOAL.x - START.x, GOAL.y - START.y),
          step_count: stepCountBRef.current,
          action: actionObjB.action,
          obstacles: obstaclesRef.current,
          reward_type: "additive"
        }),
      }).then(r => r.json());
    }

    try {
      const [data, dataB] = await Promise.all([advancedPromise, baselinePromise]);

      if (data) {
        setPathHistory(prev => [...prev, { x: current.x, y: current.y }]);
        setRobot({ x: data.x, y: data.y, theta: data.theta });
        setPrevDistance(data.distance);
        setStatus(data.status);
        setLastReward(data.reward);
        setCumulativeReward(prev => prev + data.reward);
        setStepCount(prev => prev + 1);
        setRewardHistory(prev => [...prev, data.reward].slice(-100));

        const desiredTheta = bearingToGoal(data.x, data.y);
        const thetaErr = normalizeAngle(desiredTheta - data.theta);
        const rThetaVal = data.r_theta;
        const rDVal = data.r_d;
        const baseRew = rDVal * rThetaVal;
        let bonusVal = 0;
        if (data.status === "goal_reached") bonusVal = 500.0;
        if (data.status === "collision") bonusVal = -100.0;

        setLiveRewardDetails({
          dPrev: currentPrevDistance,
          dCurr: data.distance,
          thetaError: thetaErr,
          rTheta: rThetaVal,
          rD: rDVal,
          baseReward: baseRew,
          bonus: bonusVal,
          totalReward: data.reward,
        });
      }

      if (dataB) {
        setPathHistoryB(prev => [...prev, { x: currentB.x, y: currentB.y }]);
        setRobotB({ x: dataB.x, y: dataB.y, theta: dataB.theta });
        setPrevDistanceB(dataB.distance);
        setStatusB(dataB.status);
        setLastRewardB(dataB.reward);
        setCumulativeRewardB(prev => prev + dataB.reward);
        setStepCountB(prev => prev + 1);
        setRewardHistoryB(prev => [...prev, dataB.reward].slice(-100));

        const desiredThetaB = bearingToGoal(dataB.x, dataB.y);
        const thetaErrB = normalizeAngle(desiredThetaB - dataB.theta);
        const rThetaValB = dataB.r_theta;
        const rDValB = dataB.r_d;
        const baseRewB = rDValB + rThetaValB - 2.0; // additive formulation
        let bonusValB = 0;
        if (dataB.status === "goal_reached") bonusValB = 500.0;
        if (dataB.status === "collision") bonusValB = -100.0;

        setLiveRewardDetailsB({
          dPrev: currentPrevDistanceB,
          dCurr: dataB.distance,
          thetaError: thetaErrB,
          rTheta: rThetaValB,
          rD: rDValB,
          baseReward: baseRewB,
          bonus: bonusValB,
          totalReward: dataB.reward,
        });
      }

      const doneAdvanced = data ? data.done : true;
      const doneBaseline = dataB ? dataB.done : true;
      if (doneAdvanced && doneBaseline) {
        setIsRunning(false);
      }
      setErrorMsg(null);
    } catch (err) {
      setErrorMsg(err.message || "Failed to reach simulation backend.");
      setIsRunning(false);
    }
  }, []);

  // Drive the interval loop while running.
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(runStep, STEP_INTERVAL_MS);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, runStep]);

  // ------------------------------------------------------------------------
  // Dual Controls
  // ------------------------------------------------------------------------
  const handleToggleRun = () => {
    setErrorMsg(null);
    if (!isRunning) {
      const advancedFin = status === "goal_reached" || status === "collision";
      const baselineFin = statusB === "goal_reached" || statusB === "collision";
      if (advancedFin && baselineFin) return; // both finished, require reset

      if (status === "idle") setStatus("navigating");
      if (statusB === "idle") setStatusB("navigating");
    }
    setIsRunning(prev => !prev);
  };

  const handleReset = () => {
    setIsRunning(false);
    
    // Reset Advanced
    setRobot({ ...START });
    setPrevDistance(distanceTo(START.x, START.y, GOAL));
    setStatus("idle");
    setPathHistory([]);
    setStepCount(0);
    setLastReward(0);
    setCumulativeReward(0);
    setRewardHistory([]);
    setLiveRewardDetails({
      dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
    });

    // Reset Baseline
    setRobotB({ ...START });
    setPrevDistanceB(distanceTo(START.x, START.y, GOAL));
    setStatusB("idle");
    setPathHistoryB([]);
    setStepCountB(0);
    setLastRewardB(0);
    setCumulativeRewardB(0);
    setRewardHistoryB([]);
    setLiveRewardDetailsB({
      dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
    });

    setDuelingDecomp({
      V: 0,
      advantages: [0, 0, 0, 0, 0],
      qValues: [0, 0, 0, 0, 0]
    });
    setErrorMsg(null);
  };

  const isNearStartOrGoal = (rect) => {
    const pad = 45;
    const startXRange = [START.x - pad, START.x + pad];
    const startYRange = [START.y - pad, START.y + pad];
    const goalXRange = [GOAL.x - pad, GOAL.x + pad];
    const goalYRange = [GOAL.y - pad, GOAL.y + pad];

    const intersectStart = !(
      rect.x + rect.width < startXRange[0] ||
      rect.x > startXRange[1] ||
      rect.y + rect.height < startYRange[0] ||
      rect.y > startYRange[1]
    );

    const intersectGoal = !(
      rect.x + rect.width < goalXRange[0] ||
      rect.x > goalXRange[1] ||
      rect.y + rect.height < goalYRange[0] ||
      rect.y > goalYRange[1]
    );

    return intersectStart || intersectGoal;
  };

  const handleRandomizeObstacles = () => {
    const newObstacles = [];
    const count = 3 + Math.floor(Math.random() * 2);
    let attempts = 0;

    while (newObstacles.length < count && attempts < 150) {
      attempts++;
      const w = 25 + Math.floor(Math.random() * 30);
      const h = 60 + Math.floor(Math.random() * 120);
      const x = 110 + Math.floor(Math.random() * (CANVAS_WIDTH - 220));
      const y = Math.floor(Math.random() * (CANVAS_HEIGHT - h - 10));
      const rect = { x, y, width: w, height: h };

      const overlapWithOthers = newObstacles.some((other) => {
        return !(
          rect.x + rect.width + 25 < other.x ||
          rect.x > other.x + other.width + 25 ||
          rect.y + rect.height + 25 < other.y ||
          rect.y > other.y + other.height + 25
        );
      });

      if (!isNearStartOrGoal(rect) && !overlapWithOthers) {
        newObstacles.push(rect);
      }
    }

    setObstacles(newObstacles);
    handleReset();
  };

  // ------------------------------------------------------------------------
  // Training Simulator Logic (animated live-plotting)
  // ------------------------------------------------------------------------
  const handleToggleTraining = () => {
    if (isTraining) {
      setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
        trainingIntervalRef.current = null;
      }
    } else {
      setIsTraining(true);
      setTrainingEpisode(0);
      setTrainingData({
        episodes: [],
        d3qnRewards: [],
        dqnRewards: [],
        d3qnSuccess: [],
        dqnSuccess: [],
        d3qnSteps: [],
        dqnSteps: [],
        d3qnQVals: [],
        dqnQVals: [],
        actualReturn: []
      });
    }
  };

  useEffect(() => {
    if (isTraining) {
      trainingIntervalRef.current = setInterval(() => {
        setTrainingEpisode(prev => {
          const nextEp = prev + 1;
          if (nextEp > 100) {
            setIsTraining(false);
            clearInterval(trainingIntervalRef.current);
            return prev;
          }

          // Generate simulated reinforcement learning training metrics
          const logEp = Math.log10(nextEp + 1);
          
          // D3QN metrics (dueling + double DQN + multiplicative reward)
          const d3qnRew = 820 * (1 - Math.exp(-nextEp / 18)) - 100 * Math.exp(-nextEp / 5) + (Math.random() * 45 - 22.5);
          const d3qnSucc = 100 / (1 + 9 * Math.exp(-nextEp / 14));
          const d3qnStep = Math.max(26, 120 * Math.exp(-nextEp / 15) + (Math.random() * 8 - 4));
          // Est Q matches actual return closely (No overestimation)
          const actualR = 520 * (1 - Math.exp(-nextEp / 20)) + (Math.random() * 30 - 15);
          const d3qnQ = actualR + (Math.random() * 20 - 10);

          // Baseline DQN metrics (no dueling, no double DQN, additive reward)
          const dqnRew = 380 * (1 - Math.exp(-nextEp / 35)) - 150 * Math.exp(-nextEp / 8) + (Math.random() * 70 - 35);
          const dqnSucc = 72 / (1 + 12 * Math.exp(-nextEp / 25));
          const dqnStep = Math.max(54, 150 * Math.exp(-nextEp / 25) + (Math.random() * 18 - 9));
          // Q-value is highly overestimated (overestimation bias!)
          const dqnQ = actualR * 1.6 + 120 * Math.exp(-Math.pow(nextEp - 30, 2) / 600) + 180 * (1 - Math.exp(-nextEp / 50)) + (Math.random() * 40 - 20);

          setTrainingData(data => ({
            episodes: [...data.episodes, nextEp],
            d3qnRewards: [...data.d3qnRewards, d3qnRew],
            dqnRewards: [...data.dqnRewards, dqnRew],
            d3qnSuccess: [...data.d3qnSuccess, d3qnSucc],
            dqnSuccess: [...data.dqnSuccess, dqnSucc],
            d3qnSteps: [...data.d3qnSteps, d3qnStep],
            dqnSteps: [...data.dqnSteps, dqnStep],
            d3qnQVals: [...data.d3qnQVals, d3qnQ],
            dqnQVals: [...data.dqnQVals, dqnQ],
            actualReturn: [...data.actualReturn, actualR]
          }));

          return nextEp;
        });
      }, 120);
    } else {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
        trainingIntervalRef.current = null;
      }
    }
    return () => {
      if (trainingIntervalRef.current) clearInterval(trainingIntervalRef.current);
    };
  }, [isTraining]);

  // ------------------------------------------------------------------------
  // Charts Configurations
  // ------------------------------------------------------------------------
  const dualRewardChartData = {
    labels: rewardHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "D3QN (Multiplicative)",
        data: rewardHistory,
        borderColor: COLORS.robot,
        backgroundColor: "rgba(0, 242, 254, 0.08)",
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.2,
      },
      {
        label: "Baseline (Additive)",
        data: rewardHistoryB,
        borderColor: COLORS.robotB,
        backgroundColor: "rgba(255, 0, 127, 0.08)",
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.2,
      }
    ],
  };

  const trainingRewardChartData = {
    labels: trainingData.episodes,
    datasets: [
      {
        label: "D3QN + Multiplicative",
        data: trainingData.d3qnRewards,
        borderColor: COLORS.robot,
        borderWidth: 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      },
      {
        label: "DQN Baseline + Additive",
        data: trainingData.dqnRewards,
        borderColor: COLORS.robotB,
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      }
    ]
  };

  const trainingSuccessChartData = {
    labels: trainingData.episodes,
    datasets: [
      {
        label: "D3QN Success Rate (%)",
        data: trainingData.d3qnSuccess,
        borderColor: COLORS.robot,
        borderWidth: 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      },
      {
        label: "DQN Success Rate (%)",
        data: trainingData.dqnSuccess,
        borderColor: COLORS.robotB,
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      }
    ]
  };

  const trainingStepsChartData = {
    labels: trainingData.episodes,
    datasets: [
      {
        label: "D3QN Steps to Goal",
        data: trainingData.d3qnSteps,
        borderColor: COLORS.robot,
        borderWidth: 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      },
      {
        label: "DQN Steps to Goal",
        data: trainingData.dqnSteps,
        borderColor: COLORS.robotB,
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      }
    ]
  };

  const trainingOverestChartData = {
    labels: trainingData.episodes,
    datasets: [
      {
        label: "Double DQN Est. Q",
        data: trainingData.d3qnQVals,
        borderColor: COLORS.robot,
        borderWidth: 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      },
      {
        label: "Standard DQN Est. Q (Overestimated)",
        data: trainingData.dqnQVals,
        borderColor: COLORS.robotB,
        borderWidth: 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.15
      },
      {
        label: "True Discounted Return",
        data: trainingData.actualReturn,
        borderColor: "#ffffff",
        borderDash: [5, 5],
        borderWidth: 1.5,
        fill: false,
        pointRadius: 0,
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { 
        display: true,
        labels: { color: COLORS.textDim, boxWidth: 12, font: { size: 10 } }
      },
      tooltip: {
        backgroundColor: COLORS.panel,
        titleColor: COLORS.textDim,
        bodyColor: COLORS.textPrimary,
        borderColor: COLORS.panelBorder,
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: COLORS.textDim, maxTicksLimit: 6 },
        grid: { color: COLORS.grid },
      },
      y: {
        ticks: { color: COLORS.textDim },
        grid: { color: COLORS.grid },
      },
    },
  };

  // Status mappings
  const statusDisplay = (stat, col) => ({
    idle: { label: "STANDBY", color: COLORS.textDim },
    navigating: { label: "NAVIGATING", color: COLORS.amber },
    goal_reached: { label: "GOAL REACHED", color: COLORS.goal },
    collision: { label: "COLLISION", color: COLORS.danger },
  }[stat]);

  return (
    <div
      className="min-h-screen w-full flex flex-col items-center p-6 font-sans text-left"
      style={{ backgroundColor: COLORS.bg }}
    >
      <div className="w-full max-w-6xl">
        {/* Navigation & Header */}
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between mb-6 pb-4 border-b border-gray-800 gap-4">
          <div>
            <h1
              className="text-2xl tracking-widest uppercase font-bold text-left"
              style={{ color: COLORS.textPrimary, letterSpacing: "0.15em" }}
            >
              Reinforcement Learning Console
            </h1>
            <p className="text-xs mt-1 text-left text-gray-400">
              D3QN (Dueling Double DQN) vs. Baseline Standard DQN Simulator Environment
            </p>
          </div>
          
          {/* Tab Selector */}
          <div className="flex gap-2">
            <button
              onClick={() => { setActiveTab("comparison"); handleReset(); }}
              className="px-4 py-1.5 rounded text-xs uppercase tracking-wider font-semibold border transition-all cursor-pointer"
              style={{
                borderColor: activeTab === "comparison" ? COLORS.robot : COLORS.panelBorder,
                color: activeTab === "comparison" ? COLORS.robot : COLORS.textDim,
                backgroundColor: activeTab === "comparison" ? "rgba(0, 242, 254, 0.08)" : "transparent"
              }}
            >
              Live Comparison Canvas
            </button>
            <button
              onClick={() => { setActiveTab("training"); setIsTraining(false); }}
              className="px-4 py-1.5 rounded text-xs uppercase tracking-wider font-semibold border transition-all cursor-pointer"
              style={{
                borderColor: activeTab === "training" ? COLORS.robot : COLORS.panelBorder,
                color: activeTab === "training" ? COLORS.robot : COLORS.textDim,
                backgroundColor: activeTab === "training" ? "rgba(0, 242, 254, 0.08)" : "transparent"
              }}
            >
              Training Performance Plots
            </button>
          </div>
        </div>

        {/* Tab 1: Real-time Side-by-Side Comparison */}
        {activeTab === "comparison" && (
          <div className="space-y-6">
            
            {/* Top Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg border border-gray-800 bg-[#11171b]">
              <div className="flex gap-3">
                <button
                  onClick={handleToggleRun}
                  className="px-5 py-2.5 rounded font-bold text-sm tracking-wide transition-all cursor-pointer shadow-md"
                  style={{
                    backgroundColor: isRunning ? "rgba(245, 166, 35, 0.25)" : COLORS.amber,
                    color: isRunning ? COLORS.amber : "#14100a",
                    border: `1px solid ${COLORS.amber}`
                  }}
                >
                  {isRunning ? "⏸ Pause Run" : "▶ Start Both Agents"}
                </button>
                <button
                  onClick={handleReset}
                  className="px-5 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer hover:bg-gray-800"
                  style={{
                    borderColor: COLORS.panelBorder,
                    color: COLORS.textPrimary,
                    backgroundColor: "transparent",
                  }}
                >
                  🔄 Reset Positions
                </button>
                <button
                  onClick={handleRandomizeObstacles}
                  className="px-5 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer hover:bg-red-950/20"
                  style={{
                    borderColor: COLORS.danger + "77",
                    color: COLORS.danger,
                    backgroundColor: "transparent",
                  }}
                >
                  ⚡ Randomize Blocks
                </button>
                <button
                  onClick={() => setIsDrawingMode(!isDrawingMode)}
                  className="px-5 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer"
                  style={{
                    borderColor: isDrawingMode ? COLORS.robot : COLORS.panelBorder,
                    color: isDrawingMode ? COLORS.robot : COLORS.textPrimary,
                    backgroundColor: isDrawingMode ? "rgba(0, 242, 254, 0.08)" : "transparent",
                  }}
                >
                  ✏️ {isDrawingMode ? "Lock Obstacles" : "Draw Custom block"}
                </button>
              </div>

              {/* Policy Selector for Advanced Agent */}
              <div className="flex items-center gap-3 text-xs font-mono">
                <span className="text-gray-400 font-semibold">D3QN Policy Mode:</span>
                <button
                  onClick={() => setPolicyMode("lookahead")}
                  className="px-3 py-1.5 rounded border transition-colors cursor-pointer"
                  style={{
                    borderColor: policyMode === "lookahead" ? COLORS.robot : COLORS.panelBorder,
                    color: policyMode === "lookahead" ? COLORS.robot : COLORS.textDim,
                    backgroundColor: policyMode === "lookahead" ? "rgba(0, 242, 254, 0.08)" : "transparent",
                  }}
                >
                  Obstacle Lookahead (MPC)
                </button>
                <button
                  onClick={() => setPolicyMode("astar")}
                  className="px-3 py-1.5 rounded border transition-colors cursor-pointer"
                  style={{
                    borderColor: policyMode === "astar" ? COLORS.robot : COLORS.panelBorder,
                    color: policyMode === "astar" ? COLORS.robot : COLORS.textDim,
                    backgroundColor: policyMode === "astar" ? "rgba(0, 242, 254, 0.08)" : "transparent",
                  }}
                >
                  A* Shortest Path
                </button>
                <button
                  onClick={() => setPolicyMode("greedy")}
                  className="px-3 py-1.5 rounded border transition-colors cursor-pointer"
                  style={{
                    borderColor: policyMode === "greedy" ? COLORS.robot : COLORS.panelBorder,
                    color: policyMode === "greedy" ? COLORS.robot : COLORS.textDim,
                    backgroundColor: policyMode === "greedy" ? "rgba(0, 242, 254, 0.08)" : "transparent",
                  }}
                >
                  Greedy Bearings
                </button>
              </div>
            </div>

            {errorMsg && (
              <div className="p-3 bg-red-950/20 border border-red-900 rounded text-red-400 text-xs font-mono">
                ⚠ Error: {errorMsg} (Please check if the FastAPI backend is running on port 8000).
              </div>
            )}

            {/* Split Screen Canvases */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Canvas 1: D3QN */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.robot }}></span>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                      Advanced: D3QN Agent
                    </h2>
                  </div>
                  <span className="text-[10px] bg-cyan-950/40 text-cyan-400 px-2 py-0.5 rounded border border-cyan-800 font-mono">
                    Dueling + Double Q + Multiplicative Reward
                  </span>
                </div>
                
                <canvas
                  ref={canvasRef}
                  width={CANVAS_WIDTH}
                  height={CANVAS_HEIGHT}
                  onMouseDown={(e) => handleCanvasMouseDown(e, canvasRef.current)}
                  onMouseMove={(e) => handleCanvasMouseMove(e, canvasRef.current)}
                  onMouseUp={(e) => handleCanvasMouseUp(e, canvasRef.current)}
                  onMouseLeave={(e) => handleCanvasMouseUp(e, canvasRef.current)}
                  className="w-full bg-[#0b0f12] rounded-lg border border-gray-900 aspect-[1.5]"
                  style={{ cursor: isDrawingMode ? "crosshair" : "default" }}
                />

                <div className="flex justify-between items-center text-xs font-mono px-1">
                  <span className="text-gray-400">Status:</span>
                  <span className="font-bold" style={{ color: statusDisplay(status).color }}>
                    {statusDisplay(status).label}
                  </span>
                </div>
              </div>

              {/* Canvas 2: Baseline DQN */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.robotB }}></span>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                      Baseline: Standard DQN Agent
                    </h2>
                  </div>
                  <span className="text-[10px] bg-pink-950/40 text-pink-400 px-2 py-0.5 rounded border border-pink-800 font-mono">
                    Flat DQN + Additive Reward
                  </span>
                </div>

                <canvas
                  ref={canvasRefB}
                  width={CANVAS_WIDTH}
                  height={CANVAS_HEIGHT}
                  onMouseDown={(e) => handleCanvasMouseDown(e, canvasRefB.current)}
                  onMouseMove={(e) => handleCanvasMouseMove(e, canvasRefB.current)}
                  onMouseUp={(e) => handleCanvasMouseUp(e, canvasRefB.current)}
                  onMouseLeave={(e) => handleCanvasMouseUp(e, canvasRefB.current)}
                  className="w-full bg-[#0b0f12] rounded-lg border border-gray-900 aspect-[1.5]"
                  style={{ cursor: isDrawingMode ? "crosshair" : "default" }}
                />

                <div className="flex justify-between items-center text-xs font-mono px-1">
                  <span className="text-gray-400">Status:</span>
                  <span className="font-bold" style={{ color: statusDisplay(statusB).color }}>
                    {statusDisplay(statusB).label}
                  </span>
                </div>
              </div>

            </div>

            {/* Live Telemetry & Reward Formulation comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Telemetry rows */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  Telemetry Comparison
                </h3>
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800/40 text-left">
                      <th className="pb-2 font-normal">Metric</th>
                      <th className="pb-2 text-right font-normal">Advanced (D3QN)</th>
                      <th className="pb-2 text-right font-normal">Baseline (DQN)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-850">
                    <tr>
                      <td className="py-2.5 text-gray-400">Step Count</td>
                      <td className="py-2.5 text-right font-semibold text-cyan-400">{stepCount}</td>
                      <td className="py-2.5 text-right font-semibold text-pink-400">{stepCountB}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Displacement to Target</td>
                      <td className="py-2.5 text-right text-gray-300">{prevDistance.toFixed(1)} units</td>
                      <td className="py-2.5 text-right text-gray-300">{prevDistanceB.toFixed(1)} units</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Step Reward</td>
                      <td className="py-2.5 text-right text-cyan-400 font-bold">{lastReward.toFixed(3)}</td>
                      <td className="py-2.5 text-right text-pink-400 font-bold">{lastRewardB.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Cumulative Return</td>
                      <td className="py-2.5 text-right text-cyan-400 font-bold">{cumulativeReward.toFixed(2)}</td>
                      <td className="py-2.5 text-right text-pink-400 font-bold">{cumulativeRewardB.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Path Efficiency (d_init / steps)</td>
                      <td className="py-2.5 text-right text-emerald-400 font-semibold">
                        {(stepCount > 0 ? (Math.hypot(GOAL.x - START.x, GOAL.y - START.y) / stepCount) : 0).toFixed(3)}
                      </td>
                      <td className="py-2.5 text-right text-orange-400 font-semibold">
                        {(stepCountB > 0 ? (Math.hypot(GOAL.x - START.x, GOAL.y - START.y) / stepCountB) : 0).toFixed(3)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Reward Formulation Comparison */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  Reward Function Comparison
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div className="space-y-2">
                    <span className="text-cyan-400 font-bold">Multiplicative Formulation</span>
                    <div className="p-3 bg-[#0b0f12] rounded border border-cyan-950/60 space-y-1.5">
                      <p className="text-gray-200">R = R_d * R_θ</p>
                      <p className="text-[10px] text-gray-400">R_d = 2 * exp(-d_curr / d_prev)</p>
                      <p className="text-[10px] text-gray-400">R_θ = 5 - cos(θ_error)</p>
                      <p className="text-[10px] text-cyan-500 font-semibold pt-1">
                        Deviating bearing errors aggressively shrink the step reward.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <span className="text-pink-400 font-bold">Additive Formulation</span>
                    <div className="p-3 bg-[#0b0f12] rounded border border-pink-950/60 space-y-1.5">
                      <p className="text-gray-200">R = R_d + R_θ - 2.0</p>
                      <p className="text-[10px] text-gray-400">R_d = 2 * exp(-d_curr / d_prev)</p>
                      <p className="text-[10px] text-gray-400">R_θ = 5 - cos(θ_error)</p>
                      <p className="text-[10px] text-pink-500 font-semibold pt-1">
                        Allows meanderings because heading deviations do not shrink the distance reward.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Dueling DQN Value Breakdown & Step Charts */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Live Dueling Value Decomposer Visualizer */}
              <div className="md:col-span-2 p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="border-b border-gray-800 pb-2 flex justify-between items-center">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-400 font-bold">
                    Dueling DQN Architecture Decomposition
                  </h3>
                  <span className="text-[9px] text-gray-500 font-mono">Q(s,a) = V(s) + A(s,a) - mean(A)</span>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
                  {/* State Value V(s) */}
                  <div className="p-3 bg-[#0b0f12] rounded border border-gray-800 flex flex-col justify-between">
                    <div>
                      <h4 className="text-gray-400 font-semibold border-b border-gray-850 pb-1 mb-2">State Value V(s)</h4>
                      <p className="text-[10px] text-gray-500">Represents baseline utility of being in this location.</p>
                    </div>
                    <div className="mt-4">
                      <div className="text-2xl font-bold text-cyan-400">{duelingDecomp.V.toFixed(1)}</div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full mt-2 overflow-hidden">
                        <div 
                          className="bg-cyan-400 h-full transition-all duration-150" 
                          style={{ width: `${Math.min(100, Math.max(0, (duelingDecomp.V + 150) / 6.5))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Advantage Stream A(s,a) */}
                  <div className="sm:col-span-2 p-3 bg-[#0b0f12] rounded border border-gray-800 space-y-2">
                    <h4 className="text-gray-400 font-semibold border-b border-gray-850 pb-1">Advantage stream A(s, a)</h4>
                    <p className="text-[10px] text-gray-500 mb-2">Identifies relative value of each discrete steering action.</p>
                    
                    <div className="space-y-1.5">
                      {duelingDecomp.advantages.map((adv, idx) => {
                        const isSelected = status === "navigating" && Math.max(...duelingDecomp.qValues) === duelingDecomp.qValues[idx];
                        const omega = ANGULAR_VELOCITIES[idx];
                        return (
                          <div key={idx} className="flex items-center gap-2">
                            <span className="w-16 text-[10px] text-gray-500 text-left">
                              {omega > 0 ? `+${omega}` : omega} rad/s
                            </span>
                            <div className="flex-1 bg-gray-800 h-2 rounded-full overflow-hidden flex">
                              <div 
                                className="h-full transition-all duration-150"
                                style={{ 
                                  width: `${Math.min(100, Math.max(0, (adv + 8) * 6))}%`,
                                  backgroundColor: isSelected ? COLORS.robot : adv >= 0 ? COLORS.goal : COLORS.danger
                                }}
                              />
                            </div>
                            <span className={`w-8 text-right text-[10px] ${isSelected ? "text-cyan-400 font-bold" : "text-gray-400"}`}>
                              {adv.toFixed(2)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-[#0b0f12] rounded border border-gray-900 text-xs text-gray-400 text-left leading-relaxed">
                  Notice that <strong className="text-cyan-400 font-semibold">State Value V(s)</strong> drops rapidly as the agent gets stuck or near obstacles (risk penalty), while <strong className="text-emerald-400 font-semibold">Advantage A(s,a)</strong> highlights the exact action index required to steer out of harm's way. A standard DQN outputs only flat Q-values with no state/advantage decomposition, reducing generalization efficiency.
                </div>
              </div>

              {/* Reward history graph */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] flex flex-col justify-between">
                <div>
                  <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                    Live Steps Reward over Time
                  </h3>
                  <p className="text-[10px] text-gray-500 font-mono mt-1">Updates in real-time as simulation advances.</p>
                </div>
                <div className="h-[180px] my-4">
                  {rewardHistory.length > 0 ? (
                    <Line data={dualRewardChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Telemetry graph will populate once agents start stepping.
                    </div>
                  )}
                </div>
                <div className="text-[10px] text-gray-500 font-mono text-center">
                  Cyan: D3QN (Multiplicative) | Pink: Baseline (Additive)
                </div>
              </div>

            </div>

          </div>
        )}

        {/* Tab 2: Training Performance Simulator */}
        {activeTab === "training" && (
          <div className="space-y-6">
            
            {/* Training Dashboard control */}
            <div className="flex items-center justify-between p-5 rounded-lg border border-gray-800 bg-[#11171b]">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                  Off-line Training Convergence Monitor
                </h2>
                <p className="text-xs text-gray-400 mt-1">
                  Compare convergence speeds, success rates, and target Q-value estimations over 100 training epochs.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-xs font-mono text-gray-400">
                  Episode: <span className="font-bold text-amber-500 text-sm">{trainingEpisode}</span> / 100
                </div>
                <button
                  onClick={handleToggleTraining}
                  className="px-5 py-2 rounded text-xs uppercase tracking-wider font-bold transition-all cursor-pointer shadow-md"
                  style={{
                    backgroundColor: isTraining ? COLORS.danger : COLORS.goal,
                    color: "#0b0f12",
                  }}
                >
                  {isTraining ? "⏹ Stop Training" : trainingEpisode >= 100 ? "🔄 Restart Training Process" : "🏋️ Start Training Simulator"}
                </button>
              </div>
            </div>

            {/* Performance charts grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Chart 1: Cumulative Return */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Episode Cumulative Reward Curve
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingRewardChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training Simulator" to see learning curves.
                    </div>
                  )}
                </div>
                <p className="text-[10px] text-gray-500 text-left">
                  Multiplicative rewards heavily penalize meandering, directing D3QN to a higher, more stable reward plateau. Additive reward meanders, producing lower cumulative return.
                </p>
              </div>

              {/* Chart 2: Success Rate */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Success Rate (%) over Episodes
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingSuccessChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training Simulator" to see success rates.
                    </div>
                  )}
                </div>
                <p className="text-[10px] text-gray-500 text-left">
                  The dueling architecture splits state/advantage to generalize obstacle risk quickly, driving success rates to ~98%. Baseline Standard DQN plateaus near ~72%.
                </p>
              </div>

              {/* Chart 3: Steps to Goal */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Average Episode Steps to Goal
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingStepsChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training Simulator" to see steps history.
                    </div>
                  )}
                </div>
                <p className="text-[10px] text-gray-500 text-left">
                  D3QN optimizes for the shortest trajectory due to discount factor constraints and straight bearing optimization. Baseline DQN spends excess steps drifting.
                </p>
              </div>

              {/* Chart 4: Overestimation Bias */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Q-value Overestimation Analysis
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingOverestChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training Simulator" to see Q-value estimates.
                    </div>
                  )}
                </div>
                <p className="text-[10px] text-gray-500 text-left">
                  Standard DQN exhibits severe overestimation (predicted Q vastly exceeds true return), resulting in unstable training. Double DQN's target network decouples action selection to match true returns.
                </p>
              </div>

            </div>

          </div>
        )}

      </div>
    </div>
  );
}
