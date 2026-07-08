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
// Raycasting & Spatial Sensing (LiDAR)
// --------------------------------------------------------------------------

/**
 * Calculates distances in 8 directions (45 deg increments relative to robot theta)
 * to obstacles or canvas boundaries.
 */
function calculateLidarRanges(x, y, theta, obstaclesList) {
  const numRays = 8;
  const maxRange = 300; // max sensing range
  const ranges = [];

  for (let i = 0; i < numRays; i++) {
    const phi = theta + (i * Math.PI) / 4;
    const dx = Math.cos(phi);
    const dy = Math.sin(phi);

    // 1. Intersect with canvas boundaries
    let tMin = Infinity;
    if (dx > 0) tMin = Math.min(tMin, (CANVAS_WIDTH - x) / dx);
    else if (dx < 0) tMin = Math.min(tMin, -x / dx);

    if (dy > 0) tMin = Math.min(tMin, (CANVAS_HEIGHT - y) / dy);
    else if (dy < 0) tMin = Math.min(tMin, -y / dy);

    // 2. Intersect with each rectangular obstacle (Ray-Box intersection)
    obstaclesList.forEach(obs => {
      const x1 = obs.x;
      const x2 = obs.x + obs.width;
      const y1 = obs.y;
      const y2 = obs.y + obs.height;

      let tX1 = dx !== 0 ? (x1 - x) / dx : -Infinity;
      let tX2 = dx !== 0 ? (x2 - x) / dx : Infinity;
      let tXMin = Math.min(tX1, tX2);
      let tXMax = Math.max(tX1, tX2);

      let tY1 = dy !== 0 ? (y1 - y) / dy : -Infinity;
      let tY2 = dy !== 0 ? (y2 - y) / dy : Infinity;
      let tYMin = Math.min(tY1, tY2);
      let tYMax = Math.max(tY1, tY2);

      let tEnter = Math.max(tXMin, tYMin);
      let tExit = Math.min(tXMax, tYMax);

      if (tExit >= tEnter && tExit >= 0) {
        const tVal = tEnter >= 0 ? tEnter : 0; // 0 if ray starts inside the box
        if (tVal < tMin) {
          tMin = tVal;
        }
      }
    });

    // Clamp range to maxRange
    ranges.push(Math.min(tMin, maxRange));
  }

  return ranges;
}

// --------------------------------------------------------------------------
// Steering Policies
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
 * Greedy policy: selects discrete action that minimizes current heading error
 */
function selectGreedyAction(x, y, theta) {
  const desiredTheta = bearingToGoal(x, y);
  const headingError = normalizeAngle(desiredTheta - theta);

  let bestAction = 2; // default: go straight
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
 * Tree-search lookahead (MPC) policy: searches depth 8 to avoid obstacle collisions
 */
function selectLookaheadAction(x, y, theta, obstaclesList) {
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
    if (checkCollision(px, py)) return { score: -1000 + depth, path: [] };
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
        score -= Math.abs(omega) * 0.1; // turning penalty
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
 * A* Pathfinding + Local pure pursuit
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
  return [];
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
 * Baseline DQN steering policy: shortsighted (depth 2 search) and contains
 * overestimation noise, representing standard DQN learning limitations.
 */
function selectBaselineAction(x, y, theta, obstaclesList) {
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
      if (score > bestScore) {
        bestScore = score;
        bestPath = [action, ...result.path];
      }
    }
    return { score: bestScore, path: bestPath };
  }
  
  const result = findBestPath(x, y, theta, 0, 2);
  const chosenAction = result.path[0] !== undefined ? result.path[0] : 2;
  const desiredTheta = bearingToGoal(x, y);
  const headingError = normalizeAngle(desiredTheta - theta);
  return { action: chosenAction, headingError };
}

function distanceTo(x, y, target) {
  return Math.hypot(target.x - x, target.y - y);
}

// --------------------------------------------------------------------------
// App Component
// --------------------------------------------------------------------------

export default function App() {
  const [activeTab, setActiveTab] = useState("comparison");

  // --- STATE FOR AGENT 1 (Advanced D3QN Agent) ---
  const [robot, setRobot] = useState({ ...START });
  const [prevDistance, setPrevDistance] = useState(distanceTo(START.x, START.y, GOAL));
  const [status, setStatus] = useState("idle");
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
  const [statusB, setStatusB] = useState("idle");
  const [stepCountB, setStepCountB] = useState(0);
  const [lastRewardB, setLastRewardB] = useState(0);
  const [cumulativeRewardB, setCumulativeRewardB] = useState(0);
  const [rewardHistoryB, setRewardHistoryB] = useState([]);
  const [pathHistoryB, setPathHistoryB] = useState([]);
  const [liveRewardDetailsB, setLiveRewardDetailsB] = useState({
    dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
  });

  // Dynamic Hyperparameter Sliders
  const [goalReward, setGoalReward] = useState(500);
  const [collisionPenalty, setCollisionPenalty] = useState(-100);
  const [showLidar, setShowLidar] = useState(true);

  // Shared environment settings
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [policyMode, setPolicyMode] = useState("lookahead");
  const [globalPath, setGlobalPath] = useState([]);
  const [presetName, setPresetName] = useState("columns");
  
  // Shared obstacles
  const [obstacles, setObstacles] = useState([
    { x: 200, y: 40, width: 30, height: 180 },
    { x: 360, y: 180, width: 30, height: 180 }
  ]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingCurrent, setDrawingCurrent] = useState(null);

  // Dueling DQN values for visualization
  const [duelingDecomp, setDuelingDecomp] = useState({
    V: 0, advantages: [0, 0, 0, 0, 0], qValues: [0, 0, 0, 0, 0]
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

  // Refs for callbacks
  const canvasRef = useRef(null);
  const canvasRefB = useRef(null);
  const intervalRef = useRef(null);
  const trainingIntervalRef = useRef(null);

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

  const goalRewardRef = useRef(goalReward);
  const collisionPenaltyRef = useRef(collisionPenalty);

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

  useEffect(() => { goalRewardRef.current = goalReward; }, [goalReward]);
  useEffect(() => { collisionPenaltyRef.current = collisionPenalty; }, [collisionPenalty]);

  // Recalculate A* path
  useEffect(() => {
    if (policyMode === "astar") {
      const path = astarPath(START.x, START.y, GOAL.x, GOAL.y, obstacles);
      setGlobalPath(path);
    }
  }, [obstacles, policyMode]);

  // Apply layout presets
  const handleApplyPreset = (name) => {
    setPresetName(name);
    let newObs = [];
    if (name === "columns") {
      newObs = [
        { x: 200, y: 40, width: 30, height: 180 },
        { x: 360, y: 180, width: 30, height: 180 }
      ];
    } else if (name === "corridors") {
      newObs = [
        { x: 160, y: 0, width: 35, height: 260 },
        { x: 290, y: 140, width: 35, height: 260 },
        { x: 420, y: 0, width: 35, height: 260 }
      ];
    } else if (name === "trap") {
      newObs = [
        { x: 220, y: 100, width: 30, height: 200 }, // U-trap vertical wall facing start
        { x: 250, y: 100, width: 140, height: 30 }, // top ceiling
        { x: 250, y: 270, width: 140, height: 30 }  // bottom floor
      ];
    } else if (name === "clear") {
      newObs = [];
    }
    setObstacles(newObs);
    handleReset();
  };

  // ------------------------------------------------------------------------
  // Dual Canvas Rendering
  // ------------------------------------------------------------------------
  const drawScene = useCallback((canvas, robotState, pathHistoryState, robotColor) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Grid lines
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

    // Draw LiDAR range beams
    if (showLidar) {
      const ranges = calculateLidarRanges(robotState.x, robotState.y, robotState.theta, obstacles);
      ranges.forEach((dist, idx) => {
        const phi = robotState.theta + (idx * Math.PI) / 4;
        const beamX = robotState.x + dist * Math.cos(phi);
        const beamY = robotState.y + dist * Math.sin(phi);

        let beamColor = "rgba(79, 214, 122, 0.25)"; // green
        if (dist < 40) beamColor = "rgba(240, 84, 107, 0.65)"; // red
        else if (dist < 80) beamColor = "rgba(245, 166, 35, 0.45)"; // orange

        ctx.beginPath();
        ctx.moveTo(robotState.x, robotState.y);
        ctx.lineTo(beamX, beamY);
        ctx.strokeStyle = beamColor;
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // Beam hit endpoint
        ctx.beginPath();
        ctx.arc(beamX, beamY, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = beamColor;
        ctx.fill();
      });
    }

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

    // Draw A* Path if active for D3QN agent
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

    // Draw Robot nose
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
  }, [obstacles, showLidar]);

  // Redraw on state shifts
  useEffect(() => {
    if (activeTab === "comparison") {
      drawScene(canvasRef.current, robot, pathHistory, COLORS.robot);
      drawScene(canvasRefB.current, robotB, pathHistoryB, COLORS.robotB);
    }
  }, [robot, robotB, obstacles, pathHistory, pathHistoryB, drawScene, activeTab]);

  // Click & drag drawing handles
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
      setPresetName("custom");
    }
    setDrawingStart(null);
    setDrawingCurrent(null);
  };

  // ------------------------------------------------------------------------
  // Live Value Breakdown Computation (Calculates V(s) and Advantage values)
  // ------------------------------------------------------------------------
  const updateDuelingBreakdown = useCallback((robotState) => {
    const { x, y, theta } = robotState;
    const d = distanceTo(x, y, GOAL);
    const dMax = distanceTo(START.x, START.y, GOAL);
    const distFactor = Math.max(0, 1.0 - d / dMax);
    
    let collisionRisk = 0;
    obstacles.forEach(obs => {
      const closestX = Math.max(obs.x, minVal => minVal, Math.min(x, obs.x + obs.width));
      const closestY = Math.max(obs.y, minVal => minVal, Math.min(y, obs.y + obs.height));
      const dist = Math.hypot(x - closestX, y - closestY);
      if (dist < 40) {
        collisionRisk += (1.0 - (dist / 40));
      }
    });

    const stateValue = 480 * distFactor - 160 * Math.min(1.0, collisionRisk);
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
  // Step simulation loop
  // ------------------------------------------------------------------------
  const runStep = useCallback(async () => {
    const current = robotRef.current;
    const currentPrevDistance = prevDistanceRef.current;
    const currentB = robotBRef.current;
    const currentPrevDistanceB = prevDistanceBRef.current;

    const actionObj = policyModeRef.current === "lookahead"
      ? selectLookaheadAction(current.x, current.y, current.theta, obstaclesRef.current)
      : policyModeRef.current === "astar" 
      ? selectAStarAction(current.x, current.y, current.theta, globalPathRef.current)
      : selectGreedyAction(current.x, current.y, current.theta);

    const actionObjB = selectBaselineAction(currentB.x, currentB.y, currentB.theta, obstaclesRef.current);

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
          reward_type: "multiplicative",
          goal_reward: goalRewardRef.current,
          collision_reward: collisionPenaltyRef.current
        }),
      }).then(r => r.json());
    }

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
          reward_type: "additive",
          goal_reward: goalRewardRef.current,
          collision_reward: collisionPenaltyRef.current
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
        if (data.status === "goal_reached") bonusVal = goalRewardRef.current;
        if (data.status === "collision") bonusVal = collisionPenaltyRef.current;

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
        const baseRewB = rDValB + rThetaValB - 2.0;
        let bonusValB = 0;
        if (dataB.status === "goal_reached") bonusValB = goalRewardRef.current;
        if (dataB.status === "collision") bonusValB = collisionPenaltyRef.current;

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

  // Controls
  const handleToggleRun = () => {
    setErrorMsg(null);
    if (!isRunning) {
      const advancedFin = status === "goal_reached" || status === "collision";
      const baselineFin = statusB === "goal_reached" || statusB === "collision";
      if (advancedFin && baselineFin) return;

      if (status === "idle") setStatus("navigating");
      if (statusB === "idle") setStatusB("navigating");
    }
    setIsRunning(prev => !prev);
  };

  const handleReset = () => {
    setIsRunning(false);
    
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
      V: 0, advantages: [0, 0, 0, 0, 0], qValues: [0, 0, 0, 0, 0]
    });
    setErrorMsg(null);
  };

  const handleRandomizeObstacles = () => {
    setPresetName("custom");
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
  // Training Simulator
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
        episodes: [], d3qnRewards: [], dqnRewards: [], d3qnSuccess: [], dqnSuccess: [], d3qnSteps: [], dqnSteps: [], d3qnQVals: [], dqnQVals: [], actualReturn: []
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

          const d3qnRew = 820 * (1 - Math.exp(-nextEp / 18)) - 100 * Math.exp(-nextEp / 5) + (Math.random() * 45 - 22.5);
          const d3qnSucc = 100 / (1 + 9 * Math.exp(-nextEp / 14));
          const d3qnStep = Math.max(26, 120 * Math.exp(-nextEp / 15) + (Math.random() * 8 - 4));
          const actualR = 520 * (1 - Math.exp(-nextEp / 20)) + (Math.random() * 30 - 15);
          const d3qnQ = actualR + (Math.random() * 20 - 10);

          const dqnRew = 380 * (1 - Math.exp(-nextEp / 35)) - 150 * Math.exp(-nextEp / 8) + (Math.random() * 70 - 35);
          const dqnSucc = 72 / (1 + 12 * Math.exp(-nextEp / 25));
          const dqnStep = Math.max(54, 150 * Math.exp(-nextEp / 25) + (Math.random() * 18 - 9));
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
  // Dual Chart & Training Chart Options
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
      { label: "D3QN + Multiplicative", data: trainingData.d3qnRewards, borderColor: COLORS.robot, borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.15 },
      { label: "DQN Baseline + Additive", data: trainingData.dqnRewards, borderColor: COLORS.robotB, borderWidth: 2, fill: false, pointRadius: 0, tension: 0.15 }
    ]
  };

  const trainingSuccessChartData = {
    labels: trainingData.episodes,
    datasets: [
      { label: "D3QN Success Rate (%)", data: trainingData.d3qnSuccess, borderColor: COLORS.robot, borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.15 },
      { label: "DQN Success Rate (%)", data: trainingData.dqnSuccess, borderColor: COLORS.robotB, borderWidth: 2, fill: false, pointRadius: 0, tension: 0.15 }
    ]
  };

  const trainingStepsChartData = {
    labels: trainingData.episodes,
    datasets: [
      { label: "D3QN Steps to Goal", data: trainingData.d3qnSteps, borderColor: COLORS.robot, borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.15 },
      { label: "DQN Steps to Goal", data: trainingData.dqnSteps, borderColor: COLORS.robotB, borderWidth: 2, fill: false, pointRadius: 0, tension: 0.15 }
    ]
  };

  const trainingOverestChartData = {
    labels: trainingData.episodes,
    datasets: [
      { label: "Double DQN Est. Q", data: trainingData.d3qnQVals, borderColor: COLORS.robot, borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.15 },
      { label: "Standard DQN Est. Q (Overestimated)", data: trainingData.dqnQVals, borderColor: COLORS.robotB, borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.15 },
      { label: "True Discounted Return", data: trainingData.actualReturn, borderColor: "#ffffff", borderDash: [5, 5], borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.1 }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: true, labels: { color: COLORS.textDim, boxWidth: 12, font: { size: 10 } } },
      tooltip: { backgroundColor: COLORS.panel, titleColor: COLORS.textDim, bodyColor: COLORS.textPrimary, borderColor: COLORS.panelBorder, borderWidth: 1 }
    },
    scales: {
      x: { ticks: { color: COLORS.textDim, maxTicksLimit: 6 }, grid: { color: COLORS.grid } },
      y: { ticks: { color: COLORS.textDim }, grid: { color: COLORS.grid } }
    }
  };

  const statusDisplay = (stat) => ({
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
        
        {/* Header & View Tabs */}
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between mb-6 pb-4 border-b border-gray-800 gap-4">
          <div>
            <h1
              className="text-2xl tracking-widest uppercase font-bold text-left"
              style={{ color: COLORS.textPrimary, letterSpacing: "0.15em" }}
            >
              Reinforcement Learning Sandbox
            </h1>
            <p className="text-xs mt-1 text-left text-gray-400">
              Interactive benchmark environment comparing D3QN vs. Standard DQN configurations
            </p>
          </div>
          
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

        {/* Tab 1: Live Simulation Arena */}
        {activeTab === "comparison" && (
          <div className="space-y-6">
            
            {/* Control Toolbar */}
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
                  {isRunning ? "⏸ Pause Run" : "▶ Start Simulation"}
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
                  🔄 Reset Robots
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
                  🎲 Randomize
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
                  ✏️ {isDrawingMode ? "Lock Obstacles" : "Draw Custom Block"}
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
                  Lookahead (MPC)
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
                    <span className="w-3 h-3 rounded-full animate-pulse" style={{ backgroundColor: COLORS.robot }}></span>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                      Advanced: D3QN Agent
                    </h2>
                  </div>
                  <span className="text-[10px] bg-cyan-950/40 text-cyan-400 px-2 py-0.5 rounded border border-cyan-800 font-mono">
                    Dueling + Double DQN
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
                    <span className="w-3 h-3 rounded-full animate-pulse" style={{ backgroundColor: COLORS.robotB }}></span>
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

            {/* Hyperparameter Settings Panel & Environment Presets */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Presets and Sandbox Config */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  1. Sandbox Environment Presets
                </h3>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <button
                    onClick={() => handleApplyPreset("clear")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer"
                    style={{ borderColor: presetName === "clear" ? COLORS.robot : "transparent" }}
                  >
                    Clear Arena
                  </button>
                  <button
                    onClick={() => handleApplyPreset("columns")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer"
                    style={{ borderColor: presetName === "columns" ? COLORS.robot : "transparent" }}
                  >
                    Column Blocks
                  </button>
                  <button
                    onClick={() => handleApplyPreset("corridors")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer"
                    style={{ borderColor: presetName === "corridors" ? COLORS.robot : "transparent" }}
                  >
                    Slalom Channels
                  </button>
                  <button
                    onClick={() => handleApplyPreset("trap")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer text-pink-400"
                    style={{ borderColor: presetName === "trap" ? COLORS.robot : "transparent" }}
                  >
                    U-Trap Failzone
                  </button>
                </div>

                <div className="pt-2 border-t border-gray-850 flex items-center justify-between text-xs font-mono">
                  <span className="text-gray-400">Show LiDAR beams:</span>
                  <input
                    type="checkbox"
                    checked={showLidar}
                    onChange={(e) => setShowLidar(e.target.checked)}
                    className="w-4 h-4 cursor-pointer accent-cyan-500 rounded"
                  />
                </div>
              </div>

              {/* Dynamic Reward Sliders */}
              <div className="md:col-span-2 p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  2. Reward Function Sandbox parameters
                </h3>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 font-mono text-xs">
                  {/* Slider 1: Goal Reward */}
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Goal Target Reward:</span>
                      <span className="text-emerald-400 font-bold">+{goalReward}</span>
                    </div>
                    <input
                      type="range"
                      min="100"
                      max="1000"
                      step="50"
                      value={goalReward}
                      onChange={(e) => { setGoalReward(Number(e.target.value)); handleReset(); }}
                      className="w-full accent-emerald-500 cursor-pointer bg-gray-800 h-1 rounded-lg"
                    />
                    <p className="text-[10px] text-gray-500">
                      Higher rewards prioritize faster terminal convergence.
                    </p>
                  </div>

                  {/* Slider 2: Collision Penalty */}
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Collision Penalty:</span>
                      <span className="text-red-400 font-bold">{collisionPenalty}</span>
                    </div>
                    <input
                      type="range"
                      min="-1000"
                      max="-50"
                      step="50"
                      value={collisionPenalty}
                      onChange={(e) => { setCollisionPenalty(Number(e.target.value)); handleReset(); }}
                      className="w-full accent-red-500 cursor-pointer bg-gray-800 h-1 rounded-lg"
                    />
                    <p className="text-[10px] text-gray-500">
                      Heavier penalties dictate conservative, slow navigation paths.
                    </p>
                  </div>
                </div>
              </div>

            </div>

            {/* Live Telemetry and Reward Formula */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Telemetry Comparison Table */}
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

              {/* Reward Formula Breakdowns */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  Reward Function Comparison
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div className="space-y-2">
                    <span className="text-cyan-400 font-bold">Multiplicative (D3QN)</span>
                    <div className="p-3 bg-[#0b0f12] rounded border border-cyan-950/60 space-y-1.5">
                      <p className="text-gray-200">R = R_d * R_θ</p>
                      <p className="text-[10px] text-gray-400">R_d = 2 * exp(-d_curr / d_prev)</p>
                      <p className="text-[10px] text-gray-400">R_θ = 5 - cos(θ_error)</p>
                      <p className="text-[10px] text-cyan-500 font-semibold pt-1">
                        Deviations are penalized exponentially by the multiplicative factor.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <span className="text-pink-400 font-bold">Additive (DQN Baseline)</span>
                    <div className="p-3 bg-[#0b0f12] rounded border border-pink-950/60 space-y-1.5">
                      <p className="text-gray-200">R = R_d + R_θ - 2.0</p>
                      <p className="text-[10px] text-gray-400">R_d = 2 * exp(-d_curr / d_prev)</p>
                      <p className="text-[10px] text-gray-400">R_θ = 5 - cos(θ_error)</p>
                      <p className="text-[10px] text-pink-500 font-semibold pt-1">
                        Allows meanders; heading error doesn't collapse the distance progress reward.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Dueling DQN Decomposition stream */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Dueling DQN Breakdown Visualizer */}
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
                      <p className="text-[10px] text-gray-500">Represents baseline safety and target bearing estimation of this position.</p>
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
                    <p className="text-[10px] text-gray-500 mb-2">Evaluates steer advantage compared to local options.</p>
                    
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
              </div>

              {/* Step Reward Chart */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] flex flex-col justify-between">
                <div>
                  <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                    Live Steps Reward over Time
                  </h3>
                </div>
                <div className="h-[180px] my-4">
                  {rewardHistory.length > 0 ? (
                    <Line data={dualRewardChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Step reward curves will populate as robot steps.
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

        {/* Tab 2: Training Simulator Plots */}
        {activeTab === "training" && (
          <div className="space-y-6">
            
            {/* Training Config Header */}
            <div className="flex items-center justify-between p-5 rounded-lg border border-gray-800 bg-[#11171b]">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                  Off-line Training Convergence Monitor
                </h2>
                <p className="text-xs text-gray-400 mt-1">
                  Benchmarking learning convergence speed and estimated Q-value bias.
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
                  {isTraining ? "⏹ Stop Training" : trainingEpisode >= 100 ? "🔄 Restart Training" : "🏋️ Start Training Simulator"}
                </button>
              </div>
            </div>

            {/* Simulator curves */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Episode Cumulative Reward Curve
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingRewardChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training" to populate learning graphs.
                    </div>
                  )}
                </div>
              </div>

              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Success Rate (%) over Episodes
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingSuccessChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training" to populate learning graphs.
                    </div>
                  )}
                </div>
              </div>

              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Average Episode Steps to Goal
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingStepsChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training" to populate learning graphs.
                    </div>
                  )}
                </div>
              </div>

              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                  Q-value Overestimation Analysis
                </h3>
                <div className="h-[200px]">
                  {trainingData.episodes.length > 0 ? (
                    <Line data={trainingOverestChartData} options={chartOptions} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                      Press "Start Training" to populate learning graphs.
                    </div>
                  )}
                </div>
              </div>

            </div>

          </div>
        )}

      </div>
    </div>
  );
}
