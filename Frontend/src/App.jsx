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

// Palette — dark "mission control" console, amber telemetry accent.
const COLORS = {
  bg: "#0b0f12",
  panel: "#11171b",
  panelBorder: "#20292f",
  grid: "#182126",
  robot: "#3fa9f5",
  robotNose: "#f4f6f7",
  goal: "#4fd67a",
  amber: "#f5a623",
  amberDim: "#7a5a1c",
  danger: "#f0546b",
  textPrimary: "#e7edf0",
  textDim: "#7c8b93",
};

// --------------------------------------------------------------------------
// Helpers
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

function distanceTo(x, y, target) {
  return Math.hypot(target.x - x, target.y - y);
}

// --------------------------------------------------------------------------
// Component
// --------------------------------------------------------------------------


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

export default function App() {
  // Robot / simulation state
  const [robot, setRobot] = useState({ ...START });
  const [prevDistance, setPrevDistance] = useState(distanceTo(START.x, START.y, GOAL));
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | navigating | goal_reached | collision
  const [stepCount, setStepCount] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [cumulativeReward, setCumulativeReward] = useState(0);

  const [discountedGoalValue, setDiscountedGoalValue] = useState(500.0);
  const [angularConvergence, setAngularConvergence] = useState(0.0);
  const [efficiencyScore, setEfficiencyScore] = useState(0.0);

  const [rewardHistory, setRewardHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);

  const [pathHistory, setPathHistory] = useState([]);
  const [globalPath, setGlobalPath] = useState([]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingCurrent, setDrawingCurrent] = useState(null);


  // Policy mode selection: "lookahead" (obstacle avoidance) or "greedy" (direct goal steering)
  const [policyMode, setPolicyMode] = useState("lookahead");

  // Obstacles state (rectangles)
  const [obstacles, setObstacles] = useState([
    { x: 200, y: 40, width: 30, height: 180 },
    { x: 360, y: 180, width: 30, height: 180 }
  ]);

  // Live reward formula breakdown state
  const [liveRewardDetails, setLiveRewardDetails] = useState({
    dPrev: 0,
    dCurr: 0,
    thetaError: 0,
    rTheta: 0,
    rD: 0,
    baseReward: 0,
    bonus: 0,
    totalReward: 0
  });

  
  // Recalculate A* path
  useEffect(() => {
    if (policyMode === "astar") {
      const path = astarPath(START.x, START.y, GOAL.x, GOAL.y, obstacles);
      setGlobalPath(path);
    }
  }, [obstacles, policyMode]);

  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  // Mirrors of state to read inside callback without recreating interval
  const robotRef = useRef(robot);
  const prevDistanceRef = useRef(prevDistance);
  const obstaclesRef = useRef(obstacles);
  const policyModeRef = useRef(policyMode);
  const globalPathRef = useRef(globalPath);

  const stepCountRef = useRef(stepCount);
  useEffect(() => {
    stepCountRef.current = stepCount;
  }, [stepCount]);



  useEffect(() => {
    robotRef.current = robot;
  }, [robot]);

  useEffect(() => {
    prevDistanceRef.current = prevDistance;
  }, [prevDistance]);

  useEffect(() => {
    obstaclesRef.current = obstacles;
  }, [obstacles, pathHistory, drawingStart, drawingCurrent, isDrawingMode]);

  useEffect(() => {
    policyModeRef.current = policyMode;
  }, [policyMode]);

  useEffect(() => {
    globalPathRef.current = globalPath;
  }, [globalPath]);

  // ------------------------------------------------------------------------
  // Canvas rendering
  // ------------------------------------------------------------------------

  
  const handleCanvasMouseDown = (e) => {
    if (!isDrawingMode) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawingStart({ x, y });
    setDrawingCurrent({ x, y });
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDrawingMode || !drawingStart) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawingCurrent({ x, y });
  };

  const handleCanvasMouseUp = (e) => {
    if (!isDrawingMode || !drawingStart) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const newObs = {
      x: Math.min(drawingStart.x, x),
      y: Math.min(drawingStart.y, y),
      width: Math.abs(x - drawingStart.x),
      height: Math.abs(y - drawingStart.y)
    };
    
    if (newObs.width > 5 && newObs.height > 5) {
      setObstacles(prev => [...prev, newObs]);
    }
    setDrawingStart(null);
    setDrawingCurrent(null);
  };

  const drawScene = useCallback((state) => {
    const canvas = canvasRef.current;
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
      ctx.fillStyle = "rgba(240, 84, 107, 0.25)";
      ctx.fillRect(obs.x, obs.y, obs.width, obs.height);
      ctx.strokeStyle = "rgba(240, 84, 107, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(obs.x, obs.y, obs.width, obs.height);
    });


    // Draw A* Path
    if (policyModeRef.current === "astar" && globalPathRef.current.length > 0) {
      ctx.beginPath();
      ctx.moveTo(globalPathRef.current[0].x, globalPathRef.current[0].y);
      for (let i = 1; i < globalPathRef.current.length; i++) {
        ctx.lineTo(globalPathRef.current[i].x, globalPathRef.current[i].y);
      }
      ctx.strokeStyle = "rgba(245, 166, 35, 0.4)";
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw robot trail (pathHistory)
    if (pathHistory.length > 0) {
      ctx.beginPath();
      ctx.moveTo(pathHistory[0].x, pathHistory[0].y);
      for (let i = 1; i < pathHistory.length; i++) {
        ctx.lineTo(pathHistory[i].x, pathHistory[i].y);
      }
      ctx.lineTo(state.x, state.y); // Current position
      ctx.strokeStyle = "rgba(63, 169, 245, 0.6)"; // robot color but slightly transparent
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw Drawing Obstacle
    if (drawingStart && drawingCurrent) {
      const rx = Math.min(drawingStart.x, drawingCurrent.x);
      const ry = Math.min(drawingStart.y, drawingCurrent.y);
      const rw = Math.abs(drawingCurrent.x - drawingStart.x);
      const rh = Math.abs(drawingCurrent.y - drawingStart.y);
      ctx.fillStyle = "rgba(240, 84, 107, 0.5)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = COLORS.danger;
      ctx.strokeRect(rx, ry, rw, rh);
    }
    
    // Live Displacement Line
    ctx.beginPath();
    ctx.moveTo(state.x, state.y);
    ctx.lineTo(GOAL.x, GOAL.y);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Goal
    ctx.beginPath();
    ctx.arc(GOAL.x, GOAL.y, GOAL_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.goal;
    ctx.fill();
    ctx.strokeStyle = "rgba(79, 214, 122, 0.35)";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(GOAL.x, GOAL.y, GOAL_RADIUS + 6, 0, Math.PI * 2);
    ctx.stroke();

    // Robot body
    ctx.beginPath();
    ctx.arc(state.x, state.y, ROBOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.robot;
    ctx.fill();

    // Robot heading (nose vector)
    const noseX = state.x + NOSE_LENGTH * Math.cos(state.theta);
    const noseY = state.y + NOSE_LENGTH * Math.sin(state.theta);
    ctx.beginPath();
    ctx.moveTo(state.x, state.y);
    ctx.lineTo(noseX, noseY);
    ctx.strokeStyle = COLORS.robotNose;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(noseX, noseY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.robotNose;
    ctx.fill();
  }, [obstacles, pathHistory, drawingStart, drawingCurrent, isDrawingMode]);

  // Redraw whenever the robot state or obstacles change.
  useEffect(() => {
    drawScene(robot);
  }, [robot, obstacles, drawScene]);

  // ------------------------------------------------------------------------
  // Simulation step
  // ------------------------------------------------------------------------

  const runStep = useCallback(async () => {
    const current = robotRef.current;
    const currentPrevDistance = prevDistanceRef.current;

    
    const { action } = policyModeRef.current === "lookahead"
      ? selectLookaheadAction(current.x, current.y, current.theta, obstaclesRef.current)
      : policyModeRef.current === "astar" 
      ? selectAStarAction(current.x, current.y, current.theta, globalPathRef.current)
      : selectGreedyAction(current.x, current.y, current.theta);


    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: current.x,
          y: current.y,
          theta: current.theta,
          prev_distance: currentPrevDistance,
          initial_distance: Math.hypot(GOAL.x - START.x, GOAL.y - START.y),
          step_count: stepCountRef.current,
          action,
          obstacles: obstaclesRef.current,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend responded with status ${response.status}`);
      }

      const data = await response.json();

      
      setPathHistory(prev => [...prev, {x: current.x, y: current.y}]);
      setRobot({ x: data.x, y: data.y, theta: data.theta });

      setPrevDistance(data.distance);
      setStatus(data.status);
      setLastReward(data.reward);
      setCumulativeReward((prev) => prev + data.reward);
      setStepCount((prev) => prev + 1);
      setRewardHistory((prev) => [...prev, data.reward].slice(-100));
      setErrorMsg(null);

      // Use exact values from backend
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

      if (data.done) {
        setIsRunning(false);
      }
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
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isRunning, runStep]);

  // ------------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------------

  const handleToggleRun = () => {
    setErrorMsg(null);
    if (!isRunning && (status === "goal_reached" || status === "collision")) {
      return; // require reset before restarting after a terminal state
    }
    if (!isRunning && status === "idle") {
      setStatus("navigating");
    }
    setIsRunning((prev) => !prev);
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
    setDiscountedGoalValue(500.0);
    setAngularConvergence(0.0);
    setEfficiencyScore(0.0);
    setErrorMsg(null);
    setLiveRewardDetails({
      dPrev: 0,
      dCurr: 0,
      thetaError: 0,
      rTheta: 0,
      rD: 0,
      baseReward: 0,
      bonus: 0,
      totalReward: 0
    });
  };

  // Helper to ensure random obstacles don't cover the start or goal area
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
    const count = 3 + Math.floor(Math.random() * 2); // 3 or 4 obstacles
    let attempts = 0;

    while (newObstacles.length < count && attempts < 150) {
      attempts++;
      const w = 25 + Math.floor(Math.random() * 30);
      const h = 60 + Math.floor(Math.random() * 120);
      const x = 110 + Math.floor(Math.random() * (CANVAS_WIDTH - 220));
      const y = Math.floor(Math.random() * (CANVAS_HEIGHT - h - 10));
      const rect = { x, y, width: w, height: h };

      // Avoid start/goal, and make sure we don't overlap too much with existing random obstacles
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
  // Derived display values
  // ------------------------------------------------------------------------

  const statusDisplay = {
    idle: { label: "STANDBY", color: COLORS.textDim },
    navigating: { label: "NAVIGATING", color: COLORS.amber },
    goal_reached: { label: "GOAL REACHED", color: COLORS.goal },
    collision: { label: "COLLISION", color: COLORS.danger },
  }[status];

  const chartData = {
    labels: rewardHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Step reward",
        data: rewardHistory,
        borderColor: COLORS.amber,
        backgroundColor: "rgba(245, 166, 35, 0.12)",
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.25,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
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

  return (
    <div
      className="min-h-screen w-full flex items-center justify-center p-6"
      style={{ backgroundColor: COLORS.bg }}
    >
      <div className="w-full max-w-6xl">
        {/* Header */}
        <div className="flex items-baseline justify-between mb-5">
          <div>
            <h1
              className="text-xl tracking-widest uppercase font-semibold text-left"
              style={{ color: COLORS.textPrimary, letterSpacing: "0.12em" }}
            >
              Navigation Console
            </h1>
            <p className="text-xs mt-1 text-left" style={{ color: COLORS.textDim }}>
              Discrete-action greedy policy · D3QN environment with dynamic obstacles
            </p>
          </div>
          <div
            className="px-3 py-1 rounded text-xs font-mono tracking-wide border"
            style={{
              color: statusDisplay.color,
              borderColor: statusDisplay.color + "55",
              backgroundColor: statusDisplay.color + "14",
            }}
          >
            {statusDisplay.label}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Canvas panel */}
          <div
            className="lg:col-span-2 rounded-lg border p-4"
            style={{ backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder }}
          >
            <canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseUp}
              style={{ backgroundColor: COLORS.bg, maxWidth: "100%", height: "auto", cursor: isDrawingMode ? 'crosshair' : 'default' }}

              className="rounded w-full"
              />

            {/* Controls */}
            <div className="flex gap-3 mt-4 flex-wrap">
              <button
                onClick={handleToggleRun}
                disabled={status === "goal_reached" || status === "collision"}
                className="px-4 py-2 rounded font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                style={{
                  backgroundColor: isRunning ? COLORS.amberDim : COLORS.amber,
                  color: "#14100a",
                }}
              >
                {isRunning ? "Pause Agent" : "Start Agent"}
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 rounded font-medium text-sm border transition-colors cursor-pointer"
                style={{
                  borderColor: COLORS.panelBorder,
                  color: COLORS.textPrimary,
                  backgroundColor: "transparent",
                }}
              >
                Reset Environment
              </button>
              <button
                onClick={handleRandomizeObstacles}
                className="px-4 py-2 rounded font-medium text-sm border transition-colors cursor-pointer"
                style={{
                  borderColor: COLORS.danger + "55",
                  color: COLORS.danger,
                  backgroundColor: "transparent",
                }}
              >
                Randomize Obstacles
              </button>

              <button
                onClick={() => setIsDrawingMode(!isDrawingMode)}
                className="px-4 py-2 rounded font-medium text-sm border transition-colors cursor-pointer"
                style={{
                  borderColor: isDrawingMode ? COLORS.amber : COLORS.panelBorder,
                  color: isDrawingMode ? COLORS.amber : COLORS.textPrimary,
                  backgroundColor: isDrawingMode ? COLORS.amber + "15" : "transparent",
                }}
              >
                {isDrawingMode ? "Done Drawing" : "Draw Obstacles"}
              </button>
            </div>

            {/* Policy Select Toggle */}
            <div className="flex items-center gap-3 mt-4 text-xs font-mono text-left">
              <span className="text-gray-500">Agent Steering Policy:</span>
              <button
                onClick={() => setPolicyMode("lookahead")}
                className="px-3 py-1 rounded border transition-colors cursor-pointer"
                style={{
                  borderColor: policyMode === "lookahead" ? COLORS.amber : COLORS.panelBorder,
                  color: policyMode === "lookahead" ? COLORS.amber : COLORS.textDim,
                  backgroundColor: policyMode === "lookahead" ? COLORS.amber + "15" : "transparent",
                }}
              >
                Obstacle-Aware (Lookahead/MPC)
              </button>
              <button
                onClick={() => setPolicyMode("greedy")}
                className="px-3 py-1 rounded border transition-colors cursor-pointer"
                style={{
                  borderColor: policyMode === "greedy" ? COLORS.amber : COLORS.panelBorder,
                  color: policyMode === "greedy" ? COLORS.amber : COLORS.textDim,
                  backgroundColor: policyMode === "greedy" ? COLORS.amber + "15" : "transparent",
                }}
              >
                Direct Greedy (No Avoidance)
              </button>

              <button
                onClick={() => setPolicyMode("astar")}
                className="px-3 py-1 rounded border transition-colors cursor-pointer"
                style={{
                  borderColor: policyMode === "astar" ? COLORS.amber : COLORS.panelBorder,
                  color: policyMode === "astar" ? COLORS.amber : COLORS.textDim,
                  backgroundColor: policyMode === "astar" ? COLORS.amber + "15" : "transparent",
                }}
              >
                Shortest Path (A*)
              </button>
            </div>

            {errorMsg && (
              <p className="mt-3 text-xs font-mono text-left" style={{ color: COLORS.danger }}>
                ⚠ {errorMsg} — is the FastAPI backend running on :8000?
              </p>
            )}
          </div>

          {/* Telemetry panel */}
          <div className="flex flex-col gap-5">
            <div
              className="rounded-lg border p-4"
              style={{ backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder }}
            >
              <h2
                className="text-xs uppercase tracking-widest mb-3 text-left"
                style={{ color: COLORS.textDim, letterSpacing: "0.1em" }}
              >
                Telemetry
              </h2>
              <dl className="space-y-3 font-mono text-sm">
                <TelemetryRow label="Steps passed" value={stepCount} color={COLORS.textPrimary} />
                <TelemetryRow
                  label="Live Displacement"
                  value={prevDistance.toFixed(2) + " units"}
                  color={COLORS.goal}
                />
                <TelemetryRow
                  label="Discounted Goal Value"
                  value={discountedGoalValue.toFixed(2)}
                  color={COLORS.danger}
                />
                <TelemetryRow
                  label="Angular Convergence (r_theta)"
                  value={angularConvergence.toFixed(3)}
                  color={COLORS.amber}
                />
                <TelemetryRow
                  label="Theoretical Efficiency Score"
                  value={efficiencyScore.toFixed(3)}
                  color="#4fd67a"
                />
                <TelemetryRow
                  label="Step reward"
                  value={lastReward.toFixed(3)}
                  color={COLORS.amber}
                />
                <TelemetryRow
                  label="Cumulative return"
                  value={cumulativeReward.toFixed(3)}
                  color={COLORS.textPrimary}
                />
                <TelemetryRow
                  label="Distance to goal"
                  value={prevDistance.toFixed(1)}
                  color={COLORS.textDim}
                />
              </dl>
            </div>

            <div
              className="rounded-lg border p-4 flex-1 min-h-[220px]"
              style={{ backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder }}
            >
              <h2
                className="text-xs uppercase tracking-widest mb-3 text-left"
                style={{ color: COLORS.textDim, letterSpacing: "0.1em" }}
              >
                Reward over time
              </h2>
              <div style={{ height: 180 }}>
                {rewardHistory.length > 0 ? (
                  <Line data={chartData} options={chartOptions} />
                ) : (
                  <p className="text-xs text-left" style={{ color: COLORS.textDim }}>
                    Reward history will appear once the agent starts stepping.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Live Reward Calculator Breakdown */}
        <div
          className="rounded-lg border p-4 mt-5"
          style={{ backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder }}
        >
          <h2
            className="text-xs uppercase tracking-widest mb-3 text-left font-semibold"
            style={{ color: COLORS.amber, letterSpacing: "0.1em" }}
          >
            Live Reward Calculator Breakdown
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono text-left">
            <div>
              <h3 className="text-gray-300 font-semibold mb-2">Reward Formulation</h3>
              <div className="p-3 bg-[#0b0f12] rounded border border-gray-800 space-y-2">
                <p className="text-gray-300">R = R_d * R_θ</p>
                <p className="text-gray-400 text-[11px]">
                  R_d = 2 * exp(-d_curr / d_prev)
                </p>
                <p className="text-gray-400 text-[11px]">
                  R_θ = 5 - cos(θ_error)
                </p>
              </div>
            </div>
            <div>
              <h3 className="text-gray-300 font-semibold mb-2">Current Variables</h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <span className="text-gray-500">d_prev (previous dist):</span>
                <span className="text-right text-gray-200">{liveRewardDetails.dPrev.toFixed(2)}</span>
                <span className="text-gray-500">d_curr (current dist):</span>
                <span className="text-right text-gray-200">{liveRewardDetails.dCurr.toFixed(2)}</span>
                <span className="text-gray-500">θ_error (bearing error):</span>
                <span className="text-right text-gray-200">{liveRewardDetails.thetaError.toFixed(4)} rad</span>
              </div>
              <div className="mt-3 pt-2 border-t border-gray-800 grid grid-cols-2 gap-x-4 gap-y-1">
                <span className="text-gray-400 font-semibold">R_d (distance component):</span>
                <span className="text-right text-emerald-400">{liveRewardDetails.rD.toFixed(4)}</span>
                <span className="text-gray-400 font-semibold">R_θ (heading component):</span>
                <span className="text-right text-emerald-400">{liveRewardDetails.rTheta.toFixed(4)}</span>
                <span className="text-gray-400 font-semibold">Base Multiplicative:</span>
                <span className="text-right text-amber-400">{liveRewardDetails.baseReward.toFixed(4)}</span>
                <span className="text-gray-400 font-semibold">Terminal Bonus/Penalty:</span>
                <span className="text-right text-rose-400">
                  {liveRewardDetails.bonus > 0 ? `+${liveRewardDetails.bonus}` : liveRewardDetails.bonus === 0 ? "0" : liveRewardDetails.bonus}
                </span>
              </div>
              <div className="mt-2 pt-2 border-t border-gray-800 flex justify-between font-bold text-sm">
                <span className="text-amber-500">Total Step Reward (R):</span>
                <span className="text-amber-400">{liveRewardDetails.totalReward.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* RL Explanation Panel */}
        <div
          className="rounded-lg border p-6 mt-5 text-left text-sm"
          style={{ backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder, color: COLORS.textPrimary }}
        >
          <h2
            className="text-md uppercase tracking-widest mb-4 font-semibold border-b pb-2"
            style={{ color: COLORS.amber, borderColor: COLORS.panelBorder }}
          >
            Reinforcement Learning Principles & Kinematics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 leading-relaxed">
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-gray-200 mb-1">1. Markov Decision Process (MDP)</h3>
                <p className="text-xs text-gray-400">
                  The robot navigation problem is modeled as an MDP represented by the tuple (S, A, P, R). At each simulation step, the environment receives the state s_t, selects an action a_t (angular velocity), integrates kinematics to transition to the next state s_t+1 with transition probability P(s_t+1 | s_t, a_t), and receives a scalar reward r_t.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-200 mb-1">2. Action Space & Kinematics</h3>
                <p className="text-xs text-gray-400">
                  The action space is discrete: index 0 to 4 corresponding to angular velocities &omega; &in; {"{"}-1.5, -0.75, 0.0, 0.75, 1.5{"}"} rad/s. The robot moves with a constant linear velocity v = 15.0 units/s. The next state is calculated using standard unicycle kinematics:
                  <br />
                  &theta;_(t+1) = &theta;_t + &omega; * &Delta;t
                  <br />
                  x_(t+1) = x_t + v * cos(&theta;_(t+1)) * &Delta;t
                  <br />
                  y_(t+1) = y_t + v * sin(&theta;_(t+1)) * &Delta;t
                </p>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-gray-200 mb-1">3. Multiplicative Reward Design</h3>
                <p className="text-xs text-gray-400">
                  Instead of simple linear heuristics, this environment uses a composite multiplicative reward R = R_d * R_&theta;.
                  <br />
                  • <strong>Distance Reward (R_d)</strong>: Encourages progress along the line of sight. It scales exponentially based on the ratio of current distance to previous distance.
                  <br />
                  • <strong>Heading Alignment Reward (R_&theta;)</strong>: Penalizes bearing deviation. It reaches its maximum value of 6.0 when the robot's heading aligns perfectly with the goal direction.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-200 mb-1">4. Obstacle Collision Constraints</h3>
                <p className="text-xs text-gray-400">
                  Obstacles are represented as rectangles. Collision checks verify if the distance between the robot's center and the closest point on any bounding box is less than the robot's physical radius (r = 10). A collision terminates the episode (done = true) with a heavy step penalty (r = -100).
                </p>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// Small presentational subcomponent
// --------------------------------------------------------------------------

function TelemetryRow({ label, value, color }) {
  return (
    <div className="flex items-center justify-between">
      <dt style={{ color: "#7c8b93" }}>{label}</dt>
      <dd style={{ color }}>{value}</dd>
    </div>
  );
}
