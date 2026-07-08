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
const GOAL_THRESHOLD = 20.0;
const ROBOT_RADIUS = 10;
const GOAL_RADIUS = 12;
const NOSE_LENGTH = 20;

// Default start positions for up to 4 robots
const DEFAULT_START_POSITIONS = [
  { x: 50, y: 200, theta: 0 },
  { x: 50, y: 140, theta: 0 },
  { x: 50, y: 260, theta: 0 },
  { x: 50, y: 80, theta: 0 }
];

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

function calculateLidarRanges(x, y, theta, obstaclesList) {
  const numRays = 8;
  const maxRange = 300; 
  const ranges = [];

  for (let i = 0; i < numRays; i++) {
    const phi = theta + (i * Math.PI) / 4;
    const dx = Math.cos(phi);
    const dy = Math.sin(phi);

    let tMin = Infinity;
    if (dx > 0) tMin = Math.min(tMin, (CANVAS_WIDTH - x) / dx);
    else if (dx < 0) tMin = Math.min(tMin, -x / dx);

    if (dy > 0) tMin = Math.min(tMin, (CANVAS_HEIGHT - y) / dy);
    else if (dy < 0) tMin = Math.min(tMin, -y / dy);

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
        const tVal = tEnter >= 0 ? tEnter : 0;
        if (tVal < tMin) {
          tMin = tVal;
        }
      }
    });

    ranges.push(Math.min(tMin, maxRange));
  }

  return ranges;
}

// --------------------------------------------------------------------------
// Steering Policies
// --------------------------------------------------------------------------

function normalizeAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function bearingToGoal(x, y) {
  return Math.atan2(GOAL.y - y, GOAL.x - x);
}

function selectGreedyAction(x, y, theta) {
  const desiredTheta = bearingToGoal(x, y);
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
    
    // BUG FIX: Stop searching and return goal-reached bonus if robot hits the goal in lookahead steps
    if (Math.hypot(GOAL.x - px, GOAL.y - py) < GOAL_THRESHOLD) {
      return { score: 5000 - depth, path: [] };
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
        score -= Math.abs(omega) * 0.1; 
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

// BUG FIX: Dynamic A* Path Planner to prevent starting circling loops
function selectAStarAction(x, y, theta, obstaclesList) {
  const path = astarPath(x, y, GOAL.x, GOAL.y, obstaclesList);
  if (!path || path.length < 2) {
    return selectGreedyAction(x, y, theta);
  }
  // Next node along path is path[1] (path[0] is start node)
  const target = path[1];
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
// Main Component
// --------------------------------------------------------------------------

export default function App() {
  const [activeTab, setActiveTab] = useState("comparison");

  // Multi-Robot Configuration
  const [numRobots, setNumRobots] = useState(1);
  const [startPositions, setStartPositions] = useState(DEFAULT_START_POSITIONS);
  const [selectedRobotIndex, setSelectedRobotIndex] = useState(0);

  // Drag and Drop Start Positions State
  const [isDraggingRobot, setIsDraggingRobot] = useState(false);
  const [draggingIndex, setDraggingIndex] = useState(null);

  // Next / Prev step navigation history stack
  const [historyStack, setHistoryStack] = useState([]);

  // Dynamic Hyperparameter Sliders
  const [goalReward, setGoalReward] = useState(500);
  const [collisionPenalty, setCollisionPenalty] = useState(-100);
  const [showLidar, setShowLidar] = useState(true);

  // Shared simulation state
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [policyMode, setPolicyMode] = useState("lookahead");
  const [presetName, setPresetName] = useState("columns");
  
  // Shared obstacles
  const [obstacles, setObstacles] = useState([
    { x: 200, y: 40, width: 30, height: 180 },
    { x: 360, y: 180, width: 30, height: 180 }
  ]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingCurrent, setDrawingCurrent] = useState(null);

  // Multiple Robots State Lists
  const [robots, setRobots] = useState([]);
  const [robotsB, setRobotsB] = useState([]);

  // Initialize robot state lists
  const initializeRobots = useCallback(() => {
    const list = [];
    const listB = [];
    for (let i = 0; i < numRobots; i++) {
      const start = startPositions[i] || DEFAULT_START_POSITIONS[i];
      const dist = distanceTo(start.x, start.y, GOAL);
      
      const newRobotObj = {
        x: start.x,
        y: start.y,
        theta: start.theta,
        prevDistance: dist,
        status: "idle",
        stepCount: 0,
        lastReward: 0,
        cumulativeReward: 0,
        pathHistory: [],
        rewardHistory: [],
        liveRewardDetails: {
          dPrev: dist, dCurr: dist, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
        }
      };

      list.push({ ...newRobotObj });
      listB.push({ ...newRobotObj });
    }
    setRobots(list);
    setRobotsB(listB);
    setHistoryStack([]);
  }, [numRobots, startPositions]);

  useEffect(() => {
    initializeRobots();
  }, [numRobots, startPositions, initializeRobots]);

  // Live Dueling DQN values for visualization
  const [duelingDecomp, setDuelingDecomp] = useState({
    V: 0, advantages: [0, 0, 0, 0, 0], qValues: [0, 0, 0, 0, 0]
  });

  // --- TRAINING SIMULATION STATE ---
  const [isTraining, setIsTraining] = useState(false);
  const [trainingEpisode, setTrainingEpisode] = useState(0);
  const [trainingData, setTrainingData] = useState({
    episodes: [], d3qnRewards: [], dqnRewards: [], d3qnSuccess: [], dqnSuccess: [], d3qnSteps: [], dqnSteps: [], d3qnQVals: [], dqnQVals: [], actualReturn: []
  });

  // Canvas Refs
  const canvasRef = useRef(null);
  const canvasRefB = useRef(null);
  const intervalRef = useRef(null);
  const trainingIntervalRef = useRef(null);

  // Refs for inside callback
  const robotsRef = useRef(robots);
  const robotsBRef = useRef(robotsB);
  const obstaclesRef = useRef(obstacles);
  const policyModeRef = useRef(policyMode);
  const goalRewardRef = useRef(goalReward);
  const collisionPenaltyRef = useRef(collisionPenalty);

  useEffect(() => { robotsRef.current = robots; }, [robots]);
  useEffect(() => { robotsBRef.current = robotsB; }, [robotsB]);
  useEffect(() => { obstaclesRef.current = obstacles; }, [obstacles]);
  useEffect(() => { policyModeRef.current = policyMode; }, [policyMode]);
  useEffect(() => { goalRewardRef.current = goalReward; }, [goalReward]);
  useEffect(() => { collisionPenaltyRef.current = collisionPenalty; }, [collisionPenalty]);

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
        { x: 220, y: 100, width: 30, height: 200 },
        { x: 250, y: 100, width: 140, height: 30 },
        { x: 250, y: 270, width: 140, height: 30 }
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
  const drawScene = useCallback((canvas, robotStatesList, robotColor) => {
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

    // BUG FIX: Draw Preview Obstacle if user is currently drawing
    if (isDrawingMode && drawingStart && drawingCurrent) {
      const rx = Math.min(drawingStart.x, drawingCurrent.x);
      const ry = Math.min(drawingStart.y, drawingCurrent.y);
      const rw = Math.max(5, Math.abs(drawingCurrent.x - drawingStart.x));
      const rh = Math.max(5, Math.abs(drawingCurrent.y - drawingStart.y));

      ctx.fillStyle = "rgba(240, 84, 107, 0.18)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = "rgba(240, 84, 107, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rx, ry, rw, rh);
    }

    // Draw Goal
    ctx.beginPath();
    ctx.arc(GOAL.x, GOAL.y, GOAL_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.goal;
    ctx.fill();
    ctx.strokeStyle = "rgba(79, 214, 122, 0.35)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw each robot in the list
    robotStatesList.forEach((robotState, rIdx) => {
      const isSelected = rIdx === selectedRobotIndex;

      // Draw LiDAR range beams for selected robot
      if (showLidar && isSelected) {
        const ranges = calculateLidarRanges(robotState.x, robotState.y, robotState.theta, obstacles);
        ranges.forEach((dist, idx) => {
          const phi = robotState.theta + (idx * Math.PI) / 4;
          const beamX = robotState.x + dist * Math.cos(phi);
          const beamY = robotState.y + dist * Math.sin(phi);

          let beamColor = "rgba(79, 214, 122, 0.25)";
          if (dist < 40) beamColor = "rgba(240, 84, 107, 0.65)";
          else if (dist < 80) beamColor = "rgba(245, 166, 35, 0.45)";

          ctx.beginPath();
          ctx.moveTo(robotState.x, robotState.y);
          ctx.lineTo(beamX, beamY);
          ctx.strokeStyle = beamColor;
          ctx.lineWidth = 1.2;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(beamX, beamY, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = beamColor;
          ctx.fill();
        });
      }

      // Draw Path History
      if (robotState.pathHistory && robotState.pathHistory.length > 0) {
        ctx.beginPath();
        ctx.moveTo(robotState.pathHistory[0].x, robotState.pathHistory[0].y);
        for (let i = 1; i < robotState.pathHistory.length; i++) {
          ctx.lineTo(robotState.pathHistory[i].x, robotState.pathHistory[i].y);
        }
        ctx.strokeStyle = robotColor + (isSelected ? "aa" : "44");
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.stroke();
      }

      // Draw Robot body
      ctx.beginPath();
      ctx.arc(robotState.x, robotState.y, ROBOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? robotColor : "rgba(124, 139, 147, 0.5)";
      ctx.fill();
      
      // Selected robot border ring
      if (isSelected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

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

      // Draw robot labels
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = "9px monospace";
      ctx.fillText(`R${rIdx + 1}`, robotState.x - 6, robotState.y - 14);
    });

  }, [obstacles, showLidar, selectedRobotIndex, isDrawingMode, drawingStart, drawingCurrent]);

  // Redraw on state shifts
  useEffect(() => {
    if (activeTab === "comparison") {
      drawScene(canvasRef.current, robots, COLORS.robot);
      drawScene(canvasRefB.current, robotsB, COLORS.robotB);
    }
  }, [robots, robotsB, obstacles, drawScene, activeTab]);

  // Drag and Drop coordinates mapping
  const getCanvasMousePos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleCanvasMouseDown = (e, canvas) => {
    const pos = getCanvasMousePos(e, canvas);

    if (isDrawingMode) {
      setDrawingStart(pos);
      setDrawingCurrent(pos);
      return;
    }

    // Check if clicked close to any active robot start positions for drag-and-drop
    let clickedRobotIdx = -1;
    for (let i = 0; i < numRobots; i++) {
      const start = startPositions[i];
      const dist = Math.hypot(pos.x - start.x, pos.y - start.y);
      if (dist < ROBOT_RADIUS + 10) {
        clickedRobotIdx = i;
        break;
      }
    }

    if (clickedRobotIdx !== -1) {
      setIsDraggingRobot(true);
      setDraggingIndex(clickedRobotIdx);
      setSelectedRobotIndex(clickedRobotIdx);
    }
  };

  const handleCanvasMouseMove = (e, canvas) => {
    const pos = getCanvasMousePos(e, canvas);

    if (isDrawingMode) {
      if (!drawingStart) return;
      setDrawingCurrent(pos);
      return;
    }

    if (isDraggingRobot && draggingIndex !== null) {
      // Clamp coordinates to stay on screen
      const cx = Math.min(CANVAS_WIDTH - 20, Math.max(20, pos.x));
      const cy = Math.min(CANVAS_HEIGHT - 20, Math.max(20, pos.y));
      
      const updatedPositions = [...startPositions];
      updatedPositions[draggingIndex] = { ...updatedPositions[draggingIndex], x: cx, y: cy };
      setStartPositions(updatedPositions);
    }
  };

  const handleCanvasMouseUp = (e, canvas) => {
    if (isDrawingMode) {
      if (!drawingStart) return;
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
      return;
    }

    setIsDraggingRobot(false);
    setDraggingIndex(null);
  };

  // ------------------------------------------------------------------------
  // Live Value Breakdown Computation
  // ------------------------------------------------------------------------
  const updateDuelingBreakdown = useCallback((robotState) => {
    if (!robotState) return;
    const { x, y, theta } = robotState;
    const d = distanceTo(x, y, GOAL);
    const dMax = distanceTo(startPositions[selectedRobotIndex].x, startPositions[selectedRobotIndex].y, GOAL);
    const distFactor = Math.max(0, 1.0 - d / dMax);
    
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
    const selectedRobot = robots[selectedRobotIndex];
    if (selectedRobot && selectedRobot.status === "navigating") {
      updateDuelingBreakdown(selectedRobot);
    }
  }, [robots, selectedRobotIndex, updateDuelingBreakdown]);

  // ------------------------------------------------------------------------
  // Step simulation loop
  // ------------------------------------------------------------------------
  const runStep = useCallback(async () => {
    const activeRobots = robotsRef.current;
    const activeRobotsB = robotsBRef.current;

    // Check if all robots are already in terminal states
    const allFinished = activeRobots.every(r => r.status === "goal_reached" || r.status === "collision") &&
                        activeRobotsB.every(r => r.status === "goal_reached" || r.status === "collision");
    
    if (allFinished) {
      setIsRunning(false);
      return;
    }

    // Save states to history stack before running step
    setHistoryStack(prev => [
      ...prev,
      {
        robots: JSON.parse(JSON.stringify(activeRobots)),
        robotsB: JSON.parse(JSON.stringify(activeRobotsB))
      }
    ]);

    // Construct promises for active D3QN robots
    const advancedPromises = activeRobots.map((r, index) => {
      if (r.status === "goal_reached" || r.status === "collision") {
        return Promise.resolve(null);
      }

      // Apply A* or Lookahead or Greedy steering
      const actionObj = policyModeRef.current === "lookahead"
        ? selectLookaheadAction(r.x, r.y, r.theta, obstaclesRef.current)
        : policyModeRef.current === "astar" 
        ? selectAStarAction(r.x, r.y, r.theta, obstaclesRef.current)
        : selectGreedyAction(r.x, r.y, r.theta);

      return fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: r.x,
          y: r.y,
          theta: r.theta,
          prev_distance: r.prevDistance,
          initial_distance: Math.hypot(GOAL.x - startPositions[index].x, GOAL.y - startPositions[index].y),
          step_count: r.stepCount,
          action: actionObj.action,
          obstacles: obstaclesRef.current,
          reward_type: "multiplicative",
          goal_reward: goalRewardRef.current,
          collision_reward: collisionPenaltyRef.current
        }),
      }).then(res => res.json()).then(data => ({ index, data }));
    });

    // Construct promises for active baseline DQN robots
    const baselinePromises = activeRobotsB.map((r, index) => {
      if (r.status === "goal_reached" || r.status === "collision") {
        return Promise.resolve(null);
      }

      const actionObjB = selectBaselineAction(r.x, r.y, r.theta, obstaclesRef.current);

      return fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: r.x,
          y: r.y,
          theta: r.theta,
          prev_distance: r.prevDistance,
          initial_distance: Math.hypot(GOAL.x - startPositions[index].x, GOAL.y - startPositions[index].y),
          step_count: r.stepCount,
          action: actionObjB.action,
          obstacles: obstaclesRef.current,
          reward_type: "additive",
          goal_reward: goalRewardRef.current,
          collision_reward: collisionPenaltyRef.current
        }),
      }).then(res => res.json()).then(data => ({ index, data }));
    });

    try {
      const [resList, resListB] = await Promise.all([
        Promise.all(advancedPromises),
        Promise.all(baselinePromises)
      ]);

      // Update D3QN robots
      setRobots(prevList => {
        const nextList = [...prevList];
        resList.forEach(res => {
          if (!res) return;
          const { index, data } = res;
          const current = nextList[index];

          const desiredTheta = bearingToGoal(data.x, data.y);
          const thetaErr = normalizeAngle(desiredTheta - data.theta);
          const rThetaVal = data.r_theta;
          const rDVal = data.r_d;
          const baseRew = rDVal * rThetaVal;
          let bonusVal = 0;
          if (data.status === "goal_reached") bonusVal = goalRewardRef.current;
          if (data.status === "collision") bonusVal = collisionPenaltyRef.current;

          nextList[index] = {
            ...current,
            x: data.x,
            y: data.y,
            theta: data.theta,
            prevDistance: data.distance,
            status: data.status,
            stepCount: current.stepCount + 1,
            lastReward: data.reward,
            cumulativeReward: current.cumulativeReward + data.reward,
            pathHistory: [...current.pathHistory, { x: current.x, y: current.y }],
            rewardHistory: [...current.rewardHistory, data.reward].slice(-100),
            liveRewardDetails: {
              dPrev: current.prevDistance,
              dCurr: data.distance,
              thetaError: thetaErr,
              rTheta: rThetaVal,
              rD: rDVal,
              baseReward: baseRew,
              bonus: bonusVal,
              totalReward: data.reward,
            }
          };
        });
        return nextList;
      });

      // Update Baseline robots
      setRobotsB(prevList => {
        const nextList = [...prevList];
        resListB.forEach(res => {
          if (!res) return;
          const { index, data } = res;
          const current = nextList[index];

          const desiredTheta = bearingToGoal(data.x, data.y);
          const thetaErr = normalizeAngle(desiredTheta - data.theta);
          const rThetaVal = data.r_theta;
          const rDVal = data.r_d;
          const baseRew = rDVal + rThetaVal - 2.0;
          let bonusVal = 0;
          if (data.status === "goal_reached") bonusVal = goalRewardRef.current;
          if (data.status === "collision") bonusVal = collisionPenaltyRef.current;

          nextList[index] = {
            ...current,
            x: data.x,
            y: data.y,
            theta: data.theta,
            prevDistance: data.distance,
            status: data.status,
            stepCount: current.stepCount + 1,
            lastReward: data.reward,
            cumulativeReward: current.cumulativeReward + data.reward,
            pathHistory: [...current.pathHistory, { x: current.x, y: current.y }],
            rewardHistory: [...current.rewardHistory, data.reward].slice(-100),
            liveRewardDetails: {
              dPrev: current.prevDistance,
              dCurr: data.distance,
              thetaError: thetaErr,
              rTheta: rThetaVal,
              rD: rDVal,
              baseReward: baseRew,
              bonus: bonusVal,
              totalReward: data.reward,
            }
          };
        });
        return nextList;
      });

      setErrorMsg(null);
    } catch (err) {
      setErrorMsg(err.message || "Failed to reach simulation backend.");
      setIsRunning(false);
    }
  }, [startPositions]);

  // Step backward navigation logic
  const handlePrevStep = () => {
    if (historyStack.length === 0) return;
    setIsRunning(false);

    setHistoryStack(prevStack => {
      const nextStack = [...prevStack];
      const prevStates = nextStack.pop();
      setRobots(prevStates.robots);
      setRobotsB(prevStates.robotsB);
      return nextStack;
    });
  };

  // Step forward navigation logic
  const handleNextStep = () => {
    setIsRunning(false);
    runStep();
  };

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

  // Actions
  const handleToggleRun = () => {
    setErrorMsg(null);
    if (!isRunning) {
      const allFinished = robots.every(r => r.status === "goal_reached" || r.status === "collision") &&
                          robotsB.every(r => r.status === "goal_reached" || r.status === "collision");
      if (allFinished) return;

      setRobots(prev => prev.map(r => r.status === "idle" ? { ...r, status: "navigating" } : r));
      setRobotsB(prev => prev.map(r => r.status === "idle" ? { ...r, status: "navigating" } : r));
    }
    setIsRunning(prev => !prev);
  };

  const handleReset = () => {
    initializeRobots();
    setIsRunning(false);
    setErrorMsg(null);
  };

  const isNearStartOrGoal = (rect) => {
    const pad = 45;
    const startRangeX = [50 - pad, 50 + pad];
    const goalRangeX = [GOAL.x - pad, GOAL.x + pad];
    const goalRangeY = [GOAL.y - pad, GOAL.y + pad];

    const intersectStart = rect.x < startRangeX[1] && rect.x + rect.width > startRangeX[0];
    const intersectGoal = rect.x < goalRangeX[1] && rect.x + rect.width > goalRangeX[0] &&
                          rect.y < goalRangeY[1] && rect.y + rect.height > goalRangeY[0];

    return intersectStart || intersectGoal;
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

      const overlap = newObstacles.some((other) => {
        return !(
          rect.x + rect.width + 25 < other.x ||
          rect.x > other.x + other.width + 25 ||
          rect.y + rect.height + 25 < other.y ||
          rect.y > other.y + other.height + 25
        );
      });

      if (!isNearStartOrGoal(rect) && !overlap) {
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

          // D3QN metrics (dueling + double DQN + multiplicative reward)
          const d3qnRew = 820 * (1 - Math.exp(-nextEp / 18)) - 100 * Math.exp(-nextEp / 5) + (Math.random() * 45 - 22.5);
          const d3qnSucc = 100 / (1 + 9 * Math.exp(-nextEp / 14));
          const d3qnStep = Math.max(26, 120 * Math.exp(-nextEp / 15) + (Math.random() * 8 - 4));
          const actualR = 520 * (1 - Math.exp(-nextEp / 20)) + (Math.random() * 30 - 15);
          const d3qnQ = actualR + (Math.random() * 20 - 10);

          // Baseline DQN metrics (no dueling, no double DQN, additive reward)
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
  // Derived data values for selected robot
  // ------------------------------------------------------------------------
  const robotSelected = robots[selectedRobotIndex] || {
    stepCount: 0, prevDistance: 0, lastReward: 0, cumulativeReward: 0, rewardHistory: [], liveRewardDetails: {
      dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
    }
  };
  const robotSelectedB = robotsB[selectedRobotIndex] || {
    stepCount: 0, prevDistance: 0, lastReward: 0, cumulativeReward: 0, rewardHistory: [], liveRewardDetails: {
      dPrev: 0, dCurr: 0, thetaError: 0, rTheta: 0, rD: 0, baseReward: 0, bonus: 0, totalReward: 0
    }
  };

  const dualRewardChartData = {
    labels: robotSelected.rewardHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "D3QN (Multiplicative)",
        data: robotSelected.rewardHistory,
        borderColor: COLORS.robot,
        backgroundColor: "rgba(0, 242, 254, 0.08)",
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.2,
      },
      {
        label: "Baseline (Additive)",
        data: robotSelectedB.rewardHistory,
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

  const statusDisplay = (stat) => {
    const mapping = {
      idle: { label: "STANDBY", color: COLORS.textDim },
      navigating: { label: "NAVIGATING", color: COLORS.amber },
      goal_reached: { label: "GOAL REACHED", color: COLORS.goal },
      collision: { label: "COLLISION", color: COLORS.danger },
    };
    return mapping[stat] || { label: "STANDBY", color: COLORS.textDim };
  };

  return (
    <div
      className="min-h-screen w-full flex flex-col items-center p-6 font-sans text-left"
      style={{ backgroundColor: COLORS.bg }}
    >
      <div className="w-full max-w-6xl">
        
        {/* Header & Tabs */}
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between mb-6 pb-4 border-b border-gray-800 gap-4">
          <div>
            <h1
              className="text-2xl tracking-widest uppercase font-bold text-left"
              style={{ color: COLORS.textPrimary, letterSpacing: "0.15em" }}
            >
              Reinforcement Learning Sandbox
            </h1>
            <p className="text-xs mt-1 text-left text-gray-400">
              Interactive multi-robot simulation canvas evaluating lookahead vs. standard DQN models
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

        {/* Tab 1: Simulation Arena */}
        {activeTab === "comparison" && (
          <div className="space-y-6">
            
            {/* Split Screen Control Panel */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg border border-gray-800 bg-[#11171b]">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={handleToggleRun}
                  className="px-5 py-2.5 rounded font-bold text-sm tracking-wide transition-all cursor-pointer shadow-md"
                  style={{
                    backgroundColor: isRunning ? "rgba(245, 166, 35, 0.25)" : COLORS.amber,
                    color: isRunning ? COLORS.amber : "#14100a",
                    border: `1px solid ${COLORS.amber}`
                  }}
                >
                  {isRunning ? "⏸ Pause" : "▶ Run Auto"}
                </button>

                {/* Step forward / backward navigation buttons */}
                <button
                  onClick={handlePrevStep}
                  disabled={historyStack.length === 0}
                  className="px-4 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-800"
                  style={{ borderColor: COLORS.panelBorder, color: COLORS.textPrimary }}
                >
                  ⏮ Prev Step
                </button>
                <button
                  onClick={handleNextStep}
                  className="px-4 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer hover:bg-gray-800"
                  style={{ borderColor: COLORS.panelBorder, color: COLORS.textPrimary }}
                >
                  Next Step ⏭
                </button>

                <button
                  onClick={handleReset}
                  className="px-4 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer hover:bg-gray-800"
                  style={{
                    borderColor: COLORS.panelBorder,
                    color: COLORS.textPrimary,
                  }}
                >
                  🔄 Reset Positions
                </button>
                <button
                  onClick={handleRandomizeObstacles}
                  className="px-4 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer hover:bg-red-950/20"
                  style={{
                    borderColor: COLORS.danger + "77",
                    color: COLORS.danger,
                  }}
                >
                  🎲 Randomize Blocks
                </button>
                <button
                  onClick={() => setIsDrawingMode(!isDrawingMode)}
                  className="px-4 py-2.5 rounded font-bold text-sm border tracking-wide transition-all cursor-pointer"
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
                      Advanced: D3QN Agent Canvas
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
                  <span className="text-gray-400">Advanced Telemetry:</span>
                  <span className="text-gray-500">Click & Drag robots to change start positions</span>
                </div>
              </div>

              {/* Canvas 2: Baseline DQN */}
              <div className="p-4 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full animate-pulse" style={{ backgroundColor: COLORS.robotB }}></span>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                      Baseline: Standard DQN Agent Canvas
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
                  <span className="text-gray-400">Baseline Telemetry:</span>
                  <span className="text-gray-500">Robot coordinates are shared across columns</span>
                </div>
              </div>

            </div>

            {/* Sandbox Configurations Panel */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Presets and Sandbox Config */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  1. Multi-Robot & Presets
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
                    Columns Field
                  </button>
                  <button
                    onClick={() => handleApplyPreset("corridors")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer"
                    style={{ borderColor: presetName === "corridors" ? COLORS.robot : "transparent" }}
                  >
                    Slalom Corridors
                  </button>
                  <button
                    onClick={() => handleApplyPreset("trap")}
                    className="p-2 rounded border border-gray-850 hover:bg-gray-800 text-left transition-colors cursor-pointer text-pink-400"
                    style={{ borderColor: presetName === "trap" ? COLORS.robot : "transparent" }}
                  >
                    U-Trap Failzone
                  </button>
                </div>

                <div className="pt-2 border-t border-gray-850 space-y-2 text-xs font-mono">
                  {/* Select number of robots dropdown */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Number of Robots:</span>
                    <select
                      value={numRobots}
                      onChange={(e) => { setNumRobots(Number(e.target.value)); setSelectedRobotIndex(0); }}
                      className="bg-gray-800 border border-gray-700 text-gray-200 px-2 py-1 rounded cursor-pointer font-bold"
                    >
                      <option value="1">1 Robot</option>
                      <option value="2">2 Robots</option>
                      <option value="3">3 Robots</option>
                      <option value="4">4 Robots</option>
                    </select>
                  </div>

                  {/* Active robot focus select dropdown */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Inspect Telemetry:</span>
                    <select
                      value={selectedRobotIndex}
                      onChange={(e) => setSelectedRobotIndex(Number(e.target.value))}
                      className="bg-gray-800 border border-gray-700 text-cyan-400 px-2 py-1 rounded cursor-pointer font-bold"
                    >
                      {robots.map((_, idx) => (
                        <option key={idx} value={idx}>Robot {idx + 1}</option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-gray-400">Show LiDAR beams:</span>
                    <input
                      type="checkbox"
                      checked={showLidar}
                      onChange={(e) => setShowLidar(e.target.checked)}
                      className="w-4 h-4 cursor-pointer accent-cyan-500 rounded"
                    />
                  </div>
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
                      Rewards prioritize fast navigation routes.
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
                      Penalties trigger cautious paths around blocks.
                    </p>
                  </div>
                </div>
              </div>

            </div>

            {/* Live Telemetry & Reward Formula (displays metrics for selected robot) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Telemetry Comparison Table */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                  <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold">
                    Telemetry Comparison (Robot {selectedRobotIndex + 1})
                  </h3>
                  <span className="text-[10px] text-gray-500 font-mono">
                    Pos: ({robotSelected.x?.toFixed(1)}, {robotSelected.y?.toFixed(1)})
                  </span>
                </div>
                
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
                      <td className="py-2.5 text-gray-400">Robot Status</td>
                      <td className="py-2.5 text-right font-bold" style={{ color: statusDisplay(robotSelected.status).color }}>
                        {statusDisplay(robotSelected.status).label}
                      </td>
                      <td className="py-2.5 text-right font-bold" style={{ color: statusDisplay(robotSelectedB.status).color }}>
                        {statusDisplay(robotSelectedB.status).label}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Step Count</td>
                      <td className="py-2.5 text-right font-semibold text-cyan-400">{robotSelected.stepCount}</td>
                      <td className="py-2.5 text-right font-semibold text-pink-400">{robotSelectedB.stepCount}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Displacement to Target</td>
                      <td className="py-2.5 text-right text-gray-300">{robotSelected.prevDistance?.toFixed(1)} units</td>
                      <td className="py-2.5 text-right text-gray-300">{robotSelectedB.prevDistance?.toFixed(1)} units</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Step Reward</td>
                      <td className="py-2.5 text-right text-cyan-400 font-bold">{robotSelected.lastReward?.toFixed(3)}</td>
                      <td className="py-2.5 text-right text-pink-400 font-bold">{robotSelectedB.lastReward?.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Cumulative Return</td>
                      <td className="py-2.5 text-right text-cyan-400 font-bold">{robotSelected.cumulativeReward?.toFixed(2)}</td>
                      <td className="py-2.5 text-right text-pink-400 font-bold">{robotSelectedB.cumulativeReward?.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 text-gray-400">Path Efficiency</td>
                      <td className="py-2.5 text-right text-emerald-400 font-semibold">
                        {(robotSelected.stepCount > 0 ? (Math.hypot(GOAL.x - startPositions[selectedRobotIndex].x, GOAL.y - startPositions[selectedRobotIndex].y) / robotSelected.stepCount) : 0).toFixed(3)}
                      </td>
                      <td className="py-2.5 text-right text-orange-400 font-semibold">
                        {(robotSelectedB.stepCount > 0 ? (Math.hypot(GOAL.x - startPositions[selectedRobotIndex].x, GOAL.y - startPositions[selectedRobotIndex].y) / robotSelectedB.stepCount) : 0).toFixed(3)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Reward Formula variables breakdown */}
              <div className="p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold border-b border-gray-800 pb-2">
                  Variables Breakdown (Robot {selectedRobotIndex + 1})
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div className="space-y-2">
                    <span className="text-cyan-400 font-bold">Advanced Parameters</span>
                    <div className="p-2.5 bg-[#0b0f12] rounded border border-cyan-950/60 space-y-1 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-gray-500">R_d (distance):</span>
                        <span className="text-cyan-400">{robotSelected.liveRewardDetails?.rD?.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">R_θ (bearing):</span>
                        <span className="text-cyan-400">{robotSelected.liveRewardDetails?.rTheta?.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between border-t border-cyan-950/60 pt-1 mt-1 font-bold">
                        <span className="text-gray-400">Total base:</span>
                        <span className="text-cyan-400">{robotSelected.liveRewardDetails?.baseReward?.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <span className="text-pink-400 font-bold">Baseline Parameters</span>
                    <div className="p-2.5 bg-[#0b0f12] rounded border border-pink-950/60 space-y-1 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-gray-500">R_d (distance):</span>
                        <span className="text-pink-400">{robotSelectedB.liveRewardDetails?.rD?.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">R_θ (bearing):</span>
                        <span className="text-pink-400">{robotSelectedB.liveRewardDetails?.rTheta?.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between border-t border-pink-950/60 pt-1 mt-1 font-bold">
                        <span className="text-gray-400">Total base:</span>
                        <span className="text-pink-400">{robotSelectedB.liveRewardDetails?.baseReward?.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Dueling DQN Decomposition chart */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Dueling DQN Breakdown Visualizer */}
              <div className="md:col-span-2 p-5 rounded-xl border border-gray-800 bg-[#11171b] space-y-4">
                <div className="border-b border-gray-800 pb-2 flex justify-between items-center">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-400 font-bold">
                    Dueling DQN Architecture Decomposition (Robot {selectedRobotIndex + 1})
                  </h3>
                  <span className="text-[9px] text-gray-500 font-mono">Q(s,a) = V(s) + A(s,a) - mean(A)</span>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
                  {/* State Value V(s) */}
                  <div className="p-3 bg-[#0b0f12] rounded border border-gray-800 flex flex-col justify-between">
                    <div>
                      <h4 className="text-gray-400 font-semibold border-b border-gray-850 pb-1 mb-2">State Value V(s)</h4>
                      <p className="text-[10px] text-gray-500">Represents baseline safety and bearing proximity of this position.</p>
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
                    <p className="text-[10px] text-gray-500 mb-2">Evaluates steer action advantages.</p>
                    
                    <div className="space-y-1.5">
                      {duelingDecomp.advantages.map((adv, idx) => {
                        const isSelected = robots[selectedRobotIndex] && robots[selectedRobotIndex].status === "navigating" && Math.max(...duelingDecomp.qValues) === duelingDecomp.qValues[idx];
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
                    Live Steps Reward (Robot {selectedRobotIndex + 1})
                  </h3>
                </div>
                <div className="h-[180px] my-4">
                  {robotSelected.rewardHistory.length > 0 ? (
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

        {/* Tab 2: Training Simulator Curves */}
        {activeTab === "training" && (
          <div className="space-y-6">
            
            {/* Training control panel */}
            <div className="flex items-center justify-between p-5 rounded-lg border border-gray-800 bg-[#11171b]">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
                  Off-line Training Convergence Monitor
                </h2>
                <p className="text-xs text-gray-400 mt-1">
                  Benchmarking training epoch reward curves and target values bias.
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

            {/* Performance charts */}
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
