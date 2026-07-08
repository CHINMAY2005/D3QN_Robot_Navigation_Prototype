import re

with open("Frontend/src/App.jsx", "r") as f:
    content = f.read()

# 1. Add astarPath and selectAStarAction before App component
helpers_code = """
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
"""
content = content.replace("export default function App() {", helpers_code + "\nexport default function App() {")

# 2. Add states to App component
state_code = """
  const [pathHistory, setPathHistory] = useState([]);
  const [globalPath, setGlobalPath] = useState([]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingCurrent, setDrawingCurrent] = useState(null);
"""
content = content.replace("const [errorMsg, setErrorMsg] = useState(null);", "const [errorMsg, setErrorMsg] = useState(null);\n" + state_code)

# 3. Add effect to recalculate A* path when obstacles or policy changes
effect_code = """
  // Recalculate A* path
  useEffect(() => {
    if (policyMode === "astar") {
      const path = astarPath(START.x, START.y, GOAL.x, GOAL.y, obstacles);
      setGlobalPath(path);
    }
  }, [obstacles, policyMode]);
"""
content = content.replace("const canvasRef = useRef(null);", effect_code + "\n  const canvasRef = useRef(null);")
content = content.replace("const policyModeRef = useRef(policyMode);", "const policyModeRef = useRef(policyMode);\n  const globalPathRef = useRef(globalPath);\n")
content = content.replace("useEffect(() => {\n    policyModeRef.current = policyMode;\n  }, [policyMode]);", "useEffect(() => {\n    policyModeRef.current = policyMode;\n  }, [policyMode]);\n\n  useEffect(() => {\n    globalPathRef.current = globalPath;\n  }, [globalPath]);")

# 4. Canvas event handlers
canvas_handlers_code = """
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
"""
content = content.replace("const drawScene = useCallback((state) => {", canvas_handlers_code + "\n  const drawScene = useCallback((state) => {")

# Add drawing handlers to <canvas>
content = content.replace("""<canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}""", """<canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseUp}
              style={{ backgroundColor: COLORS.bg, maxWidth: "100%", height: "auto", cursor: isDrawingMode ? 'crosshair' : 'default' }}
""")
content = content.replace('style={{ backgroundColor: COLORS.bg, maxWidth: "100%", height: "auto" }}\n            />', '/>')

# 5. Drawing the paths and live displacement in drawScene
draw_paths_code = """
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
"""
content = content.replace("    // Goal", draw_paths_code + "\n    // Goal")
content = content.replace("  }, [obstacles]);", "  }, [obstacles, pathHistory, drawingStart, drawingCurrent, isDrawingMode]);")

# 6. runStep update policy selection and pathHistory
run_step_code = """
    const { action } = policyModeRef.current === "lookahead"
      ? selectLookaheadAction(current.x, current.y, current.theta, obstaclesRef.current)
      : policyModeRef.current === "astar" 
      ? selectAStarAction(current.x, current.y, current.theta, globalPathRef.current)
      : selectGreedyAction(current.x, current.y, current.theta);
"""
content = re.sub(r'const \{ action \} = policyModeRef\.current === "lookahead"[\s\S]*?: selectGreedyAction\(current\.x, current\.y, current\.theta\);', run_step_code, content)

path_history_append_code = """
      setPathHistory(prev => [...prev, {x: current.x, y: current.y}]);
      setRobot({ x: data.x, y: data.y, theta: data.theta });
"""
content = content.replace("setRobot({ x: data.x, y: data.y, theta: data.theta });", path_history_append_code)

# 7. handleReset clear pathHistory
content = content.replace("setStatus(\"idle\");", "setStatus(\"idle\");\n    setPathHistory([]);")

# 8. UI Buttons
drawing_toggle_button = """
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
"""
content = content.replace("Randomize Obstacles\n              </button>\n            </div>", "Randomize Obstacles\n              </button>\n" + drawing_toggle_button + "            </div>")

astar_policy_button = """
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
"""
content = content.replace("Direct Greedy (No Avoidance)\n              </button>\n            </div>", "Direct Greedy (No Avoidance)\n              </button>\n" + astar_policy_button + "            </div>")

# 9. Live Displacement Metric
telemetry_displacement = """<TelemetryRow label="Steps passed" value={stepCount} color={COLORS.textPrimary} />
                <TelemetryRow
                  label="Live Displacement"
                  value={prevDistance.toFixed(2) + " units"}
                  color={COLORS.goal}
                />"""
content = content.replace("<TelemetryRow label=\"Steps passed\" value={stepCount} color={COLORS.textPrimary} />", telemetry_displacement)

with open("Frontend/src/App.jsx", "w") as f:
    f.write(content)
