import re

with open("Frontend/src/App.jsx", "r") as f:
    content = f.read()

# 1. Fix drawScene useEffect dependencies
old_deps = "}, [obstacles, pathHistory, drawingStart, drawingCurrent, isDrawingMode]);"
new_deps = "}, [obstacles, pathHistory, drawingStart, drawingCurrent, isDrawingMode, globalPath, policyMode]);"
content = content.replace(old_deps, new_deps)

# 2. Add stateHistory and historyIndex to state
state_code = """
  const [stateHistory, setStateHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
"""
content = content.replace("const [drawingCurrent, setDrawingCurrent] = useState(null);", "const [drawingCurrent, setDrawingCurrent] = useState(null);\n" + state_code)

# 3. Add applySnapshot and handlers
handlers_code = """
  const applySnapshot = (snap) => {
    setRobot(snap.robot);
    setPrevDistance(snap.prevDistance);
    setStatus(snap.status);
    setLastReward(snap.lastReward);
    setCumulativeReward(snap.cumulativeReward);
    setStepCount(snap.stepCount);
    setLiveRewardDetails(snap.liveRewardDetails);
    setPathHistory(snap.pathHistory);
    setRewardHistory(snap.rewardHistory);
  };

  const handleStepBack = () => {
    setIsRunning(false);
    let targetIndex = historyIndex === -1 ? stateHistory.length - 2 : historyIndex - 1;
    if (targetIndex < 0) return;
    applySnapshot(stateHistory[targetIndex]);
    setHistoryIndex(targetIndex);
  };

  const handleStepForward = () => {
    if (historyIndex === -1 || historyIndex >= stateHistory.length - 1) return;
    const targetIndex = historyIndex + 1;
    applySnapshot(stateHistory[targetIndex]);
    setHistoryIndex(targetIndex === stateHistory.length - 1 ? -1 : targetIndex);
  };
"""
content = content.replace("  const handleRandomizeObstacles = () => {", handlers_code + "\n  const handleRandomizeObstacles = () => {")

# 4. Modify handleReset to reset history
content = content.replace("setPathHistory([]);", "setPathHistory([]);\n    setStateHistory([]);\n    setHistoryIndex(-1);")

# 5. Modify runStep to save state snapshots
# We need to find the block where state is updated in runStep and replace it
run_step_find = """
      setPathHistory(prev => [...prev, {x: current.x, y: current.y}]);
      setRobot({ x: data.x, y: data.y, theta: data.theta });
      setPrevDistance(data.distance);
      setStatus(data.status);
      setLastReward(data.reward);
      setCumulativeReward((prev) => prev + data.reward);
      setStepCount((prev) => prev + 1);
      setRewardHistory((prev) => [...prev, data.reward].slice(-100));
      setErrorMsg(null);

      // Compute live reward formula variables on the fly
      const desiredTheta = bearingToGoal(data.x, data.y);
      const thetaErr = normalizeAngle(desiredTheta - data.theta);
      const rThetaVal = 5.0 - Math.cos(thetaErr);
      const rDVal = 2.0 * Math.exp(-data.distance / currentPrevDistance);
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
"""

run_step_replace = """
      const newLiveRewardDetails = {
        dPrev: currentPrevDistance,
        dCurr: data.distance,
        thetaError: normalizeAngle(bearingToGoal(data.x, data.y) - data.theta),
        rTheta: 5.0 - Math.cos(normalizeAngle(bearingToGoal(data.x, data.y) - data.theta)),
        rD: 2.0 * Math.exp(-data.distance / currentPrevDistance),
        baseReward: (2.0 * Math.exp(-data.distance / currentPrevDistance)) * (5.0 - Math.cos(normalizeAngle(bearingToGoal(data.x, data.y) - data.theta))),
        bonus: data.status === "goal_reached" ? 500.0 : (data.status === "collision" ? -100.0 : 0),
        totalReward: data.reward,
      };

      const newStateSnapshot = {
        robot: { x: data.x, y: data.y, theta: data.theta },
        prevDistance: data.distance,
        status: data.status,
        lastReward: data.reward,
        cumulativeReward: cumulativeReward + data.reward,
        stepCount: stepCount + 1,
        liveRewardDetails: newLiveRewardDetails,
        pathHistory: [...pathHistory, {x: current.x, y: current.y}],
        rewardHistory: [...rewardHistory, data.reward].slice(-100)
      };

      setStateHistory(prev => {
        const hist = historyIndex !== -1 ? prev.slice(0, historyIndex + 1) : prev;
        return [...hist, newStateSnapshot];
      });

      if (historyIndex !== -1) setHistoryIndex(-1);

      applySnapshot(newStateSnapshot);
      setErrorMsg(null);
"""
# Need to use regex or string replace. Let's use re.sub for safety, since whitespace might differ slightly.
content = re.sub(
    r'setPathHistory\(prev => \[\.\.\.prev, \{x: current\.x, y: current\.y\}\]\);[\s\S]*?totalReward: data\.reward,\n\s+\}\);',
    run_step_replace.strip().replace('\\', '\\\\'),
    content
)

# 6. Add the buttons to the UI
buttons_ui = """
              <button
                onClick={handleStepBack}
                disabled={stateHistory.length < 2 || (historyIndex !== -1 && historyIndex <= 0)}
                className="px-4 py-2 rounded font-medium text-sm border transition-colors cursor-pointer disabled:opacity-40"
                style={{ borderColor: COLORS.panelBorder, color: COLORS.textPrimary }}
              >
                ⏪ Step Back
              </button>
              <button
                onClick={handleStepForward}
                disabled={historyIndex === -1 || historyIndex === stateHistory.length - 1}
                className="px-4 py-2 rounded font-medium text-sm border transition-colors cursor-pointer disabled:opacity-40"
                style={{ borderColor: COLORS.panelBorder, color: COLORS.textPrimary }}
              >
                Step Forward ⏩
              </button>
"""
content = content.replace("Reset Environment\n              </button>", "Reset Environment\n              </button>\n" + buttons_ui)

with open("Frontend/src/App.jsx", "w") as f:
    f.write(content)
