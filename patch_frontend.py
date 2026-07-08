import re

with open("Frontend/src/App.jsx", "r") as f:
    content = f.read()

# Add stepCountRef
step_count_ref_code = """
  const stepCountRef = useRef(stepCount);
  useEffect(() => {
    stepCountRef.current = stepCount;
  }, [stepCount]);
"""
content = content.replace("const globalPathRef = useRef(globalPath);", "const globalPathRef = useRef(globalPath);\n" + step_count_ref_code)

# Add state variables for the new metrics
new_metrics_state = """
  const [discountedGoalValue, setDiscountedGoalValue] = useState(500.0);
  const [angularConvergence, setAngularConvergence] = useState(0.0);
  const [efficiencyScore, setEfficiencyScore] = useState(0.0);
"""
content = content.replace("const [cumulativeReward, setCumulativeReward] = useState(0);", "const [cumulativeReward, setCumulativeReward] = useState(0);\n" + new_metrics_state)

# Update runStep API call body
api_call_body_old = """
          prev_distance: currentPrevDistance,
          action,
          obstacles: obstaclesRef.current,
"""
api_call_body_new = """
          prev_distance: currentPrevDistance,
          initial_distance: Math.hypot(GOAL.x - START.x, GOAL.y - START.y),
          step_count: stepCountRef.current,
          action,
          obstacles: obstaclesRef.current,
"""
content = content.replace(api_call_body_old, api_call_body_new)

# Update state setters in runStep
state_setters_old = """
      setRobot({ x: data.x, y: data.y, theta: data.theta });
      setPrevDistance(data.distance);
      setStatus(data.status);
      setLastReward(data.reward);
      setCumulativeReward((prev) => prev + data.reward);
      setStepCount((prev) => prev + 1);
      setRewardHistory((prev) => [...prev, data.reward].slice(-100));
      setErrorMsg(null);
"""
state_setters_new = """
      setRobot({ x: data.x, y: data.y, theta: data.theta });
      setPrevDistance(data.distance);
      setStatus(data.status);
      setLastReward(data.reward);
      setCumulativeReward((prev) => prev + data.reward);
      setStepCount((prev) => prev + 1);
      setRewardHistory((prev) => [...prev, data.reward].slice(-100));
      setErrorMsg(null);
      
      // Update new metrics from backend
      setAngularConvergence(data.r_theta);
      setEfficiencyScore(data.efficiency_score);
      setDiscountedGoalValue(500.0 * Math.pow(0.99, stepCountRef.current + 1));
"""
content = content.replace(state_setters_old, state_setters_new)

# Reset new metrics in handleReset
reset_old = """
    setCumulativeReward(0);
    setRewardHistory([]);
"""
reset_new = """
    setCumulativeReward(0);
    setRewardHistory([]);
    setDiscountedGoalValue(500.0);
    setAngularConvergence(0.0);
    setEfficiencyScore(0.0);
"""
content = content.replace(reset_old, reset_new)

# Update Live Reward Details setting
live_reward_details_old = """
      // Compute live reward formula variables on the fly
      const desiredTheta = bearingToGoal(data.x, data.y);
      const thetaErr = normalizeAngle(desiredTheta - data.theta);
      const rThetaVal = 5.0 - Math.cos(thetaErr);
      const rDVal = 2.0 * Math.exp(-data.distance / currentPrevDistance);
      const baseRew = rDVal * rThetaVal;
"""
live_reward_details_new = """
      // Use exact values from backend
      const desiredTheta = bearingToGoal(data.x, data.y);
      const thetaErr = normalizeAngle(desiredTheta - data.theta);
      const rThetaVal = data.r_theta;
      const rDVal = data.r_d;
      const baseRew = rDVal * rThetaVal;
"""
content = content.replace(live_reward_details_old, live_reward_details_new)

# Add telemetry to UI
telemetry_old = """
                <TelemetryRow
                  label="Live Displacement"
                  value={prevDistance.toFixed(2) + " units"}
                  color={COLORS.goal}
                />
"""
telemetry_new = """
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
"""
content = content.replace(telemetry_old, telemetry_new)

with open("Frontend/src/App.jsx", "w") as f:
    f.write(content)
