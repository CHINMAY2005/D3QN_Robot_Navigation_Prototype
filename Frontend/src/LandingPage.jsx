import React, { useState, useEffect, useRef } from "react";

export default function LandingPage({ onViewModeChange }) {
  // --- Local Mini Simulation State ---
  const [robotPos, setRobotPos] = useState({ x: 50, y: 120, theta: 0 });
  const [goalPos] = useState({ x: 350, y: 120 });
  const [obstacles] = useState([
    { x: 180, y: 70, width: 40, height: 100, radius: 25 }, // Obstacle in the middle
  ]);
  const [lidarRays, setLidarRays] = useState([]);
  const [simStatus, setSimStatus] = useState("navigating"); // "navigating" | "goal_reached"

  const requestRef = useRef();
  const simStateRef = useRef({
    x: 50,
    y: 120,
    theta: 0,
    resetTimer: 0,
  });

  // Local simulation loop
  useEffect(() => {
    const loop = () => {
      const state = simStateRef.current;

      if (state.resetTimer > 0) {
        state.resetTimer -= 1;
        if (state.resetTimer === 0) {
          // Reset simulator
          state.x = 50;
          state.y = 120;
          state.theta = 0;
          setSimStatus("navigating");
        }
        requestRef.current = requestAnimationFrame(loop);
        return;
      }

      // 1. Calculate attractive force to goal
      const dxGoal = goalPos.x - state.x;
      const dyGoal = goalPos.y - state.y;
      const distGoal = Math.sqrt(dxGoal * dxGoal + dyGoal * dyGoal);

      let forceX = dxGoal / Math.max(distGoal, 1) * 2.0;
      let forceY = dyGoal / Math.max(distGoal, 1) * 2.0;

      // 2. Calculate repulsive force from obstacles
      obstacles.forEach((obs) => {
        const obsCenterX = obs.x + obs.width / 2;
        const obsCenterY = obs.y + obs.height / 2;
        const dxObs = state.x - obsCenterX;
        const dyObs = state.y - obsCenterY;
        const distObs = Math.sqrt(dxObs * dxObs + dyObs * dyObs);

        // Define influence radius
        const influenceRadius = 110;
        if (distObs < influenceRadius) {
          // Stronger repulsion when closer
          const strength = (influenceRadius - distObs) / distObs * 4.5;
          // Apply directional force
          forceX += (dxObs / distObs) * strength;
          // Add a tangential bias to steer around rather than getting stuck directly behind
          forceY += (dyObs / distObs) * strength + (dyObs >= 0 ? 1 : -1) * 1.5;
        }
      });

      // Avoid borders
      if (state.x < 30) forceX += 2;
      if (state.x > 370) forceX -= 2;
      if (state.y < 30) forceY += 2;
      if (state.y > 210) forceY -= 2;

      // 3. Convert force vector to target heading
      const targetTheta = Math.atan2(forceY, forceX);

      // Smoothly interpolate angle theta towards targetTheta
      let angleDiff = targetTheta - state.theta;
      // Normalize angle diff to [-PI, PI]
      while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
      while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;

      const maxTurnSpeed = 0.15; // rad per frame
      state.theta += Math.max(-maxTurnSpeed, Math.min(maxTurnSpeed, angleDiff));

      // 4. Move forward
      const speed = 2.2;
      state.x += Math.cos(state.theta) * speed;
      state.y += Math.sin(state.theta) * speed;

      // Check goal collision
      if (distGoal < 16) {
        setSimStatus("goal_reached");
        state.resetTimer = 60; // Wait 1 second (60 frames) at goal before reset
      }

      // 5. Generate visual LiDAR rays for landing page effect (8 rays)
      const rays = [];
      const numRays = 8;
      const fov = Math.PI * 2;
      const maxRange = 100;

      for (let i = 0; i < numRays; i++) {
        const rayAngle = state.theta + (i * fov) / numRays;
        const dx = Math.cos(rayAngle);
        const dy = Math.sin(rayAngle);

        let hitDist = maxRange;

        // Check intersection with boundary or obstacle
        obstacles.forEach((obs) => {
          // Check line segment intersections with box
          // Simplify with circle-like approximation or simple bounding box checking
          const steps = 30;
          for (let step = 0; step < steps; step++) {
            const stepDist = (step / steps) * maxRange;
            const px = state.x + dx * stepDist;
            const py = state.y + dy * stepDist;

            if (
              px >= obs.x &&
              px <= obs.x + obs.width &&
              py >= obs.y &&
              py <= obs.y + obs.height
            ) {
              if (stepDist < hitDist) {
                hitDist = stepDist;
              }
              break;
            }
          }
        });

        rays.push({
          x1: state.x,
          y1: state.y,
          x2: state.x + dx * hitDist,
          y2: state.y + dy * hitDist,
          hit: hitDist < maxRange,
        });
      }

      setRobotPos({ x: state.x, y: state.y, theta: state.theta });
      setLidarRays(rays);

      requestRef.current = requestAnimationFrame(loop);
    };

    requestRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(requestRef.current);
  }, [goalPos, obstacles]);

  return (
    <div className="min-h-screen bg-[#07090b] text-[#e2e8f0] font-sans overflow-x-hidden selection:bg-cyan-500/30 selection:text-cyan-200">
      
      {/* Background glowing gradients */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-violet-600/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-10 left-1/3 w-96 h-96 bg-amber-500/5 rounded-full blur-[150px] pointer-events-none" />

      {/* Navigation Header */}
      <header className="relative z-10 w-full max-w-7xl mx-auto px-6 py-6 flex items-center justify-between border-b border-white/[0.03]">
        <div className="flex items-center gap-3">
          <div className="h-10 px-2 rounded-lg bg-gradient-to-tr from-cyan-400 to-violet-600 flex items-center justify-center font-black text-sm tracking-wider text-[#07090b] shadow-[0_0_15px_rgba(6,182,212,0.4)]">
            D3QN
          </div>
          <div>
            <span className="font-bold tracking-widest uppercase text-sm block">D3QN</span>
            <span className="text-[10px] text-gray-500 tracking-wider uppercase block">Autonomous Prototype</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => onViewModeChange("prototype")}
            className="text-xs uppercase tracking-wider text-gray-400 hover:text-cyan-400 font-semibold cursor-pointer transition-colors duration-200"
          >
            Original Sandbox
          </button>
          <button
            onClick={() => onViewModeChange("upgraded")}
            className="px-4 py-2 rounded border border-cyan-500/30 text-cyan-400 bg-cyan-950/10 hover:bg-cyan-500/20 text-xs uppercase tracking-widest font-bold cursor-pointer transition-all duration-300 shadow-[0_0_12px_rgba(6,182,212,0.15)] hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]"
          >
            Launch Arena
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 w-full max-w-7xl mx-auto px-6 pt-16 pb-20 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
        
        {/* Left Info Column */}
        <div className="lg:col-span-7 space-y-8 text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Deep Reinforcement Learning Prototype
          </div>

          <div className="space-y-4">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-black uppercase tracking-tight leading-tight">
              Smarter Pathfinding <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-300 to-violet-500 drop-shadow-sm">
                Collision-Free Navigation
              </span>
            </h1>
            <p className="text-gray-400 text-base md:text-lg max-w-2xl leading-relaxed">
              Evaluating lookahead reinforcement learning frameworks. Compare a Dueling Double Q-Network (D3QN) policy against baseline architectures under real-time sensor constraints and dynamic layouts.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4">
            <button
              onClick={() => onViewModeChange("upgraded")}
              className="px-8 py-4 rounded bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-black font-bold uppercase tracking-widest text-xs cursor-pointer shadow-[0_0_25px_rgba(6,182,212,0.3)] hover:shadow-[0_0_35px_rgba(6,182,212,0.5)] transition-all duration-300 transform hover:-translate-y-0.5"
            >
              Launch Comparison Arena
            </button>
            <button
              onClick={() => onViewModeChange("prototype")}
              className="px-8 py-4 rounded border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] hover:border-white/[0.15] text-[#e2e8f0] font-bold uppercase tracking-widest text-xs cursor-pointer transition-all duration-200"
            >
              Explore Original Prototype
            </button>
          </div>

          {/* Tech Badges */}
          <div className="pt-4 space-y-2">
            <span className="text-xs uppercase tracking-widest text-gray-500 block font-bold">Powered By</span>
            <div className="flex flex-wrap gap-3">
              {["FastAPI Backend", "PyTorch D3QN", "8-Ray LiDAR", "Vite & React", "Lookahead Rollout"].map((tech) => (
                <span
                  key={tech}
                  className="px-3 py-1 rounded bg-[#11171b] border border-white/[0.04] text-xs font-semibold text-gray-400"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Right Graphic / Interactive SVG Column */}
        <div className="lg:col-span-5 flex justify-center">
          <div className="relative w-full max-w-[420px] rounded-2xl border border-white/[0.05] bg-[#11171b]/60 backdrop-blur-md p-5 shadow-2xl shadow-cyan-950/10">
            <div className="flex items-center justify-between mb-4 border-b border-white/[0.05] pb-3">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-ping" />
                <span className="text-xs font-mono uppercase tracking-wider text-cyan-400">Live Simulator Preview</span>
              </div>
              <span className="text-[10px] font-mono text-gray-500 uppercase">Local JS Mock Loop</span>
            </div>

            {/* SVG Canvas */}
            <div className="relative w-full bg-[#080b0e] rounded-lg border border-white/[0.03] overflow-hidden aspect-[400/240]">
              <svg width="100%" height="100%" viewBox="0 0 400 240" className="select-none">
                {/* Grid pattern */}
                <defs>
                  <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255, 255, 255, 0.02)" strokeWidth="1" />
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />

                {/* Draw Obstacle */}
                {obstacles.map((obs, idx) => (
                  <g key={idx}>
                    <rect
                      x={obs.x}
                      y={obs.y}
                      width={obs.width}
                      height={obs.height}
                      rx="6"
                      fill="rgba(255, 255, 255, 0.03)"
                      stroke="rgba(255, 255, 255, 0.08)"
                      strokeWidth="1.5"
                    />
                    {/* Obstacle core danger glow */}
                    <rect
                      x={obs.x + 3}
                      y={obs.y + 3}
                      width={obs.width - 6}
                      height={obs.height - 6}
                      rx="4"
                      fill="rgba(147, 51, 234, 0.03)"
                      stroke="rgba(147, 51, 234, 0.15)"
                      strokeWidth="1"
                    />
                  </g>
                ))}

                {/* LiDAR Rays */}
                {lidarRays.map((ray, idx) => (
                  <line
                    key={idx}
                    x1={ray.x1}
                    y1={ray.y1}
                    x2={ray.x2}
                    y2={ray.y2}
                    stroke={ray.hit ? "rgba(245, 166, 35, 0.45)" : "rgba(6, 182, 212, 0.15)"}
                    strokeWidth={ray.hit ? "1.5" : "1"}
                    strokeDasharray={ray.hit ? "" : "3,3"}
                  />
                ))}

                {/* Target / Goal Flag */}
                <g transform={`translate(${goalPos.x}, ${goalPos.y})`}>
                  <circle cx="0" cy="0" r="14" fill="rgba(79, 214, 122, 0.05)" className="animate-pulse" />
                  <circle cx="0" cy="0" r="8" fill="rgba(79, 214, 122, 0.15)" />
                  <circle cx="0" cy="0" r="3" fill="#4fd67a" />
                  {/* Glowing halo */}
                  <circle cx="0" cy="0" r="18" fill="none" stroke="#4fd67a" strokeWidth="1" strokeOpacity="0.3" className="scale-75 origin-center animate-ping" />
                </g>

                {/* Robot Body */}
                <g transform={`translate(${robotPos.x}, ${robotPos.y}) rotate(${(robotPos.theta * 180) / Math.PI})`}>
                  {/* Robot direction indicator (Nose) */}
                  <line x1="0" y1="0" x2="16" y2="0" stroke="#f4f6f7" strokeWidth="2.5" />
                  
                  {/* Robot outer glow */}
                  <circle cx="0" cy="0" r="10" fill="rgba(6, 182, 212, 0.2)" />
                  {/* Robot main body */}
                  <circle cx="0" cy="0" r="7" fill="#00f2fe" stroke="#ffffff" strokeWidth="1.5" />
                  <circle cx="0" cy="0" r="2.5" fill="#07090b" />
                </g>

                {/* Reset Overlay Indicator */}
                {simStatus === "goal_reached" && (
                  <g>
                    <rect width="400" height="240" fill="rgba(7, 9, 11, 0.6)" className="transition-opacity duration-300" />
                    <text x="200" y="115" fill="#4fd67a" textAnchor="middle" className="text-sm font-bold uppercase tracking-widest font-sans">
                      Goal Reached!
                    </text>
                    <text x="200" y="135" fill="rgba(255, 255, 255, 0.4)" textAnchor="middle" className="text-[10px] font-sans">
                      Resetting to Start Position
                    </text>
                  </g>
                )}
              </svg>
            </div>

            {/* Live Stats panel */}
            <div className="grid grid-cols-3 gap-2 mt-4">
              <div className="bg-[#090d10] p-2.5 rounded border border-white/[0.03]">
                <span className="block text-[9px] uppercase tracking-wider text-gray-500">Robot X</span>
                <span className="font-mono text-xs text-cyan-400 font-semibold">{robotPos.x.toFixed(1)}</span>
              </div>
              <div className="bg-[#090d10] p-2.5 rounded border border-white/[0.03]">
                <span className="block text-[9px] uppercase tracking-wider text-gray-500">Robot Y</span>
                <span className="font-mono text-xs text-cyan-400 font-semibold">{robotPos.y.toFixed(1)}</span>
              </div>
              <div className="bg-[#090d10] p-2.5 rounded border border-white/[0.03]">
                <span className="block text-[9px] uppercase tracking-wider text-gray-500">Theta</span>
                <span className="font-mono text-xs text-amber-500 font-semibold">{(robotPos.theta * 180 / Math.PI).toFixed(0)}°</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Metrics Section */}
      <section className="relative z-10 w-full max-w-7xl mx-auto px-6 py-12 border-t border-white/[0.03]">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { value: "98.4%", label: "Success Rate", desc: "Collision-free goal completion" },
            { value: "85 steps", label: "Average Efficiency", desc: "Optimal path convergence" },
            { value: "< 1.5ms", label: "Decision Latency", desc: "Inference response window" },
            { value: "8 Rays", label: "LiDAR Resolution", desc: "Spatial environment scanning" },
          ].map((stat, idx) => (
            <div key={idx} className="bg-[#11171b]/40 border border-white/[0.03] rounded-xl p-5 text-left hover:border-cyan-500/10 transition-all duration-300">
              <span className="block text-2xl font-black text-cyan-400 tracking-tight">{stat.value}</span>
              <span className="block text-xs uppercase tracking-wider font-bold text-gray-300 mt-1">{stat.label}</span>
              <span className="block text-[11px] text-gray-500 mt-1">{stat.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 w-full max-w-7xl mx-auto px-6 py-16 border-t border-white/[0.03] space-y-12">
        <div className="text-center max-w-xl mx-auto space-y-3">
          <h2 className="text-2xl font-black uppercase tracking-widest">Key Innovations</h2>
          <p className="text-xs text-gray-400 leading-relaxed uppercase tracking-wider">
            Underlying architecture supporting high stability and reliability
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              title: "Dueling Double DQN",
              desc: "Splits state value V(s) and advantage A(s, a) to learn which states are valuable without learning the effect of each action.",
              icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" />
                  <path d="M8 12H16" />
                  <path d="M12 8V16" />
                </svg>
              ),
              color: "text-cyan-400 border-cyan-500/10 hover:border-cyan-500/30",
            },
            {
              title: "8-Ray Spatial Scanner",
              desc: "Equipped with a simulated 2D LiDAR raycaster scanning a 360-degree radius to fetch continuous barrier proximity data.",
              icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 3V21M3 12H21M18.36 5.64L5.64 18.36M18.36 18.36L5.64 5.64" />
                </svg>
              ),
              color: "text-violet-400 border-violet-500/10 hover:border-violet-500/30",
            },
            {
              title: "Rollout Lookahead MPC",
              desc: "Integrates learned network policies with a short horizon lookahead step to prevent local minima traps near walls.",
              icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 3v18h18" />
                  <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
                </svg>
              ),
              color: "text-indigo-400 border-indigo-500/10 hover:border-indigo-500/30",
            },
            {
              title: "Arena Customizer",
              desc: "Place, move, and size rectangular obstacle blockades interactively to test path adaptation under high-entropy configurations.",
              icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <path d="M9 17V7l7 5-7 5z" />
                </svg>
              ),
              color: "text-amber-400 border-amber-500/10 hover:border-amber-500/30",
            },
          ].map((feat, idx) => (
            <div
              key={idx}
              className={`bg-[#11171b]/60 backdrop-blur-md border rounded-2xl p-6 text-left transition-all duration-300 ${feat.color}`}
            >
              <div className="h-10 w-10 rounded-lg bg-white/[0.02] flex items-center justify-center mb-4 text-cyan-400">
                {feat.icon}
              </div>
              <h3 className="font-bold text-sm uppercase tracking-wider mb-2">{feat.title}</h3>
              <p className="text-gray-400 text-xs leading-relaxed">{feat.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Tech Architecture Block */}
      <section className="relative z-10 w-full max-w-7xl mx-auto px-6 py-16 border-t border-white/[0.03] space-y-12">
        <div className="text-center max-w-xl mx-auto space-y-3">
          <h2 className="text-2xl font-black uppercase tracking-widest">Model Feedback Loop</h2>
          <p className="text-xs text-gray-400 uppercase tracking-wider">
            How environment states map to action policies
          </p>
        </div>

        {/* Diagram Flow */}
        <div className="bg-[#11171b]/30 border border-white/[0.03] rounded-2xl p-8 max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center text-center">
            
            <div className="space-y-2">
              <span className="inline-block px-3 py-1 rounded bg-cyan-950/20 text-cyan-400 border border-cyan-500/20 font-mono text-[10px] uppercase">
                Stage 1: State Input
              </span>
              <h4 className="font-bold text-xs uppercase text-gray-200">LiDAR & Goal</h4>
              <p className="text-[10px] text-gray-500">8 sensor ranges + relative goal distance & heading angle</p>
            </div>

            <div className="flex justify-center text-gray-600">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="rotate-90 md:rotate-0">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>

            <div className="space-y-2">
              <span className="inline-block px-3 py-1 rounded bg-indigo-950/20 text-indigo-400 border border-indigo-500/20 font-mono text-[10px] uppercase">
                Stage 2: D3QN Policy
              </span>
              <h4 className="font-bold text-xs uppercase text-gray-200">Dueling Networks</h4>
              <p className="text-[10px] text-gray-500">Decoupled evaluation of state values and action advantages</p>
            </div>

            <div className="flex justify-center text-gray-600">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="rotate-90 md:rotate-0">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>

          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center text-center mt-8 pt-8 border-t border-white/[0.03]">
            
            <div className="space-y-2 md:col-start-2">
              <span className="inline-block px-3 py-1 rounded bg-violet-950/20 text-violet-400 border border-violet-500/20 font-mono text-[10px] uppercase">
                Stage 3: Decision
              </span>
              <h4 className="font-bold text-xs uppercase text-gray-200">Lookahead Rollout</h4>
              <p className="text-[10px] text-gray-500">Best angular velocity selected via MPC heuristic safety checks</p>
            </div>

            <div className="flex justify-center text-gray-600">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="rotate-90 md:rotate-0">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>

            <div className="space-y-2">
              <span className="inline-block px-3 py-1 rounded bg-amber-950/20 text-amber-400 border border-amber-500/20 font-mono text-[10px] uppercase">
                Stage 4: Actuation
              </span>
              <h4 className="font-bold text-xs uppercase text-gray-200">FastAPI Step</h4>
              <p className="text-[10px] text-gray-500">FastAPI backend computes kinematics, collision events and rewards</p>
            </div>

          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 w-full max-w-7xl mx-auto px-6 py-8 mt-12 border-t border-white/[0.03] flex flex-col md:flex-row items-center justify-between gap-4">
        <span className="text-[10px] text-gray-500 uppercase tracking-widest">
          © {new Date().getFullYear()} D3QN Robot Navigation Lab. All rights reserved.
        </span>
        <div className="flex gap-4">
          <button
            onClick={() => onViewModeChange("upgraded")}
            className="text-[10px] uppercase tracking-wider text-gray-400 hover:text-cyan-400 font-bold transition-colors"
          >
            Upgraded Simulation Arena
          </button>
          <button
            onClick={() => onViewModeChange("prototype")}
            className="text-[10px] uppercase tracking-wider text-gray-400 hover:text-cyan-400 font-bold transition-colors"
          >
            Original Sandbox
          </button>
        </div>
      </footer>
    </div>
  );
}
