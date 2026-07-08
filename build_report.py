import os

artifact_path = "/home/csi/.gemini/antigravity-ide/brain/1ba40604-6be8-4647-997d-264fbbf5b187/RL_Demonstration_Architecture.md"

with open("Backend/main.py", "r") as f:
    backend_code = f.read()

with open("Frontend/src/App.jsx", "r") as f:
    frontend_code = f.read()

content = f"""# Deep Reinforcement Learning Architecture

This document contains the complete and fully integrated code blocks for both the backend and frontend of our mobile robot navigation prototype.

> [!NOTE]
> **Emergent Shortest-Path Optimization Explained**
> 
> The shortest-path behavior you see is not explicitly hard-coded into the greedy action selector. It emerges mathematically from the reinforcement learning components:
> 1. **The Multiplicative Reward ($R_d \\times R_\\theta$)**: Strongly aligns the robot towards the goal bearing and strictly limits perpendicular/tangential movement, as any deviation aggressively shrinks the step reward.
> 2. **Discount Factor ($\\gamma = 0.99$)**: This is the primary driver of the "least distant path." Because the final terminal reward ($+500$) decays exponentially by a factor of $0.99^{{\\text{{steps}}}}$, every additional step taken to reach the goal burns roughly 1% of its final value. To maximize cumulative return, the optimal policy mathematically converges to the trajectory requiring the minimum possible number of discrete steps—the exact straight line to the goal.

## 1. Backend API (`Backend/main.py`)
```python
{backend_code}
```

## 2. Frontend Application (`Frontend/src/App.jsx`)
```jsx
{frontend_code}
```
"""

with open(artifact_path, "w") as f:
    f.write(content)

print("Report generated successfully.")
