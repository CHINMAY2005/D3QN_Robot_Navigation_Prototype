"""
main.py

FastAPI backend implementing a discrete-action mobile robot navigation
environment, inspired by D3QN / DDQN robot navigation research.

Endpoint:
    POST /step -> given current state + discrete action, returns next state,
                  composite multiplicative reward, done flag, and status.

Run with:
    uvicorn main:app --reload
"""

import math
import random
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Robot Navigation RL Environment",
    description="Discrete-action step simulator for D3QN/DDQN robot navigation research.",
    version="1.0.0",
)

# Allow a locally running React dev server (CRA / Vite defaults) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Environment constants
# --------------------------------------------------------------------------

# Discrete action space: index -> angular velocity (rad/s)
ANGULAR_VELOCITIES = [-1.5, -0.75, 0.0, 0.75, 1.5]
LINEAR_VELOCITY = 15.0  # units/s, fixed for all actions

TIME_STEP = 1.0  # s, simulation integration step

GOAL_THRESHOLD = 20.0  # distance under which the goal is considered reached
GOAL_REWARD = 500.0
COLLISION_REWARD = -100.0

# Canvas / world boundaries (adjust to match frontend canvas size)
CANVAS_MIN_X = 0.0
CANVAS_MAX_X = 600.0
CANVAS_MIN_Y = 0.0
CANVAS_MAX_Y = 400.0

# Fixed goal position for this simulation (could be made dynamic/request-driven)
GOAL_X = 520.0
GOAL_Y = 200.0

StatusType = Literal["navigating", "goal_reached", "collision"]


# --------------------------------------------------------------------------
# Request / Response schemas
# --------------------------------------------------------------------------

class Obstacle(BaseModel):
    x: float
    y: float
    width: float
    height: float


class StepRequest(BaseModel):
    x: float = Field(..., description="Current robot x position")
    y: float = Field(..., description="Current robot y position")
    theta: float = Field(..., description="Current robot heading angle (radians)")
    prev_distance: float = Field(
        ..., gt=0.0, description="Distance to goal at the previous timestep (must be > 0)"
    )
    initial_distance: float = Field(
        ..., gt=0.0, description="Initial straight-line distance to the goal"
    )
    step_count: int = Field(
        ..., ge=0, description="Current step index for discount decay tracking"
    )
    action: int = Field(..., ge=0, le=4, description="Discrete action index [0-4]")
    obstacles: list[Obstacle] = Field(default_factory=list, description="List of rectangular obstacles")
    reward_type: Literal["multiplicative", "additive"] = Field(
        default="multiplicative", description="Type of reward formulation to use"
    )
    goal_reward: float = Field(default=500.0, description="Reward for reaching the goal")
    collision_reward: float = Field(default=-100.0, description="Penalty for collision or out of bounds")
    goal_x: float = Field(default=520.0, description="Dynamic Goal X coordinate")
    goal_y: float = Field(default=200.0, description="Dynamic Goal Y coordinate")
    noise_level: float = Field(default=0.0, description="Actuator noise magnitude")


class StepResponse(BaseModel):
    x: float
    y: float
    theta: float
    distance: float
    reward: float
    r_d: float
    r_theta: float
    efficiency_score: float
    done: bool
    status: StatusType


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_distance(x: float, y: float, goal_x: float, goal_y: float) -> float:
    """Euclidean distance from (x, y) to the goal."""
    return math.hypot(goal_x - x, goal_y - y)


def compute_theta_error(x: float, y: float, theta: float, goal_x: float, goal_y: float) -> float:
    """
    Normalized angular difference between the robot's current heading and
    the bearing angle pointing directly at the goal.
    """
    desired_theta = math.atan2(goal_y - y, goal_x - x)
    return normalize_angle(desired_theta - theta)


def is_out_of_bounds(x: float, y: float) -> bool:
    """Check whether the robot has hit/exceeded the canvas boundaries."""
    return (
        x <= CANVAS_MIN_X
        or x >= CANVAS_MAX_X
        or y <= CANVAS_MIN_Y
        or y >= CANVAS_MAX_Y
    )


def is_colliding_with_obstacles(x: float, y: float, obstacles: list[Obstacle]) -> bool:
    """Check whether the robot intersects any of the obstacles."""
    robot_radius = 10.0
    for obs in obstacles:
        closest_x = max(obs.x, min(x, obs.x + obs.width))
        closest_y = max(obs.y, min(y, obs.y + obs.height))
        dist = math.hypot(x - closest_x, y - closest_y)
        if dist < robot_radius:
            return True
    return False


def compute_reward(
    current_distance: float,
    prev_distance: float,
    theta_error: float,
    reward_type: str = "multiplicative",
) -> tuple[float, float, float]:
    """
    Composite reward:
    - Multiplicative: R = R_d * R_theta
    - Additive: R = R_d + R_theta - 2.0

    R_theta = 5.0 - cos(theta_error)   -> range [4.0, 6.0], peaks (min) when
                                          heading aligns with the goal bearing.
    R_d     = 2.0 * exp(-current_distance / prev_distance) -> higher reward
                                          the more the robot has closed the
                                          distance to the goal this step.
    """
    r_theta = 5.0 - math.cos(theta_error)
    r_d = 2.0 * math.exp(-current_distance / prev_distance)
    if reward_type == "additive":
        return r_d, r_theta, r_d + r_theta - 2.0
    else:
        return r_d, r_theta, r_d * r_theta


# --------------------------------------------------------------------------
# API endpoint
# --------------------------------------------------------------------------

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """
    Advance the simulation by one timestep given the current state and a
    discrete action index, returning the new state, reward, and status.
    """
    # 1. Map discrete action to angular velocity, use fixed linear velocity.
    angular_velocity = ANGULAR_VELOCITIES[request.action]
    linear_velocity = LINEAR_VELOCITY
    
    # Actuator Noise Simulation
    if request.noise_level > 0.0:
        angular_velocity += random.uniform(-1.0, 1.0) * request.noise_level
        linear_velocity += random.uniform(-0.15, 0.15) * request.noise_level * linear_velocity

    dist_to_goal = compute_distance(request.x, request.y, request.goal_x, request.goal_y)
    if dist_to_goal < linear_velocity * TIME_STEP:
        linear_velocity = dist_to_goal / TIME_STEP

    # 2. Integrate kinematics (simple differential-drive / unicycle model).
    new_theta = normalize_angle(request.theta + angular_velocity * TIME_STEP)
    new_x = request.x + linear_velocity * math.cos(new_theta) * TIME_STEP
    new_y = request.y + linear_velocity * math.sin(new_theta) * TIME_STEP

    # 3. Compute new distance to goal and heading error.
    current_distance = compute_distance(new_x, new_y, request.goal_x, request.goal_y)
    theta_error = compute_theta_error(new_x, new_y, new_theta, request.goal_x, request.goal_y)

    # 4. Base reward.
    r_d, r_theta, reward = compute_reward(current_distance, request.prev_distance, theta_error, request.reward_type)

    # Theoretical Efficiency Score calculation
    # Ratio of initial straight line distance to current steps taken
    efficiency_score = request.initial_distance / max(1, request.step_count)

    # 5. Determine terminal conditions (goal reached / collision) and adjust reward.
    done = False
    status: StatusType = "navigating"

    if current_distance < GOAL_THRESHOLD:
        reward += request.goal_reward
        done = True
        status = "goal_reached"
    elif is_out_of_bounds(new_x, new_y) or is_colliding_with_obstacles(new_x, new_y, request.obstacles):
        reward += request.collision_reward
        done = True
        status = "collision"
        # Clamp position so the returned state stays within the canvas.
        new_x = min(max(new_x, CANVAS_MIN_X), CANVAS_MAX_X)
        new_y = min(max(new_y, CANVAS_MIN_Y), CANVAS_MAX_Y)

    return StepResponse(
        x=new_x,
        y=new_y,
        theta=new_theta,
        distance=current_distance,
        reward=reward,
        r_d=r_d,
        r_theta=r_theta,
        efficiency_score=efficiency_score,
        done=done,
        status=status,
    )


@app.get("/")
def health_check():
    """Simple health check / root endpoint."""
    return {"status": "ok", "message": "Robot Navigation RL Environment is running."}
