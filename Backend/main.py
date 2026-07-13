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
import json
import os
from datetime import datetime
from threading import Lock
from typing import Literal, Union

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

# Allow any origin to call the API in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    w_d: float = Field(default=1.0, description="Weight factor for progress reward")
    w_theta: float = Field(default=1.0, description="Weight factor for alignment reward")
    goal_x: float = Field(default=520.0, description="Target goal x coordinate")
    goal_y: float = Field(default=200.0, description="Target goal y coordinate")
    prev_vx: float = Field(default=0.0, description="Previous linear velocity component in x")
    prev_vy: float = Field(default=0.0, description="Previous linear velocity component in y")
    prev_omega: float = Field(default=0.0, description="Previous angular velocity")
    momentum: float = Field(default=0.0, description="Linear momentum friction coefficient [0.0 - 0.95]")
    drift: float = Field(default=0.0, description="Rotational drift damping factor [0.0 - 0.95]")


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
    vx: float
    vy: float
    omega: float


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
    w_d: float = 1.0,
    w_theta: float = 1.0,
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
    r_theta = (5.0 - math.cos(theta_error)) * w_theta
    r_d = (2.0 * math.exp(-current_distance / prev_distance)) * w_d
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
    
    dist_to_goal = compute_distance(request.x, request.y, request.goal_x, request.goal_y)
    if dist_to_goal < LINEAR_VELOCITY * TIME_STEP:
        linear_velocity = dist_to_goal / TIME_STEP

    # Target velocity components
    target_vx = linear_velocity * math.cos(request.theta)
    target_vy = linear_velocity * math.sin(request.theta)

    # 2. Apply Momentum & Drift (Inertial physical factors)
    vx = (1 - request.momentum) * target_vx + request.momentum * request.prev_vx
    vy = (1 - request.momentum) * target_vy + request.momentum * request.prev_vy
    omega = (1 - request.drift) * angular_velocity + request.drift * request.prev_omega

    # Integrate kinematics (damped differential-drive / unicycle model).
    new_theta = normalize_angle(request.theta + omega * TIME_STEP)
    new_x = request.x + vx * TIME_STEP
    new_y = request.y + vy * TIME_STEP

    # 3. Compute new distance to goal and heading error.
    current_distance = compute_distance(new_x, new_y, request.goal_x, request.goal_y)
    theta_error = compute_theta_error(new_x, new_y, new_theta, request.goal_x, request.goal_y)

    # 4. Base reward.
    r_d, r_theta, reward = compute_reward(
        current_distance,
        request.prev_distance,
        theta_error,
        request.reward_type,
        request.w_d,
        request.w_theta,
    )

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
        vx=vx,
        vy=vy,
        omega=omega,
    )


@app.get("/")
def health_check():
    """Simple health check / root endpoint."""
    return {"status": "ok", "message": "Robot Navigation RL Environment is running."}


# --------------------------------------------------------------------------
# Agent Running Records Persistent Storage Endpoints
# --------------------------------------------------------------------------

RECORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_running_records.json")
file_lock = Lock()

class RunningRecord(BaseModel):
    id: str | None = Field(default=None, description="Unique record identifier")
    timestamp: str | None = Field(default=None, description="ISO-8601 timestamp")
    episode: int = Field(..., description="Episode number")
    model: str = Field(..., description="Model name (e.g. D3QN, DQN Baseline)")
    reward: float = Field(..., description="Cumulative episode reward")
    steps: int = Field(..., description="Total steps in the episode")
    status: str = Field(..., description="Run outcome status (e.g. SUCCESS, COLLISION, TIMEOUT)")
    loss: float = Field(..., description="Average loss at the end of the episode")
    epsilon: float = Field(..., description="Exploration rate (epsilon) at the end of the episode")

@app.get("/records")
def get_records():
    """Retrieve all stored agent running records."""
    with file_lock:
        if not os.path.exists(RECORDS_FILE):
            return []
        try:
            with open(RECORDS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read records: {str(e)}"}

@app.post("/records")
def create_record(record: Union[RunningRecord, list[RunningRecord]]):
    """Store one or more agent running records."""
    with file_lock:
        input_records = record if isinstance(record, list) else [record]

        records = []
        if os.path.exists(RECORDS_FILE):
            try:
                with open(RECORDS_FILE, "r") as f:
                    records = json.load(f)
            except Exception:
                # If file is corrupted or empty, start with empty list
                records = []

        now_str = datetime.utcnow().isoformat() + "Z"
        base_timestamp = int(datetime.utcnow().timestamp() * 1000)

        for i, rec in enumerate(input_records):
            # If standard RunningRecord model, convert to dict. Otherwise if dict already, use it.
            rec_dict = rec.dict() if hasattr(rec, "dict") else dict(rec)
            if not rec_dict.get("timestamp"):
                rec_dict["timestamp"] = now_str
            if not rec_dict.get("id"):
                rec_dict["id"] = f"run-{base_timestamp}-{i}"
            records.append(rec_dict)

        try:
            with open(RECORDS_FILE, "w") as f:
                json.dump(records, f, indent=2)
            return {"status": "success", "count": len(input_records)}
        except Exception as e:
            return {"error": f"Failed to write records: {str(e)}"}
