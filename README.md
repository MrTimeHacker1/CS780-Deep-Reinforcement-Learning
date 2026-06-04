# Autonomous Delivery Robot (OBELIX): Deep Reinforcement Learning Capstone

## Project Overview
This repository contains the environment and framework for the **CS780: Deep Reinforcement Learning** Capstone Project (IIT Kanpur). The project centers on developing a robust, autonomous controller for a warehouse delivery robot named **OBELIX** using Reinforcement Learning.

The objective is for OBELIX to navigate an unmapped, enclosed arena to locate a target payload (a grey box), safely attach to it, and efficiently push it outside the arena boundaries. 

## Mathematical Formulation (POMDP)
Unlike standard grid-world problems that provide absolute coordinate states, this environment is formulated as a strict **Partially Observable Markov Decision Process (POMDP)**. 

Formally, the environment is defined by the 7-tuple:
$$\mathcal{M} = \langle S, A, T, R, \Omega, O, \gamma \rangle$$

Where:
* $S$ is the hidden state space (the absolute $x, y, \theta$ coordinates of the robot and the box).
* $A$ is the discrete action space of the robot.
* $T(s' | s, a)$ is the transition dynamics of the physics engine.
* $R(s, a)$ is the reward function.
* $\Omega$ is the observation space (the local sensor readings).
* $O(o | s', a)$ is the emission probability of observing $o \in \Omega$ given the true state $s'$.
* $\gamma \in [0, 1)$ is the discount factor.

Because the true state $s_t$ is hidden, the agent must optimize its policy $\pi$ over a history of observations and actions $h_t = (o_0, a_0, o_1, a_1, \dots, o_t)$ to maximize the expected cumulative discounted reward:
$$J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

This introduces severe challenges for standard RL agents:
* **Perceptual Aliasing:** Many different absolute states $s_i \neq s_j$ produce the exact same observation $o_t$, requiring the agent to infer its position from historical context.
* **Temporal Ambiguity:** The agent must remember if it has seen the box recently to decide whether to continue exploring or return to a previously scanned area.

## Environment Mechanics

### Observation Space ($\Omega$)
The robot's perception is severely constrained to a discrete 18-bit sensory array, where $o_t \in \{0, 1\}^{18}$:
* **16 Sonar Bits:** These act as a 360-degree proximity radar, indicating whether obstacles (walls or the box) are "near" or "far" in 16 specific radial directions.
* **1 IR Bit:** A forward-pointing infrared sensor dedicated exclusively to detecting the target box when it is directly in front of the robot.
* **1 Collision Bit:** A boolean flag that triggers if the robot physically wedges itself against a wall.

### Action Space ($A$)
The kinematic controls are restricted to 5 discrete movements, forcing the agent to plan efficient movement sequences rather than relying on continuous steering:
1. `Move Forward (FW)`
2. `Rotate Left 22.5°`
3. `Rotate Left 45°`
4. `Rotate Right 22.5°`
5. `Rotate Right 45°`

### Reward Dynamics ($R$)
The environment provides a structured reward signal to guide the learning process:
* **Dense Sensor Rewards:** $+1$ to $+5$ for successful sensor pings and maintaining the box in the forward IR sensor.
* **Efficiency Penalty:** $-1$ per time step to encourage finding the shortest path.
* **Collision Penalty:** $-200$ for colliding with walls, heavily penalizing erratic movement.
* **Terminal Reward:** $+2000$ for successfully pushing the box outside the arena boundary.

## Curriculum Levels
To ensure policies generalize and can handle dynamic noise, the environment is split into three escalating difficulty tiers. Obstacle walls are procedurally generated in different configurations across episodes.

* **Level 1 (Static Box):** The payload remains stationary in the arena. The agent must focus on basic navigation, obstacle avoidance, and pathfinding.
* **Level 2 (Blinking Box):** The payload remains stationary but randomly disappears and reappears from the robot's sensors. This simulates severe sensor dropout. Crucially, the robot cannot physically interact with or push the box while it is "invisible," requiring the agent to wait or maneuver carefully during signal loss.
* **Level 3 (Moving & Blinking Box):** The ultimate challenge. The payload moves continuously with a constant velocity while experiencing intermittent sensor dropout. The agent must perform dynamic target interception, predicting where the box will be when the signal returns.

---

## Operating Instructions

### Prerequisites
The environment and training scripts are designed to run exclusively on CPU resources, in accordance with the original competition constraints.
* Python 3.8 or higher
* PyTorch (CPU version)

### Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/CS780-Obelix-RL.git](https://github.com/yourusername/CS780-Obelix-RL.git)
cd CS780-Obelix-RL
pip install -r requirements.txt
