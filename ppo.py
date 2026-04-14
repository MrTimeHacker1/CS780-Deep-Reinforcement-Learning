import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Assumes obelix.py is in the same directory
from obelix import OBELIX


@dataclass
class PPOConfig:
    # Environment settings
    arena_size: int = 500
    max_steps_per_episode: int = 1000
    wall_obstacles: bool = True
    difficulty: int = 3
    box_speed: int = 2

    # Training hyperparameters
    total_timesteps: int = 300_000
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 8
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    hidden_size: int = 128

    # Logging & Evaluation
    eval_interval: int = 20_000
    eval_episodes: int = 20
    seed: int = 3275628
    device: str = "cpu"


# ==========================================
# 1. Environment Wrapper
# ==========================================
class ObelixGymWrapper(gym.Env):
    """
    Wraps the raw OBELIX environment to work nicely with standard RL loops.
    Adds some reward shaping to help the agent learn faster.
    """
    def __init__(self, config: PPOConfig, seed=None):
        super().__init__()

        # Scale things relative to the arena size
        scaling_factor = config.arena_size / 500.0

        # Shaping constants
        self.distance_reward_multiplier = 0.5
        self.bad_turn_penalty = -1.5

        self.env = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=config.arena_size,
            max_steps=config.max_steps_per_episode,
            wall_obstacles=config.wall_obstacles,
            difficulty=config.difficulty,
            box_speed=config.box_speed,
            seed=seed
        )

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32)

        # Track recent actions to prevent the bot from just spinning in circles
        self.recent_actions = deque(maxlen=6)
        self.previous_distance_to_box = 0.0

    def get_distance_to_box(self):
        """Simple helper to calculate straight-line distance to the box."""
        dx = self.env.bot_center_x - self.env.box_center_x
        dy = self.env.bot_center_y - self.env.box_center_y
        return np.hypot(dx, dy)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed)

        self.recent_actions.clear()
        self.previous_distance_to_box = self.get_distance_to_box()

        return obs.astype(np.float32), {}

    def step(self, action_index):
        # Translate the integer action from the model back to the string the env expects
        action_mapping = ["L45", "L22", "FW", "R22", "R45"]
        action_string = action_mapping[action_index]
        self.recent_actions.append(action_index)

        # Take a step in the base environment
        obs, base_reward, is_done = self.env.step(action_string, render=False)

        terminated = is_done
        truncated = False
        shaped_reward = base_reward

        # Reward Shaping: Penalize turning in place if it didn't help
        if action_string != "FW" and base_reward == -1.0:
            shaped_reward = self.bad_turn_penalty

        # Reward Shaping: Encourage getting closer to the box (if we haven't touched it yet)
        current_distance = self.get_distance_to_box()
        if not self.env.enable_push:
            distance_closed = self.previous_distance_to_box - current_distance
            shaped_reward += (distance_closed * self.distance_reward_multiplier)

        self.previous_distance_to_box = current_distance

        # Anti-spin mechanism: Truncate the episode if it's just spinning without moving forward
        is_spinning = len(self.recent_actions) == 6 and all(act != 2 for act in self.recent_actions)
        if is_spinning:
            shaped_reward -= 50.0
            truncated = True

        info = {"raw_reward": base_reward}
        return obs.astype(np.float32), float(shaped_reward), terminated, truncated, info


# ==========================================
# 2. PPO Networks & Memory
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        policy_logits = self.actor(obs)
        state_value = self.critic(obs).squeeze(-1)
        return policy_logits, state_value


class RolloutMemory:
    """Holds data collected during environment interaction."""
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


# ==========================================
# 3. The PPO Agent
# ==========================================
class PPOAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCritic(obs_dim, action_dim, config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        """Picks an action. Randomly samples during training, picks the best during eval."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_tensor)

        action_distribution = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = action_distribution.sample()

        return int(action.item()), float(action_distribution.log_prob(action).item()), float(value.item())

    def update_policy(self, memory, next_state_value):
        """Runs the actual PPO math to update the neural network weights."""
        obs = torch.stack(memory.observations)
        actions = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs)
        values = torch.stack(memory.values)

        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(memory.dones, dtype=torch.float32, device=self.device)

        # 1. Calculate Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for t in reversed(range(len(rewards))):
            is_not_done = 1.0 - dones[t]
            next_value = next_state_value if t == len(rewards) - 1 else values[t + 1]

            delta = rewards[t] + (self.config.gamma * next_value * is_not_done) - values[t]
            last_advantage = delta + (self.config.gamma * self.config.gae_lambda * is_not_done * last_advantage)
            advantages[t] = last_advantage

        returns = advantages + values

        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # 2. Train the network for a few epochs over the collected batch
        dataset_size = len(obs)

        for _ in range(self.config.update_epochs):
            # Shuffle indices for minibatches
            indices = torch.randperm(dataset_size, device=self.device)

            for start_idx in range(0, dataset_size, self.config.minibatch_size):
                batch_indices = indices[start_idx : start_idx + self.config.minibatch_size]

                # Get fresh predictions from the network
                logits, current_values = self.model(obs[batch_indices])
                distribution = torch.distributions.Categorical(logits=logits)

                new_log_probs = distribution.log_prob(actions[batch_indices])
                entropy = distribution.entropy()

                # Calculate PPO clipped loss
                probability_ratio = (new_log_probs - old_log_probs[batch_indices]).exp()
                batch_advantages = advantages[batch_indices]

                unclipped_loss = probability_ratio * batch_advantages
                clipped_loss = torch.clamp(probability_ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * batch_advantages

                actor_loss = -torch.min(unclipped_loss, clipped_loss).mean()

                # Critic loss (MSE)
                critic_loss = 0.5 * (returns[batch_indices] - current_values).pow(2).mean()

                # Total loss combines actor, critic, and an entropy bonus (encourages exploration)
                total_loss = actor_loss + (self.config.value_coef * critic_loss) - (self.config.entropy_coef * entropy.mean())

                # Backpropagate and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()


# ==========================================
# 4. Utilities
# ==========================================
def check_if_successful(env):
    """Check if the robot successfully pushed the box to the boundary."""
    # Handle the fact that the env might be wrapped by Gym
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    if not base_env.enable_push:
        return False
    return base_env._box_touches_boundary(base_env.box_center_x, base_env.box_center_y)


def run_evaluation(agent, config, num_episodes, seed_offset):
    """Runs a few episodes without exploration to see how good the policy currently is."""
    episode_returns = []
    successes = 0

    for episode in range(num_episodes):
        env = ObelixGymWrapper(config, seed=config.seed + seed_offset + episode)
        obs, _ = env.reset()

        total_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_returns.append(total_reward)
        if check_if_successful(env.env):
            successes += 1

    return {
        "mean_return": np.mean(episode_returns),
        "success_rate": successes / num_episodes
    }


def save_plot_png(filepath: Path, x_data, y_data, title):
    """Uses matplotlib to save a clean PNG line chart of the training progress."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not x_data or not y_data:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='#1f77b4', markersize=4)
    plt.title(title, pad=15)
    plt.xlabel("Global Step")
    plt.ylabel("Evaluation Mean Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ==========================================
# 5. Main Training Loop
# ==========================================
def start_training():
    config = PPOConfig()

    # Set seeds for reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup output directory
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment and agent
    env = ObelixGymWrapper(config, config.seed)
    initial_obs, _ = env.reset()

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(obs_size, action_size, config)
    memory = RolloutMemory()

    # Trackers
    global_step = 0
    current_episode_return = 0.0
    best_eval_score = float("-inf")

    history_episode_returns = []
    history_eval_steps = []
    history_eval_scores = []

    next_eval_step = config.eval_interval
    current_obs = initial_obs

    # Main interaction loop
    while global_step < config.total_timesteps:
        memory.clear()

        # 1. Collect a batch of experience from the environment
        for _ in range(config.rollout_steps):
            action, log_prob, state_value = agent.select_action(current_obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            is_done = terminated or truncated

            # Save experience to memory
            memory.observations.append(torch.tensor(current_obs, dtype=torch.float32, device=agent.device))
            memory.actions.append(torch.tensor(action, device=agent.device))
            memory.log_probs.append(torch.tensor(log_prob, dtype=torch.float32, device=agent.device))
            memory.values.append(torch.tensor(state_value, dtype=torch.float32, device=agent.device))
            memory.rewards.append(reward)
            memory.dones.append(float(is_done))

            current_episode_return += reward
            global_step += 1
            current_obs = next_obs

            # Handle episode end
            if is_done:
                history_episode_returns.append(current_episode_return)
                current_obs, _ = env.reset(seed=config.seed + global_step)
                current_episode_return = 0.0

        # 2. Update the policy using the collected data
        with torch.no_grad():
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            _, next_state_value = agent.model(obs_tensor)

        agent.update_policy(memory, next_state_value.item())

        # 3. Evaluate and save checkpoints periodically
        if global_step >= next_eval_step or global_step >= config.total_timesteps:
            eval_results = run_evaluation(agent, config, config.eval_episodes, seed_offset=500_000 + global_step)

            history_eval_steps.append(global_step)
            history_eval_scores.append(eval_results["mean_return"])

            print(f"[Evaluation] Step: {global_step} | Mean Return: {eval_results['mean_return']:.2f} | Success Rate: {eval_results['success_rate']:.2%}")

            # Save best model
            if eval_results["mean_return"] > best_eval_score:
                best_eval_score = eval_results["mean_return"]
                torch.save(agent.model.state_dict(), output_dir / "best_model.pt")

            next_eval_step += config.eval_interval

        # Print basic progress updates during training
        if global_step % 10_000 < config.rollout_steps and history_episode_returns:
            recent_avg = np.mean(history_episode_returns[-20:])
            print(f"[Training] Step: {global_step} | Avg Return (Last 20 episodes): {recent_avg:.2f}")

    # Wrap up and save final artifacts
    torch.save(agent.model.state_dict(), output_dir / "latest_model.pt")

    # Save the matplotlib plot as a PNG
    save_plot_png(output_dir / "evaluation_curve.png", history_eval_steps, history_eval_scores, "Evaluation Returns Over Time")


if __name__ == "__main__":
    start_training()