import math
import random
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from obelix import OBELIX


@dataclass
class PPOConfig:
    arena_size: int = 500
    max_steps_per_episode: int = 1000
    wall_obstacles: bool = True
    difficulty: int = 3
    box_speed: int = 2

    numWorkers: int = 4
    maxEpisodes: int = 2000
    maxEpisodeSteps: int = 1024
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    hidDim: list = None
    
    policyOptimizerLR: float = 3e-4
    policyOptimizationEpochs: int = 8
    policyClipRange: float = 0.2
    policySampleRatio: float = 0.25
    policyStoppingKL: float = 0.015
    MAX_POLICY_GRAD: float = 0.5
    
    valueOptimizerLR: float = 1e-3
    valueOptimizationEpochs: int = 8
    valueClipRange: float = 0.2
    valueSampleRatio: float = 0.25
    valueStoppingMSE: float = 10.0
    MAX_VALUE_GRAD: float = 0.5
    
    entropyCoef: float = 0.01
    
    MAX_EVAL_EPISODES: int = 20
    eval_interval: int = 50
    seed: int = 3275628
    device: str = "cpu"

    def __post_init__(self):
        if self.hidDim is None:
            self.hidDim = [128, 128]


class ObelixGymWrapper(gym.Env):
    def __init__(self, config: PPOConfig, seed=None):
        super().__init__()
        scaling_factor = config.arena_size / 500.0
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
        from collections import deque
        self.recent_actions = deque(maxlen=6)
        self.previous_distance_to_box = 0.0

    def get_distance_to_box(self):
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
        action_mapping = ["L45", "L22", "FW", "R22", "R45"]
        action_string = action_mapping[action_index]
        self.recent_actions.append(action_index)

        obs, base_reward, is_done = self.env.step(action_string, render=False)
        terminated = is_done
        truncated = False
        shaped_reward = base_reward

        if action_string != "FW" and base_reward == -1.0:
            shaped_reward = self.bad_turn_penalty

        current_distance = self.get_distance_to_box()
        if not self.env.enable_push:
            distance_closed = self.previous_distance_to_box - current_distance
            shaped_reward += (distance_closed * self.distance_reward_multiplier)
        self.previous_distance_to_box = current_distance

        is_spinning = len(self.recent_actions) == 6 and all(act != 2 for act in self.recent_actions)
        if is_spinning:
            shaped_reward -= 50.0
            truncated = True

        return obs.astype(np.float32), float(shaped_reward), terminated, truncated, {"raw_reward": base_reward}


class MultiEnv:
    def __init__(self, config, N_WORKERS, seed):
        self.N_WORKERS = N_WORKERS
        self.pipes = [mp.Pipe() for _ in range(N_WORKERS)]
        self.workers = []
        for id in range(N_WORKERS):
            p = mp.Process(target=self.work, args=(id, config, seed, self.pipes[id][1]))
            self.workers.append(p)
        for w in self.workers: w.start()

    def reset(self, id=None, **kwargs):
        if id is not None:
            parent_end, _ = self.pipes[id]
            self.send_msg(('reset', {}), id)
            s = parent_end.recv()
            return s
        
        self.broadcast(('reset', kwargs))
        s_list = [parent_end.recv() for parent_end, _ in self.pipes]
        return np.array(s_list)

    def step(self, actions):
        for id in range(self.N_WORKERS):
            self.send_msg(('step', {'action': actions[id]}), id)
        
        results = []
        for id in range(self.N_WORKERS):
            parent_end, _ = self.pipes[id]
            s, r, terminated, truncated, info = parent_end.recv()
            done = terminated or truncated
            results.append((s, r, done))
            
        return results

    def work(self, id, config, seed, worker_end):
        env = ObelixGymWrapper(config, seed=seed + id)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                obs, _ = env.reset(**kwargs)
                worker_end.send(obs)
            elif cmd == 'step':
                obs, r, term, trunc, info = env.step(kwargs['action'])
                worker_end.send((obs, r, term, trunc, info))
            else:
                env.close()
                worker_end.close()
                break

    def broadcast(self, m):
        for parent_end, _ in self.pipes:
            parent_end.send(m)

    def send_msg(self, m, id):
        parent_end, _ = self.pipes[id]
        parent_end.send(m)

    def close(self):
        self.broadcast(('close', {}))
        for w in self.workers: w.join()


class ValueNetwork(nn.Module):
    def __init__(self, stateDim, hDims, activationFn=F.relu):
        super().__init__()
        self.activation = activationFn
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hDims)-1):
            self.hLayers.append(nn.Linear(hDims[i], hDims[i+1]))
        self.out = nn.Linear(hDims[-1], 1)

    def forward(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        l = self.activation(self.inputLayer(states))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        q = self.out(l)
        return q.squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hDims, activationFn=F.relu):
        super().__init__()
        self.activation = activationFn
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hDims)-1):
            self.hLayers.append(nn.Linear(hDims[i], hDims[i+1]))
        self.out = nn.Linear(hDims[-1], actionDim)

    def forward(self, states, actions=None):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        l = self.activation(self.inputLayer(states))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        logits = self.out(l)
        
        distrib = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            actions = distrib.sample()
            
        logPs = distrib.log_prob(actions)
        entropies = distrib.entropy()
        actions_greedy = torch.argmax(logits, dim=-1)
        
        return actions, logPs, entropies, actions_greedy


class EpisodeBuffer:
    def __init__(self, gamma, lam, stateDim, numWorkers, maxEpisodeSteps):
        self.gamma = gamma
        self.lam = lam
        self.numWorkers = numWorkers
        self.maxEpisodeSteps = maxEpisodeSteps
        self.reset()

    def reset(self):
        self.bufferStates = []
        self.bufferActions = []
        self.bufferReturns = []
        self.bufferGAEs = []
        self.bufferLogp_as = []

    def fill(self, envs, pNetwork, vNetwork, device):
        ss = envs.reset()
        
        states, actions, logps, rewards, values, dones = [], [], [], [], [], []
        
        for _ in range(self.maxEpisodeSteps):
            with torch.no_grad():
                ss_tensor = torch.tensor(ss, dtype=torch.float32).to(device)
                a, logp_a, _, _ = pNetwork(ss_tensor)
                v = vNetwork(ss_tensor)
                
            results = envs.step(a.cpu().numpy())
            
            sNexts = np.array([res[0] for res in results])
            rs = np.array([res[1] for res in results])
            ds = np.array([res[2] for res in results])
            
            states.append(ss_tensor)
            actions.append(a)
            logps.append(logp_a)
            rewards.append(torch.tensor(rs, dtype=torch.float32).to(device))
            values.append(v)
            dones.append(torch.tensor(ds, dtype=torch.float32).to(device))
            
            ss = sNexts
            
            if ds.sum() > 0:
                dones_ids = np.flatnonzero(ds)
                for id in dones_ids:
                    ss[id] = envs.reset(id)

        with torch.no_grad():
            next_v = vNetwork(torch.tensor(ss, dtype=torch.float32).to(device))
            
        states = torch.stack(states)
        actions = torch.stack(actions)
        logps = torch.stack(logps)
        values = torch.stack(values)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        gaes = torch.zeros_like(rewards).to(device)
        last_gae = 0.0
        
        for t in reversed(range(self.maxEpisodeSteps)):
            if t == self.maxEpisodeSteps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_v
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nextnonterminal * last_gae
            gaes[t] = last_gae

        returns = gaes + values

        self.bufferStates = states.view(-1, states.shape[-1])
        self.bufferActions = actions.view(-1)
        self.bufferReturns = returns.view(-1)
        self.bufferGAEs = gaes.view(-1)
        self.bufferLogp_as = logps.view(-1)

    def returnElements(self):
        return (self.bufferStates, self.bufferActions, 
                self.bufferReturns, self.bufferGAEs, self.bufferLogp_as)


class PPO:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        temp_env = ObelixGymWrapper(config)
        stateDim = temp_env.observation_space.shape[0]
        actionDim = temp_env.action_space.n
        temp_env.close()

        self.envs = MultiEnv(config, config.numWorkers, config.seed)
        
        self.pNetwork = PolicyNetwork(stateDim, actionDim, config.hidDim).to(self.device)
        self.policyOptimizerFn = optim.Adam(self.pNetwork.parameters(), lr=config.policyOptimizerLR)
        
        self.vNetwork = ValueNetwork(stateDim, config.hidDim).to(self.device)
        self.valueOptimizerFn = optim.Adam(self.vNetwork.parameters(), lr=config.valueOptimizerLR)
        
        self.rBuffer = EpisodeBuffer(config.gamma, config.gae_lambda, stateDim, 
                                     config.numWorkers, config.maxEpisodeSteps)
        
        self.current_episode = 0
        self.history_eval_steps = []
        self.history_eval_scores = []

    def runPPO(self):
        self.trainAgent()
        self.evaluateAgent()
        self.envs.close()
        
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_plot_png(output_dir / "evaluation_curve.png", self.history_eval_steps, self.history_eval_scores, "Evaluation Returns Over Time")

    def trainAgent(self):
        while self.current_episode < self.cfg.maxEpisodes:
            self.rBuffer.fill(self.envs, self.pNetwork, self.vNetwork, self.device)
            self.trainNetworks()
            self.rBuffer.reset()
            
            self.current_episode += 1
            if self.current_episode % self.cfg.eval_interval == 0:
                eval_mean, eval_std = self.evaluateAgent()
                print(f"[Evaluation] Episode: {self.current_episode} | Mean Return: {eval_mean:.2f}")
                self.history_eval_steps.append(self.current_episode)
                self.history_eval_scores.append(eval_mean)

    def trainNetworks(self):
        ss, as_buf, returns, gaes, logps_old = self.rBuffer.returnElements()
        
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
        nSamples = len(as_buf)
        batchSize = int(self.cfg.policySampleRatio * nSamples)
        
        for e in range(self.cfg.policyOptimizationEpochs):
            indices = torch.randperm(nSamples)
            
            for start in range(0, nSamples, batchSize):
                batchIDs = indices[start:start + batchSize]
                
                ss_Batch = ss[batchIDs]
                as_Batch = as_buf[batchIDs]
                gaes_Batch = gaes[batchIDs]
                logps_Batch = logps_old[batchIDs]
                
                _, logPs, entropies, _ = self.pNetwork(ss_Batch, as_Batch)
                
                ratios = torch.exp(logPs - logps_Batch)
                pi = ratios * gaes_Batch
                ratios_clipped = torch.clamp(ratios, 1.0 - self.cfg.policyClipRange, 1.0 + self.cfg.policyClipRange)
                pi_clipped = ratios_clipped * gaes_Batch
                
                pLoss = -1.0 * torch.min(pi, pi_clipped).mean()
                entropyLoss = -self.cfg.entropyCoef * entropies.mean()
                
                loss = pLoss + entropyLoss
                
                self.policyOptimizerFn.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pNetwork.parameters(), self.cfg.MAX_POLICY_GRAD)
                self.policyOptimizerFn.step()

            with torch.no_grad():
                _, logPs_all, _, _ = self.pNetwork(ss, as_buf)
                kl = torch.mean(logps_old - logPs_all)
            if kl.item() > self.cfg.policyStoppingKL:
                break

        for e in range(self.cfg.valueOptimizationEpochs):
            indices = torch.randperm(nSamples)
            
            for start in range(0, nSamples, batchSize):
                batchIDs = indices[start:start + batchSize]
                
                ss_Batch = ss[batchIDs]
                rs_Batch = returns[batchIDs]
                
                with torch.no_grad():
                    vs_Batch = self.vNetwork(ss_Batch)
                
                vs_p = self.vNetwork(ss_Batch)
                
                vs_p_clipped = torch.clamp(vs_p - vs_Batch, -self.cfg.valueClipRange, self.cfg.valueClipRange)
                vs_p_clipped = vs_Batch + vs_p_clipped
                
                vLoss1 = (rs_Batch - vs_p)**2
                vLoss_clipped = (rs_Batch - vs_p_clipped)**2
                
                vLoss = 0.5 * torch.max(vLoss1, vLoss_clipped).mean()
                
                self.valueOptimizerFn.zero_grad()
                vLoss.backward()
                nn.utils.clip_grad_norm_(self.vNetwork.parameters(), self.cfg.MAX_VALUE_GRAD)
                self.valueOptimizerFn.step()

            with torch.no_grad():
                vs_all = self.vNetwork(ss)
                mse = torch.mean(0.5 * (returns - vs_all)**2)
            if mse.item() > self.cfg.valueStoppingMSE:
                break

    def evaluateAgent(self):
        eval_env = ObelixGymWrapper(self.cfg)
        rewards = []
        
        for e in range(self.cfg.MAX_EVAL_EPISODES):
            rs = 0
            s, _ = eval_env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                    _, _, _, actions_greedy = self.pNetwork(s_tensor)
                    
                s, r, term, trunc, _ = eval_env.step(actions_greedy.item())
                rs += r
                done = term or trunc
                
                if done:
                    rewards.append(rs)
                    break
                    
        eval_env.close()
        return np.mean(rewards), np.std(rewards)

    def save_plot_png(self, filepath: Path, x_data, y_data, title):
        if not x_data or not y_data:
            return
        plt.figure(figsize=(10, 5))
        plt.plot(x_data, y_data, marker='o', linestyle='-', color='#1f77b4', markersize=4)
        plt.title(title, pad=15)
        plt.xlabel("Episodes")
        plt.ylabel("Evaluation Mean Return")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()


if __name__ == "__main__":
    config = PPOConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    agent = PPO(config)
    agent.runPPO()
