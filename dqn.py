from collections import deque
import gymnasium as gym
from itertools import count
from logging import Logger
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import ReplayBuffer


EPS_S = 1.0
EPS_E = 0.01
EPS_DC = 0.995

class DQN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        l1_n: int,
        l2_n: int
    ) -> None:
        super(DQN, self).__init__()
        self.in_d = input_dim
        self.out_d = output_dim
        # feature layers
        self.layer1 = nn.Linear(self.in_d, l1_n)
        self.layer2 = nn.Linear(l1_n, l2_n)
        self.layer3 = nn.Linear(l2_n, self.out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)

class DuelingDQN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        l1_n: int,
        l2_n: int,
        l3_n: int
    ) -> None:
        super(DuelingDQN, self).__init__()
        self.in_d = input_dim
        self.out_d = output_dim
        # feature layers
        self.layer1 = nn.Linear(self.in_d, l1_n)
        self.layer2 = nn.Linear(l1_n, l2_n)
        # value layers
        self.val_l3 = nn.Linear(l2_n, l3_n)
        self.val_out = nn.Linear(l3_n, 1)
        # advantage layers
        self.adv_l3 = nn.Linear(l2_n, l3_n)
        self.adv_out = nn.Linear(l3_n, self.out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        features = F.relu(self.layer2(x))
        v = F.relu(self.val_l3(features))
        value = self.val_out(v)
        a = F.relu(self.adv_l3(features))
        advantages = self.adv_out(a)
        q_values = value + (advantages - advantages.mean())
        return q_values

class Agent:

    def __init__(
        self,
        env: gym.Env,
        lr: float = 2e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        bufsize: int = 10000,
        update_step: int = 4,
        logger: Logger | None = None
    ) -> None:
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.upd_step = update_step
        self.log = logger
        #
        obs_n = env.observation_space.shape[0]
        act_n = env.action_space.n
        self.replaybuf = ReplayBuffer(max_size = bufsize, obs_n = obs_n)
        #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        self.policy_net = DuelingDQN(obs_n, act_n, 64, 64, 64).to(self.device)
        self.target_net = DuelingDQN(obs_n, act_n, 64, 64, 64).to(self.device)
        #self.policy_net = DQN(obs_n, act_n, 64, 64).to(self.device)
        #self.target_net = DQN(obs_n, act_n, 64, 64).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.loss = nn.HuberLoss() #self.loss = nn.SmoothL1Loss()

    def get_action(
        self,
        state: torch.Tensor,
        eps: float
    ) -> int:
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        return self.env.action_space.sample().item()

    def optimize_model(self, batch_size: int) -> None:
        batch = self.replaybuf.sample(batch_size)
        if batch is None:
            return
        st_b, ac_b, rw_b, ns_b, dn_b = batch
        st = torch.FloatTensor(st_b).to(self.device)
        ac = torch.LongTensor(ac_b).to(self.device)
        rw = torch.FloatTensor(rw_b).to(self.device)
        ns = torch.FloatTensor(ns_b).to(self.device)
        dn = torch.LongTensor(dn_b).to(self.device)
        #
        Q = self.policy_net(st).gather(1, ac)
        Qn = self.target_net(ns).detach().max(1)[0]
        Qe = (rw + self.gamma * Qn * (1 - dn)).unsqueeze(1)
        #
        loss = self.loss(Q, Qe)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # soft update of the target network: θʹ ← τ·θ + (1-τ)·θʹ 
        tn_st = self.target_net.state_dict()
        pn_st = self.policy_net.state_dict()
        for key in pn_st:
            tn_st[key] = pn_st[key]*self.tau + tn_st[key]*(1-self.tau)
        self.target_net.load_state_dict(tn_st)

    def train(
        self,
        episode_n: int = 1000,
        batch_size: int = 100,
        ma_p: int = 100
    ) -> tuple[list[float], list[int]]:
        rewards = []
        rewards_ma = []
        rewards_w = deque(maxlen=ma_p)
        eps = EPS_S
        if self.log is not None:
            self.log.info('Start training the agent')
        for i in range(episode_n):
            st, _ = self.env.reset()
            st_t = torch.FloatTensor(st).to(self.device).unsqueeze(0)
            trw = 0.
            for t in count():
                a = self.get_action(st_t, eps)
                ns, rw, term, trun, _ = self.env.step(a)
                trw += rw
                done = term or trun
                self.replaybuf.push(st, a, rw, ns, term)
                st = ns
                st_t = torch.FloatTensor(st).to(self.device).unsqueeze(0)
                if (t+1) % self.upd_step == 0:
                    self.optimize_model(batch_size)
                if done:
                    break
            rewards_w.append(trw)
            rewards.append(trw)
            ma = np.mean(rewards_w)
            rewards_ma.append(ma)
            if self.log is not None:
                if (i+1) % 20 == 0:
                    li = 'Episode #{:d}: reward={:.2f}; avg.reward={:.2f}; eps={:.2f}'.format(i+1,trw,ma,eps)
                    self.log.info(li)
            eps = max(EPS_E, eps*EPS_DC)
        if self.log is not None:
            self.log.info('done.')
        return (rewards, rewards_ma)

    def save(self, fn: str) -> None:
        torch.save(self.policy_net.state_dict(), fn)
        if self.log is not None:
            self.log.info(f'Model saved to {fn}')

    def load(self, fn: str) -> None:
        if self.log is not None:
            self.log.info(f'Loading model from {fn}')
        self.policy_net.load_state_dict(torch.load(fn))
