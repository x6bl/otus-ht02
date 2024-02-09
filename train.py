import gymnasium as gym
import logging
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dueling_dqn import Agent


# load config
with open('.config.yml', 'r') as cf:
    config = yaml.safe_load(cf)
# setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(config['logger']['train_log_path'])
fh.setLevel(logging.INFO)
fh.setFormatter(logger_formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logger_formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# Taxi environment
env = gym.make('LunarLander-v2')
# agent parameters
fname = config['agent']['save_to']
episode_n = config['agent']['episode_n']
lr = config['agent']['learning_rate']
g = config['agent']['gamma']
tau = config['agent']['tau']
bs = config['agent']['buffer_size']
ma_p = config['agent']['ma_period']
# learn agent with Q-Learning algorithm (default)
agent = Agent(env, lr=lr, gamma=g, tau=tau, bufsize=bs, logger=logger)
#
er, rma = agent.train(episode_n = episode_n, ma_p = ma_p)
# draw rewards plot
plt.figure(figsize=(12.8, 8))
en = np.arange(len(er))
plt.plot(en, er, color='blue', label='Rewards')
plt.plot(en, rma, color='orange', label='MA-{:d}'.format(ma_p))
#plt.plot(el, color='red', label=f'Rewards Moving Average (period={ma_p})')
plt.title('Agent Learning Dynamics for LunarLander-v2 Environment')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()
# Save agent Q-function
agent.save(fname)

