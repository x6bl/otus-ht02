import gymnasium as gym
from itertools import count
import logging
import torch
import yaml
from dqn import Agent


# Load config
with open('.config.yml', 'r') as cf:
    config = yaml.safe_load(cf)
# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(config['logger']['run_log_path'])
fh.setLevel(logging.INFO)
fh.setFormatter(logger_formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logger_formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# Setup environment
env = gym.make('LunarLander-v2', render_mode='human')
# Agent parameters
fname = config['agent']['save_to']
# Load model
agent = Agent(env, logger=logger)
agent.load(fname)
# run Agent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPISODE_N = 5
logger.info('Run the Agent for {:d} episodes'.format(EPISODE_N))
for i in range(EPISODE_N):
    trw = 0
    st, _ = env.reset()
    st_t = torch.FloatTensor(st).to(device).unsqueeze(0)
    for j in count():
        ac = agent.get_action(st_t, 0.)
        env.render()
        st, rw, term, trun, _ = env.step(ac)
        st_t = torch.FloatTensor(st).to(device).unsqueeze(0)
        trw += rw
        if term or trun:
            break
    logger.info('Episode {:d}: Total Reward is {:.2f}'.format(i+1, trw))
env.close()

