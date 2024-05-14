from DDQN.Agent import DDQNAgent
from DDQN.ReplayMemory import ReplayMemory
import tqdm
import numpy as np

class DDQNTrainerParams:
    def __init__(self):
        # RL Training parameters
        self.batch_size = 128
        self.num_steps = 500
        self.num_steps_memory = 500  # Number of steps taken by episodes while filling the memory buffer,
        # allows for a different value from num_steps to be used
        self.num_episodes = 2000

        # Callbacks
        self.save_interval = min(250, max(int(self.num_episodes) // 5, 10))  # Save interval is s=num_episodes//5,
        # as long as s is in the range (10, 250)
        self.eval_interval = 5

        # Replay Memory parameters
        self.rm_pre_fill_ratio = 0.6
        self.rm_size = 15000


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state, done):
        for i in range(np.shape(state)[0]):
            self.replay_memory.store((state[i], action[i], np.float64(reward[i]), next_state[i], done[i]))

    def train_agent(self):
        if self.params.batch_size > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        loss = self.agent.train(mini_batch)
        return loss

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True
