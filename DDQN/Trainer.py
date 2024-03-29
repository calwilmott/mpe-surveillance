from DDQN.Agent import DDQNAgent
from DDQN.ReplayMemory import ReplayMemory
import tqdm
import numpy as np

class DDQNTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 100#1e6
        self.num_steps_memory = 1000
        self.num_episodes = 5#1e5
        self.save_interval = min(5000, self.num_episodes // 5)
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 10000#1000000
        self.load_model = ""


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent

        if self.params.load_model != "":
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state, done):
        for i in range(np.shape(state)[0]):
            self.replay_memory.store((state[i], action[i], np.float64(reward[i]), next_state[i], done[i]))

    def train_agent(self):
        if self.params.batch_size > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        self.agent.train(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            #print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True
