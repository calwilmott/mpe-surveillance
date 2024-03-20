from DDQN.Agent import DDQNAgent, DDQNAgentParams
from DDQN.Trainer import DDQNTrainer, DDQNTrainerParams
import tqdm
import time
import copy
import matplotlib.pyplot as plt


class BaseTrainer:
    def __init__(self, environment, obs_mode, is_render=False):
        self.episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.is_render = is_render
        self.env = environment

        action_space = self.env.action_space[0]
        example_obs = self.env.scenario.observation(self.env.agents[0], self.env.world)
        self.agent = DDQNAgent(params=DDQNAgentParams(), example_state=example_obs, observation_mode=obs_mode,
                               action_space=action_space)
        self.trainer = DDQNTrainer(params=DDQNTrainerParams(), agent=self.agent)

    def run(self):
        self.fill_replay_memory()
        bar = tqdm.tqdm(total=int(self.trainer.params.num_episodes))
        last_ep = 0
        rewards = []
        while self.episode_count < self.trainer.params.num_episodes:
            bar.update(self.episode_count - last_ep)
            last_ep = self.episode_count
            reward = self.train_episode()
            rewards.append(reward)

        self.env.render()
        plt.plot(rewards)
        plt.show()
        input()

    def fill_replay_memory(self):
        while self.trainer.should_fill_replay_memory():
            self.step_count = 0
            state = self.env.reset()
            while True:
                self.step(state, random=self.trainer.params.rm_pre_fill_random)
                if self.step_count >= self.trainer.params.num_steps_memory:
                    break

    def step(self, state, random=False):
        finished = False
        if random:
            action = self.agent.get_random_action()
        else:
            action = self.agent.act(state)
        next_state, reward, done, info = self.env.step(action)
        self.trainer.add_experience(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count >= self.trainer.params.num_steps:
            finished = True

        if self.is_render:
            self.env.render()
            time.sleep(0.001)

        return copy.deepcopy(next_state), reward, finished

    def train_episode(self):
        self.step_count = 0
        state = self.env.reset()
        done = False
        while not done:
            state, reward, done = self.step(state)
            self.trainer.train_agent()

        self.episode_count += 1
        final_reward = sum(reward)

        return final_reward
