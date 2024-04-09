import numpy as np
import tqdm
import time
import copy
import os
import pickle
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
from DDQN.Agent import DDQNAgent, DDQNAgentParams
from DDQN.Trainer import DDQNTrainer, DDQNTrainerParams


class BaseTrainer:
    def __init__(self, environment, obs_mode, num_agents, deep_discretization=False, is_render=False):
        self.episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.env = environment
        self.num_agents = num_agents
        self.deep_discretization = deep_discretization
        self.is_render = is_render
        self.run_path = self.get_run_path()

        if deep_discretization:
            action_space = int(pow(self.env.action_space[0].n, 4))
        else:
            action_space = self.env.action_space[0].n
        example_obs = self.env.scenario.observation(self.env.agents[0], self.env.world)

        self.agent = DDQNAgent(params=DDQNAgentParams(), example_state=example_obs,
                               observation_mode=obs_mode, action_space=action_space,
                               num_agents=num_agents, deep_discretization=deep_discretization)
        self.trainer = DDQNTrainer(params=DDQNTrainerParams(), agent=self.agent)
        self.save_run_description()
        self.writer = tf.summary.create_file_writer(self.run_path + "/logs/")

    def get_run_path(self):
        if not os.path.exists("runs/"):
            os.mkdir("runs/")
            path = "runs/run0/"
            os.mkdir(path)
            return path
        else:
            i = 0
            while i < 1000:
                path = f"runs/run{i}/"
                if not os.path.exists(path):
                    os.mkdir(path)
                    print(f"\n---\nSTARTING RUN {i}\n---\n")
                    return path
                i += 1

        print("TOO MANY LOGS!")
        exit(1)

    def save_run_description(self):
        with open(self.run_path + "run_description.txt", "w") as f:
            f.write(self.run_path)
            f.write("\nTRAINER:\n")
            trainer = DDQNTrainerParams()
            for att in vars(trainer).items():
                f.write(f"{att[0]}: {att[1]}\n")
            f.write("\nAGENT:\n")
            agent = DDQNAgentParams()
            for att in vars(agent).items():
                f.write(f"{att[0]}: {att[1]}\n")
            f.write("\nENV:\n")
            for att in vars(self.env.scenario).items():
                f.write(f"{att[0]}: {att[1]}\n")
            f.write("\nDEEP DISCRETIZATION:\n")
            f.write(f"deep_discretization: {self.deep_discretization}\n")

    def run(self):
        self.fill_replay_memory()
        best_avg_ep_reward = 0
        best_run = None
        last_run = None
        ep_step_rewards = None
        bar = tqdm.tqdm(total=int(self.trainer.params.num_episodes))
        gc.collect()

        while self.episode_count < self.trainer.params.num_episodes:
            bar.update()
            ep_step_rewards, loss = self.train_episode()
            avg_ep_reward = np.average(ep_step_rewards)
            self.log_tensorboard(avg_ep_reward, ep_step_rewards, loss)
            last_run = ((self.env.world.obstacle_mask, self.env.world.reward_mask), self.agent)

            if avg_ep_reward > best_avg_ep_reward:
                best_avg_ep_reward = avg_ep_reward
                self.log_run(ep_step_rewards, self.episode_count)
                best_run = last_run

            if self.episode_count % self.trainer.params.eval_interval == 0:
                self.test_episode()

            if self.episode_count % self.trainer.params.save_interval == 0:
                self.save_run_to_disk(best_run)

        self.log_run(ep_step_rewards, self.episode_count, "last")
        self.save_run_to_disk(last_run, "last")

    def log_tensorboard(self, avg_ep_reward, ep_step_rewards, loss=None, title="train"):
        with self.writer.as_default():
            tf.summary.scalar(title + "/average", avg_ep_reward, step=self.episode_count)
            tf.summary.scalar(title + "/max", np.max(ep_step_rewards), step=self.episode_count)
            tf.summary.scalar(title + "/final_step", ep_step_rewards[-1], step=self.episode_count)

            if title == "train":
                tf.summary.scalar(title + "/loss", loss, step=self.episode_count)

    def save_run_to_disk(self, run, title="best"):
        world = run[0]
        with open(self.run_path + title + "_world.pickle", 'wb') as file:
            pickle.dump(world, file)

        agent = run[1]
        agent.save_weights(self.run_path + title + "_weights")

    def log_run(self, rewards, episode, title="best"):
        run_path = self.run_path + title + ".png"
        if os.path.exists(run_path):
            os.remove(run_path)

        plt.plot(rewards)
        caps_title = title[0].upper() + title[1:]
        plt.title(f"{caps_title} run ({episode})")
        plt.savefig(run_path)
        plt.clf()
        plt.close()

    def test_episode(self, render_test=False):
        self.step_count = 0
        state = self.env.reset()
        done = False
        ep_step_rewards = []

        while not done:
            state, reward, done = self.step(state, is_test=True)
            step_reward = sum(reward) / self.num_agents
            ep_step_rewards.append(step_reward)

            if render_test:
                self.env.render()
                time.sleep(0.0001)

        avg_ep_reward = np.average(ep_step_rewards)
        self.log_tensorboard(avg_ep_reward, ep_step_rewards, title="test")

        return ep_step_rewards

    def fill_replay_memory(self):
        while self.trainer.should_fill_replay_memory():
            self.step_count = 0
            state = self.env.reset()
            while True:
                self.step(state, is_filling_memory=True)
                if self.step_count >= self.trainer.params.num_steps_memory:
                    break

    def step(self, state, is_filling_memory=False, is_test=False):
        finished = False
        if is_filling_memory:
            action = self.agent.get_random_action()
        elif is_test:
            action = self.agent.get_exploitation_action_target(state)
        else:
            action = self.agent.act(state)

        # Adjusts dimensions of array if needed
        if np.shape(action)[0] == 7:
            action = [action]

        # Converts action if deep discretization is being used
        if self.deep_discretization:
            env_action = self.get_deep_action(action)
        else:
            env_action = action

        next_state, reward, done, info = self.env.step(env_action)
        self.step_count += 1

        if not is_test:
            self.trainer.add_experience(state, action, reward, next_state, done)

        # Uses map reward to allow for a fair comparison between reward types
        if self.env.scenario.reward_type != "map":
            single_reward = self.env.scenario.get_map_reward(self.env.world)
            reward = [single_reward] * self.num_agents

        if self.step_count >= self.trainer.params.num_steps:
            finished = True

        if self.is_render and not is_filling_memory and not is_test:
            self.env.render()
            time.sleep(0.0001)

        return copy.deepcopy(next_state), reward, finished

    def train_episode(self):
        self.step_count = 0
        loss = None
        state = self.env.reset()
        done = False
        ep_step_rewards = []

        while not done:
            state, reward, done = self.step(state)
            step_reward = sum(reward) / self.num_agents
            ep_step_rewards.append(step_reward)
            loss = self.trainer.train_agent()

        self.episode_count += 1

        return ep_step_rewards, loss

    def get_deep_action(self, deep_action):
        action_space = self.env.action_space[0].n
        action_array = np.zeros((self.num_agents, action_space))

        for i in range(len(deep_action)):
            action_num = np.nonzero(deep_action)[1][i]
            for j in range(3, -1, -1):
                index = action_num // int(pow(action_space, j))
                action_num = action_num % int(pow(action_space, j))
                action_array[i][index] += 0.25

        return action_array
