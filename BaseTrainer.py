import numpy as np
import tqdm
import time
import copy
import os
import pickle
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from DDQN.Agent import DDQNAgent, DDQNAgentParams
from DDQN.Trainer import DDQNTrainer, DDQNTrainerParams


class BaseTrainer:
    def __init__(self, environment, obs_mode, is_render=False):
        self.episode_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.is_render = is_render
        self.env = environment
        self.run_path = self.get_run_path()

        action_space = self.env.action_space[0]
        example_obs = self.env.scenario.observation(self.env.agents[0], self.env.world)
        self.agent = DDQNAgent(params=DDQNAgentParams(), example_state=example_obs, observation_mode=obs_mode,
                               action_space=action_space)
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
            i = 1
            while i < 1000:
                path = f"runs/run{i}/"
                if not os.path.exists(path):
                    os.mkdir(path)
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

    def run(self):
        self.fill_replay_memory()
        bar = tqdm.tqdm(total=int(self.trainer.params.num_episodes))
        last_ep = 0
        best_avg_ep_reward = 0
        best_run = None
        while self.episode_count < self.trainer.params.num_episodes:
            bar.update(self.episode_count - last_ep)
            last_ep = self.episode_count
            avg_ep_reward, sum_ep_reward, final_step_reward, ep_step_rewards = self.train_episode()
            with self.writer.as_default():
                tf.summary.scalar("average", avg_ep_reward, step=self.episode_count)
                tf.summary.scalar("sum", sum_ep_reward, step=self.episode_count)
                tf.summary.scalar("final_step", final_step_reward, step=self.episode_count)

            if avg_ep_reward > best_avg_ep_reward:
                best_avg_ep_reward = avg_ep_reward
                self.log_best_run(ep_step_rewards, self.episode_count)
                best_run = ((self.env.world.obstacle_mask, self.env.world.reward_mask), self.agent)

            if self.episode_count % self.trainer.params.save_interval == 0:
                self.save_best_run_to_disk(best_run)

    def save_best_run_to_disk(self, best_run):
        world = best_run[0]
        with open(self.run_path + "world.pickle", 'wb') as file:
            pickle.dump(world, file)

        agent = best_run[1]
        agent.save_model(self.run_path + "model")
        agent.save_weights(self.run_path + "weights")

    def log_best_run(self, rewards, iter):
        best_run_path = self.run_path + "best.png"
        if os.path.exists(best_run_path):
            shutil.rmtree(best_run_path)

        plt.plot(rewards)
        plt.title(f"Best run ({iter})")
        plt.savefig(best_run_path)
        plt.clf()
        plt.close()

    def fill_replay_memory(self):
        while self.trainer.should_fill_replay_memory():
            self.step_count = 0
            state = self.env.reset()
            while True:
                self.step(state, random=self.trainer.params.rm_pre_fill_random, is_filling_memory=True)
                if self.step_count >= self.trainer.params.num_steps_memory:
                    break

    def step(self, state, random=False, is_filling_memory=False):
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

        if self.is_render and not is_filling_memory:
            self.env.render()
            time.sleep(0.0001)

        return copy.deepcopy(next_state), reward, finished

    def train_episode(self):
        self.step_count = 0
        state = self.env.reset()
        done = False
        ep_step_rewards = []
        while not done:
            state, reward, done = self.step(state)
            step_reward = sum(reward) / 3.0
            ep_step_rewards.append(step_reward)
            self.trainer.train_agent()

        self.episode_count += 1
        avg_episode_reward = np.average(ep_step_rewards)
        sum_episode_reward = sum(ep_step_rewards)
        final_step_reward = step_reward

        return avg_episode_reward, sum_episode_reward, final_step_reward, ep_step_rewards
