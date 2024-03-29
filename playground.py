import numpy as np
import time
import copy
import tqdm
import matplotlib.pyplot as plt
from multiagent.survey_environment import SurveyEnv
from BaseTrainer import BaseTrainer
from DDQN.Trainer import DDQNTrainer, DDQNTrainerParams
from DDQN.Agent import DDQNAgent, DDQNAgentParams


observation_mode = "hybrid"
env = SurveyEnv(num_agents=3, num_obstacles=4, vision_dist=0.2, grid_resolution=10, grid_max_reward=1,
                reward_delta=0.001, observation_mode=observation_mode)
base_trainer = BaseTrainer(env, observation_mode, is_render=True)
base_trainer.run()


# counter = 0

# while True:
#     # first dimension in action space is number of agents, second is action space
#     action_n = np.random.random(size=(3, 7))
#     obs, rew, done, info = env.step(action_n)
#
#     # if counter % 50 == 0 or counter == 1:
#     #     obs_array = np.array(obs[0])
#     #     for i in range(7):
#     #         plt.imshow(np.flipud(obs_array[:, :, i].T), cmap='Greys',  interpolation='nearest')
#     #         plt.show()
#     #     input()
#     counter += 1
#     # print(obs)
#     # print(type(obs))
#     env.render()
#     # visualize_image_observation(obs[0])
#     time.sleep(0.1)
