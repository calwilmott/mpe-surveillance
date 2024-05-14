from multiagent.survey_environment import SurveyEnv
from BaseTrainer import BaseTrainer


observation_mode = "hybrid"
num_agents = 3
env = SurveyEnv(num_agents=num_agents, num_obstacles=4, vision_dist=0.2, grid_resolution=10, grid_max_reward=1,
                reward_delta=0.01, observation_mode=observation_mode, seed=None, reward_type="pov", world_filename=None)
base_trainer = BaseTrainer(env, observation_mode, num_agents, deep_discretization=False, is_render=True)
base_trainer.run()
