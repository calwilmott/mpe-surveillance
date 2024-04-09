from multiagent.survey_environment import SurveyEnv
from BaseTrainer import BaseTrainer

observation_mode = "hybrid"
num_agents = 1
env = SurveyEnv(num_agents=num_agents, num_obstacles=4, vision_dist=0.2, grid_resolution=10, grid_max_reward=1,
                reward_delta=0.001, observation_mode=observation_mode, seed=81, reward_type="map")
base_trainer = BaseTrainer(env, observation_mode, num_agents, deep_discretization=True, is_render=True)
weights_path = input("Relative path to weights:\n")
base_trainer.agent.load_weights(weights_path)
base_trainer.test_episode(render_test=True)
