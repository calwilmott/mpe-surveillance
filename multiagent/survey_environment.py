import copy

import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.simple_survey_region import SurveyScenario
import pickle

class SurveyEnv(MultiAgentEnv):
    def __init__(self, num_obstacles: int = 4, num_agents: int = 3, vision_dist: float = 0.5, grid_resolution: int = 10,
                 grid_max_reward: float = 1.0, reward_delta: float = 0.0001, observation_mode: str = "image",
                 seed: int = None, reward_type: str = "pov", collaborative=False, world_filename=None):
        """
        Initializes the Survey environment with the specified configuration.

        Parameters:
            num_obstacles (int): The number of obstacles to be placed in the environment. These obstacles obstruct the agents' paths and limit their ability to navigate and view grid squares.
            num_agents (int): The number of agents to be included in the environment. Each agent can interact with the environment.
            vision_dist (float): The maximum distance at which an agent can perceive its surroundings. This affects the length of the 'red line' in the rendered simulation.
            grid_resolution (int): The resolution of the grid used to represent the environment. A higher resolution means a finer grid with more cells.
            grid_max_reward (float): The maximum reward value that a grid cell can have. This value is used to initialize the reward distribution in the environment, and to cap reward from increasing indefinitely.
            reward_delta (float): The amount by which the reward of a grid cell increases after each time step. This can be used to control the frequency with which an optimal policy would dictate than an agent should re-visit a grid square.
            observation_mode (str): The mode of observation for the agents. This can determine how agents perceive their environment, e.g., as raw pixel values ("image") or as processed features ("dense").
                                    Select the "hybrid" mode for a combination of both feature representations.
            seed (int): The seed for random generation of obstacles and areas of no interest.
            reward_type (str): "pov" or "map". Determines if the reward will be calculated by cells seen by the agent at the current step or the map.

        Returns:
            None
        """

        # Load the scenario with the specified parameters
        self.scenario = SurveyScenario(num_obstacles, num_agents, vision_dist, grid_resolution, grid_max_reward,
                                       reward_delta, observation_mode, seed, reward_type, collaborative, world_filename)

        # Create the world
        if world_filename is None:
            world = self.scenario.make_world()
            reset_callback = self.scenario.reset_world
            self.loaded_world = None
        else:
            self.load_world(world_filename)
            world = self.world
            reset_callback = self.reload_world

        if observation_mode == "image":
            obs_shape = self.scenario._get_img_obs(world.agents[0], world).shape
        elif observation_mode == "upscaled_image":
            obs_shape = self.scenario._get_upscaled_img_obs(world.agents[0], world).shape
        else:  # "hybrid". "dense" does not use obs_shape
            obs_shape = self.scenario._get_img_obs(world.agents[0], world)[:, :, 4:].shape

        # Initialize the parent class with the necessary functions
        obs_shape = self.scenario._get_img_obs(world.agents[0], world).shape

        super().__init__(world, reset_callback, self.scenario.reward, self.scenario.observation,
                         observation_mode=observation_mode, observation_shape=obs_shape)

    def save_world(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.world, f)

    def load_world(self, filename):
        with open(filename, 'rb') as f:
            self.world = pickle.load(f)
            self.loaded_world = copy.deepcopy(self.world)

    def reload_world(self, _):
        self.world = copy.deepcopy(self.loaded_world)

        self.world.grid = np.ones((self.scenario.grid_resolution, self.scenario.grid_resolution))

        # Random properties for agents
        for i, agent in enumerate(self.world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # Initialize agent position
            self.scenario.initialize_agent_position(agent, self.world)
            self.scenario.reward(agent, self.world)