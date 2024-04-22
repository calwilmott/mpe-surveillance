import time
import numpy as np

from multiagent.survey_environment import SurveyEnv
from BaseTrainer import BaseTrainer


def get_value_from_file(value, type, line):
    if type == str:
        # Removes \n from strings
        value = type(line[-1][:-1])
    elif type == bool:
        # Checks for booleans values
        value = line[-1][0] == "T"
    elif line[-1][0] == "N" and (type == int or value is None):
        # Value in file is None
        value = None
    elif value is None:
        # Special condition for world_filename
        value = str(line[-1][:-1])
    else:
        value = type(line[-1])

    return value


run_number = input("Run number (ex: 5):\n")
multiple_runs = input("Execute multiple runs? (y/N)\n")

params = {
    "num_agents": 1,
    "reward_delta": 0.001,
    "observation_mode": "hybrid",
    "reward_type": "pov",
    "deep_discretization": False,
    "original_seed": 81,
    "world_filename": None
}

with open("runs/run" + run_number + "/run_description.txt") as f:
    for line in f:
        for key in params.keys():
            if key in line:
                line_values = line.split(" ")
                param_type = type(params[key])
                params[key] = get_value_from_file(params[key], param_type, line_values)

env = SurveyEnv(num_agents=params["num_agents"], num_obstacles=4, vision_dist=0.2, grid_resolution=10,
                grid_max_reward=1, reward_delta=params["reward_delta"], observation_mode=params["observation_mode"],
                seed=params["original_seed"], reward_type=params["reward_type"],
                world_filename=params["world_filename"])
base_trainer = BaseTrainer(env, params["observation_mode"], params["num_agents"],
                           deep_discretization=params["deep_discretization"], is_render=True, is_test=True)

base_trainer.agent.load_weights("runs/run" + run_number + "/best_weights")

if multiple_runs.lower() == "y":
    ep_rewards = []
    for i in range(10):
        avg_ep_reward = np.average(base_trainer.test_episode(render_test=True))
        if avg_ep_reward < 11:
            print("BUG DETECTED!")
            exit(1)
        ep_rewards.append(avg_ep_reward)

    ep_rewards.remove(np.min(ep_rewards))
    ep_rewards.remove(np.max(ep_rewards))
    print("AVG:", np.average(ep_rewards))
else:
    base_trainer.test_episode(render_test=True)
