from multiagent.survey_environment import SurveyEnv
from BaseTrainer import BaseTrainer

run_number = input("Run number (ex: 5):\n")

params = {
    "num_agents": 1,
    "reward_delta": 0.001,
    "observation_mode": "hybrid",
    "reward_type": "pov",
    "deep_discretization": False
}

with open("runs/run" + run_number + "/run_description.txt") as f:
    for line in f:
        for key in params.keys():
            if key in line:
                line_values = line.split(" ")
                param_type = type(params[key])
                if param_type == str:
                    # Removes \n from strings
                    params[key] = param_type(line_values[-1][:-1])
                else:
                    params[key] = param_type(line_values[-1])

env = SurveyEnv(num_agents=params["num_agents"], num_obstacles=4, vision_dist=0.2, grid_resolution=10,
                grid_max_reward=1, reward_delta=params["reward_delta"], observation_mode=params["observation_mode"],
                seed=81, reward_type=params["reward_type"])
base_trainer = BaseTrainer(env, params["observation_mode"], params["num_agents"],
                           deep_discretization=params["deep_discretization"], is_render=True, is_test=True)

base_trainer.agent.load_weights("runs/run" + run_number + "/best_weights")
base_trainer.test_episode(render_test=True)
