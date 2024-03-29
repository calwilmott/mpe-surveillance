import numpy as np
import time
from multiagent.survey_environment import SurveyEnv
import queue
from matplotlib import pyplot as plt
from multiagent.utils.visualization import visualize_image_observation
np.random.seed(102)

class GridWorld:
    def __init__(self, observation):
        self.obstacles = observation[0][:, :, 4]
        self.other_agents_pos = np.where(observation[0][:, :, 2] == 1)

        self.walls = np.argwhere((self.obstacles == 1) | (self.other_agents_pos == 1))

        self.walls = [tuple(idx) for idx in self.walls]
        self.agent_pos = np.where(observation[0][:, :, 0] == 1)
        self.width = self.obstacles.shape[0]
        self.height = self.obstacles.shape[1]

        self.grid = np.where(self.obstacles == 0, '.', '#')
        self.grid[self.agent_pos] = 'X'
        self.grid[self.other_agents_pos] = '@'

        self.grid[9][0] = 'G'

    def in_bounds(self, point):
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    def is_passable(self, point):
        return point not in self.walls

    def neighbors(self, point):
        (x, y) = point
        # Centers of the neighboring cells
        neighbors = [(x - 1 + 0.5, y + 0.5), (x + 1 + 0.5, y + 0.5), 
                     (x + 0.5, y - 1 + 0.5), (x + 0.5, y + 1 + 0.5)]
        if (x + y) % 2 == 0: 
            neighbors.reverse()
        neighbors = filter(self.in_bounds, neighbors)
        neighbors = filter(self.is_passable, neighbors)
        return neighbors

    def render(self, path):
        for point in path:
            if self.grid[int(point[0])][int(point[1])] != 'X':
                self.grid[int(point[0])][int(point[1])] = '*'
        return np.flip(self.grid, axis=1).T

    def cost(self, current, next):
        if next in self.walls:
            return 1000
        else:
            return 1




def manhattan_distance(point1, point2):
    '''
    Calculates the Manhattan distance between two points.
    '''
    return ( abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]) )



def is_goal(point, goal):
    '''
    Checks if a point is the goal.
    '''
    print('is goal: ', point, goal)
    return point == goal

def get_path(grid, start_pos, goal_pos):
    start_pos = tuple(arr.item() for arr in start_pos)
    frontier = queue.PriorityQueue()
    frontier.put((start_pos, 0))
    from_cell = dict()
    cost_so_far = dict()

    from_cell[start_pos] = None
    cost_so_far[start_pos] = 0

    while not frontier.empty():
        current = frontier.get()
        current = current[0]
        # print('CURRENT ', current)

        if current == goal_pos:
            break
        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + grid.cost(current, next)
            # new_cost = cost_so_far[current]
            # print('NEW COST', new_cost)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + manhattan_distance(next, goal_pos)
                # print('priority: ', priority)
                frontier.put((next, priority))
                from_cell[next] = current


    # print(from_cell, cost_so_far)
    return from_cell, cost_so_far



def create_path(from_dict, start_pos, goal_pos):
    current_pos = goal_pos

    path = []

    while current_pos != start_pos:
        path.append(from_dict.get(current_pos))
        current_pos = from_dict[current_pos]
    # print(path)
    return path



def translate_to_step(current_pos, goal_pos):
    x1, y1 = current_pos
    x2, y2 = goal_pos

    print(current_pos, goal_pos)

    # get deltas in x and y coordinate positions
    dx = x2 - x1
    dy = y2 - y1

    action_x, action_y, action_torque = 0, 0, 0

    # agent movement for left or right
    if abs(dx) > abs(dy):
        if dx > 0:
            # right
            action_x = -1
            action = 2
            print('right')
        else:
            # left
            action_x = 1
            action = 1

            print('left')
        action_y = 0
    else:
        print(dy, dx)
        if dy > 0:
            # up
            action_y = -1
            action = 3

            print('up')
        else:
            # down
            action_y = 1
            action = 4

            print('down')
        action_x = 0
    action_torque = 0
    return np.array([[0, 0, action_x, 0,  action_y, action_torque, 0]])
    # return np.array([action_x, action_y])
    # return np.array([action])





env = SurveyEnv(num_agents=1, num_obstacles=10, vision_dist=0.2, grid_resolution=10, grid_max_reward=1, reward_delta=0.001, observation_mode="image")
env.reset()

# action_n = np.array([2])
action_n = np.zeros((1,7))

# action_n = np.random.random(size=(1, 7))

i = 0
not_at_goal = True
# while i < 2:
prev_pos = None
while not_at_goal:
    # print('ACTION N: ', action_n)
    obs, rew, done, info = env.step(action_n)

    obs_grid = obs[0][:,:,4]
    agent_pos = obs[0][:,:,0]

    

    start_pos = tuple([int(index) for index in list(np.where(agent_pos == 1))])
    goal_pos = (9,0)
    prev_pos = start_pos

    if start_pos == goal_pos: not_at_goal = False
    # print('Agent Position: ', tuple([int(index) for index in list(np.where(agent_pos == 1))]))

    gw = GridWorld(obs)
    ideal_path_dict, cost_path = get_path(gw, gw.agent_pos, goal_pos)
    ideal_path = create_path(ideal_path_dict, start_pos, goal_pos)

    print(gw.render(ideal_path))

    if len(ideal_path) > 0:
        # print(ideal_path)
        # break
        ideal_action = translate_to_step(start_pos, ideal_path[1])
        # ideal_action = np.array([[ 0, 0, 5, 0, 0, 0,  0]])
        action_n = ideal_action
        next_pos = ideal_path[0]
        # print(action_n)
        # env.agents[0].action.u = ideal_action
        # action_n = ideal_action

        # action_n = np.random.random(size=(1, 7))
        # print('second if')

        # env.agents[0].action.u[0] = ideal_action[0]
        # env.agents[0].action.u[1] = ideal_action[1]
        #
        #
        # # ADDING TORQUE ACTIONS
        # env.agents[0].action.u[2] = ideal_action[2]

        # env.agents[0].action.c[0] = ideal_action[2]

        # action_n = env.agents[0].action.u + env.agents[0].action.c

        # print('Ideal Path: ', ideal_path)
        # print('action: ', action_n.__doc__())
        # print('other action 1: ', env.agents[0].action.u)
        # print('other action 2: ', env.agents[0].action.c)
        print('ideal action: ', ideal_action)

    print(prev_pos, next_pos)
    if prev_pos == next_pos:
        print('Goal reached')
        break

    env.render()
    time.sleep(.1)

    i += 1
    # break
