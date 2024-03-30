import numpy as np
import time
from multiagent.survey_environment import SurveyEnv
import queue


# np.random.seed(1086)
# np.random.seed(10077988)


class GridWorld:
    def __init__(self, observation):
        self.obstacles = observation[0][:, :, 1]
        self.other_agents_pos = np.where(observation[0][:, :, 0] == 1)

        self.walls = np.argwhere((self.obstacles == 1) | (self.other_agents_pos == 1))
        self.waypoints = np.argwhere((self.obstacles == 0 )| (self.other_agents_pos == 1))
        self.walls = [tuple(idx) for idx in self.walls]
        self.agent_pos = np.where(observation[0][:, :, 0] == 1)
        self.width = self.obstacles.shape[0]
        self.height = self.obstacles.shape[1]

        self.grid = np.where(self.obstacles == 0, '.', '#')
        self.grid[self.agent_pos] = 'X'
        self.grid[self.other_agents_pos] = '@'

        self.grid[0][0] = 'G'

    def in_bounds(self, point):
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    def is_passable(self, point):
        return point not in self.walls

    def neighbors(self, point):
        (x, y) = point
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

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


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



def is_goal(point, goal):
    '''
    Checks if a point is the goal.
    '''
    print('is goal: ', point, goal)
    return point == goal

def get_path(grid, start_pos, goal_pos):
    start_pos = tuple(arr.item() for arr in start_pos)
    # print('start pos: ', start_pos)
    frontier = queue.PriorityQueue()
    frontier.put((start_pos, 0))
    from_cell = dict()
    cost_so_far = dict()

    from_cell[start_pos] = None
    cost_so_far[start_pos] = 0

    while not frontier.empty():
        current = frontier.get()
        current = current[0]

        if current == goal_pos:
            break
        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + grid.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + euclidean_distance(next, goal_pos)
                frontier.put((next, priority))
                from_cell[next] = current

    return from_cell, cost_so_far



def create_path(from_dict, start_pos, goal_pos):
    current_pos = goal_pos

    path = []
    # print(from_dict.keys())
    while current_pos != start_pos:
        path.append(from_dict.get(current_pos))
        current_pos = from_dict[current_pos]

    return path



def translate_to_step(current_pos, goal_pos):
    x1, y1 = current_pos
    x2, y2 = goal_pos

    # get deltas in x and y coordinate positions
    dx = x2 - x1
    dy = y2 - y1

    action_x, action_y, action_torque = 0, 0, 0

    # agent movement for left or right
    if abs(dx) > abs(dy):
        if dx > 0:
            # right
            action_x = -1
            direc = 'right'

        else:
            # left
            action_x = 1
            direc = 'left'

        action_y = 0
    else:
        if dy > 0:
            # up
            action_y = -1
            direc = 'up'

        else:
            # down
            action_y = 1
            direc = 'down'

        action_x = 0
    action_torque = 0
    return np.array([[0, 0, action_x, 0,  action_y, action_torque, 0]]), direc


def update_current_pos(current_pos_x, current_pos_y):
    policy_pos_x, policy_pos_y = None, None
    if np.negative(current_pos_y):
        policy_pos_y = np.around(current_pos_y - 0.21, 2)
    elif np.positive(current_pos_y):
        policy_pos_y = np.around(current_pos_y + 0.21, 2)

    if np.negative(current_pos_x):
        policy_pos_x = np.around(current_pos_x - 0.2, 2)
    elif np.positive(current_pos_x):
        policy_pos_x = np.around(current_pos_x + 0.2, 2)


    return policy_pos_x, policy_pos_y



def is_center_of_cell(agent_pos_x, agent_pos_y, direction):
    cell_width = 0.2
    cell_height = 0.8
    agent_radius = 0.05  # Assuming agent's radius is 0.05

    # Calculate the center coordinates of the cell
    cell_center_x = cell_width / 2
    cell_center_y = cell_height / 2

    # Adjust the agent's position based on its radius
    adjusted_agent_pos_x = agent_pos_x - agent_radius
    adjusted_agent_pos_y = agent_pos_y - agent_radius

    # Calculate the cell coordinates of the adjusted agent's position
    cell_x = int(adjusted_agent_pos_x * 10)  # Assuming a 10x10 grid
    cell_y = int(adjusted_agent_pos_y * 10)

    # Calculate the distance from the agent's position to the center of the cell
    distance_x = np.floor(abs(adjusted_agent_pos_x - (cell_x / 10 + cell_center_x)))
    distance_y = np.floor(abs(adjusted_agent_pos_y - (cell_y / 10 + cell_center_y)))

    # distance_x = np.ceil(adjusted_agent_pos_x - (cell_x / 10 + cell_center_x))
    # distance_y = np.ceil(adjusted_agent_pos_y - (cell_y / 10 + cell_center_y))

    # Check if the distance is less than or equal to half of the cell size
    if direction == 'left' or direction == 'right':
        return distance_x <= cell_width / 2  # Check if within half the cell width
    if direction == 'up' or direction == 'down':
        return distance_y <= cell_height / 2  # Check if within half the cell height





env = SurveyEnv(num_agents=1, num_obstacles=9, vision_dist=0.2, grid_resolution=10, grid_max_reward=1, reward_delta=0.001, observation_mode="upscaled_image")
env.reset()

action_n = np.zeros((1,7))
not_at_goal = True
goal_pos = (1, 1)
# env.agents[0].state.p_pos[0] = 0.9
# env.agents[0].state.p_pos[1] = 0.9

first_pass = True
while not_at_goal:

    obs, rew, done, info = env.step(action_n)
    obs_grid = obs[0][:,:,1]
    agent_pos = obs[0][:,:,0]
    # print(obs[0].shape)

    gw = GridWorld(obs)
    grid_pos = tuple([int(index) for index in list(np.where(agent_pos == 1))])

    print('AGENT POS: ', np.where(agent_pos == 1))

    ideal_path_dict, cost_path = get_path(gw, gw.agent_pos, goal_pos)
    ideal_path = create_path(ideal_path_dict, grid_pos, goal_pos)
    ideal_path.reverse()
    ideal_path.append(goal_pos)
    # print(gw.render(ideal_path))


    if len(ideal_path) == 0:
        break

    if len(ideal_path) > 2:
        ideal_action, direction = translate_to_step(grid_pos, ideal_path[1])
        next_pos = ideal_path[2]
        del ideal_path[0]

    else:
        del ideal_path[0]
        ideal_action, direction = translate_to_step(grid_pos, ideal_path[0])
        next_pos = goal_pos

    if first_pass:
        action_n = ideal_action
        first_pass = False
        prev_action = action_n

    if is_center_of_cell(env.agents[0].state.p_pos[0], env.agents[0].state.p_pos[1], direction):
        action_n = ideal_action

        # print('here')
    else:
        # print('rip')

        action_n = prev_action

    prev_action = action_n

    env.render()
    time.sleep(.1)