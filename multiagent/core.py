import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        self.p_angle = None
        self.p_angle_vel = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # moment of inertia
        self.initial_moi = 1.0

    @property
    def mass(self):
        return self.initial_mass
    @property
    def moi(self):
        return self.initial_moi
class Obstacle(Entity):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.shape = 'rectangle'  # default shape
        self.width = 0.1  # default width
        self.height = 0.1  # default height
        self.movable = False
# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # Vision distance
        self.vision_dist = 0.2

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.obstacles = []  # list of obstacles

        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.9
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a][:2] = f_a + p_force[a][:2]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b][:2] = f_b + p_force[b][:2]      
        return p_force

        # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue

            # Apply damping and update velocity
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i][:2] / entity.mass) * self.dt

            # Limit velocity to max speed
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed

            # Compute proposed new position
            new_pos = entity.state.p_pos + entity.state.p_vel * self.dt

            # Check for collision with any obstacle
            for obstacle in self.obstacles:
                if self.is_collision_rectangular(entity, obstacle, new_pos):
                    # Prevent the entity from moving into the obstacle
                    # Adjust the velocity to zero in the direction of the collision
                    ox, oy = obstacle.state.p_pos
                    half_width, half_height = obstacle.width / 2, obstacle.height / 2

                    # Determine which axis the collision occurred on
                    if abs(new_pos[0] - ox) < half_width + entity.size:
                        entity.state.p_vel[0] = 0  # Stop movement in x-axis
                    if abs(new_pos[1] - oy) < half_height + entity.size:
                        entity.state.p_vel[1] = 0  # Stop movement in y-axis

                    # Recompute new_pos with adjusted velocity
                    new_pos = entity.state.p_pos + entity.state.p_vel * self.dt

            # Update position and clip to bounds
            entity.state.p_pos = new_pos
            entity.state.p_pos = np.clip(entity.state.p_pos, -1, 1)

            # Apply damping and update angular velocity
            entity.state.p_angle_vel = entity.state.p_angle_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_angle_vel += (p_force[i][2] / entity.moi) * self.dt
                        # Compute proposed new position
            entity.state.p_angle += (entity.state.p_angle_vel * self.dt)
            entity.state.p_angle %= (2 * np.pi)


    def is_collision_rectangular(self, agent, obstacle, new_pos):
        ax, ay = new_pos
        agent_radius = agent.size
        ox, oy = obstacle.state.p_pos
        half_width, half_height = obstacle.width / 2, obstacle.height / 2

        left_bound = ox - half_width - agent_radius
        right_bound = ox + half_width + agent_radius
        bottom_bound = oy - half_height - agent_radius
        top_bound = oy + half_height + agent_radius

        return (left_bound <= ax <= right_bound) and (bottom_bound <= ay <= top_bound)


    
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        if dist == 0:
            force = 0
        else:
            force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]