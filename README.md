# DDQN Algorithm

_Code by Mateus Karvat Camara, adapated from Mirco Theile's [uavSim](https://github.com/theilem/uavSim/tree/icar) and Callaghan Wilmott's [mpe-surveillance fork](https://github.com/calwilmott/mpe-surveillance)_.

---

## Installation

Having python 3.8+ installed, make sure all required libraries are installed by running ``pip install -r requirements.txt``.

---

## Training

Training can be executed by running the ``train.py`` file. The `train.py` file has some environment/training parameters that may be changed:

<ul>
    <li>num_agents: 1 for single agent, 3 for multi-agent;</li>
    <li>reward_delta: indicates how much the value of a cell will decay at each time step;</li>
    <li>seed: an integer may be used to force the environment to always use the same world (baseline worlds will not be used, however);</li>
    <li>reward_type: "pov" (standard) or "map";</li>
    <li>world_filename: if a baseline world is used, the relative path to the PKL file must be provided. Ex: "baseline_worlds/world_00.pkl";</li>
    <li>reward_type: "pov" (standard) or "map";</li>
    <li>deep_discretization: True or False (standard);</li>
    <li>is_render: determines if environment should be rendered while training. True or False.</li>
</ul>

Other parameters in ``train.py`` may also be changed, but have not been experimented with:

<ul>
    <li>num_obstacles: number of obstacles in the world;</li>
    <li>vision_dist: maximum distance that an agent can perceive;</li>
    <li>grid_resolution: resolution of the grid;</li>
    <li>grid_max_reward: maximum value that a grid cell might have.</li>
</ul>

The DDQN algorithm was designed considering the ``hybrid`` observation mode, therefore this parameter should not be changed.

Relevant parameters for the DDQN algorithm are also found in the ```DDQN/Agent.py``` (parameters related to the network) and ``DDQN/Trainer.py`` (parameters related to the RL algorithm) files, most of which are very self-explanatory (such as ``learning_rate``). Out of all of these parameters, the only ones that were modified in our experiments were:

<ul>
    <li>In DDQN/Agent, use_global_local: some experiments were performed by deactivating it;
    <li>In DDQN/Trainer, num_episodes: 2000 episodes was used for most experiments, but 5000 episodes was used for experiments with training being done on randomized worlds.</li>
</ul>

---

## Testing

Each model trained will generate a new ``run`` folder, which can later be tested by running ```test.py```. The program will ask for a run number, which refers to the individual ``run`` folders located in the ``runs\`` folder. 

We provide the following models, which refer to results presented in the report:

<ul>
    <li>run5: preliminary experiment with POV reward and no deep discretization;</li>
    <li>run6: preliminary experiment with MAP reward and global-local approach;</li>
    <li>run7: preliminary experiment with POV reward and deep discretization;</li>
    <li>run13: preliminary experiment with MAP reward and no global-local approach;</li>
    <li>run17: single agent experiment with scenario A;</li>
    <li>run18: single agent experiment with scenario B;</li>
    <li>run19: single agent experiment with scenario C;</li>
    <li>run20: single agent experiment with scenario D;</li>
    <li>run21: single agent experiment with scenario E;</li>
    <li>run22: multi agent experiment with scenario F;</li>
    <li>run23: multi agent experiment with scenario G;</li>
    <li>run24: multi agent experiment with scenario H;</li>
    <li>run25: multi agent experiment with scenario I;</li>
    <li>run26: multi agent experiment with scenario J;</li>
    <li>run29: single agent experiment trained on randomized scenarios;</li>
    <li>run31: multi agent experiment trained on randomized scenarios.</li>
</ul>

While running ``test.py``, after inputting the corresponding number of the run to be evaluated (e.g. 24), the program will ask if multiple runs are to be executed. The results presented in the report have all gone through 10 successful evaluation runs (runs in which the collision bug did not occur), from which the lowest and highest values were discarded and the remaining 8 were averaged. Therefore, if the 10 runs are to be executed, then the multiple runs option should be chosen. Otherwise, it can be dismissed. The results from each run will be shown in the terminal. 