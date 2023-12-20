A forked version of OpenAI's multiagent particle environment that includes the simple_survey_region scenario for training agents in a simple surveillance task. 

<p align="center">
  <img src="https://github.com/calwilmott/mpe-surveillance/assets/37785396/6dfdeba5-d8f9-4af6-88bb-b84758107ec6" alt="Screenshot from 2023-12-19 19-25-46">
</p>

The environment contains:

- **N agents**: Each agent can move continuously through a “grid world”, and can rotate their orientation.
- **Red “Obstacle” grid squares**: These are areas which agents cannot pass through.
- **Agents’ line of sight**: Represented by red lines, which do not pass through the red obstacles.
- **Rewards**: Agents are rewarded in each time step for passing through or viewing the black squares. These squares are then set to white before gradually fading back to black. Darker squares offer higher rewards than lighter squares.

## Paper citation

If you used this environment for your experiments or found it helpful, consider citing the following papers:

Environments in this repo:
<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>


Original particle world environment:
<pre>
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
</pre>
