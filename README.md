# PCC - A pytorch implementation
## A branch from https://github.com/orilinial/PCC
This project is a pytorch implementation of the PCC-RL project by Jay et al., given <a href="https://github.com/PCCproject/PCC-RL">here</a>. The paper is given <a href=
"https://arxiv.org/abs/1810.03259"> here</a>.
</br>
In this project we implemented the PCC-RL agent, to learn how to adapt the client's sending rate instead of the regular congestion control mechanisms provided with linux. The agent learns using the network simulation provided by the original repository.</br>

### Implementation
We implemented the RL agent using a standard Actor-Critic method, with pytorch. </br>
The network is simple: one hidden layer with 128 units for the actor and critic. The actor learns the mean of the policy distribution as a Normal distribution, with a predifined variance. The given states are history of 10 monitor-intervals statictics, with specific features (as described in the original paper).

### Results
We ran a task similar to the one provided in PCC-RL paper, and received similar results:</br>
![Throughput of RL vs TCP](results/graphs/thpt_send_rl_tcp_10.png)

### Requirments
#### Ubuntu
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install gym

pip install numpy

pip install matplotlib
