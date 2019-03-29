[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/47237461-d2a90b00-d3e7-11e8-96a0-f0c9a0b7ad1d.png "Algorithm"
[image2]: https://raw.githubusercontent.com/lutaodai/DRL-Continuous-Control/master/score.png "Plot of Rewards 1"
[image3]: https://raw.githubusercontent.com/lutaodai/DRL-Continuous-Control/master/score_long.png "Plot of Rewards 2"

# Report - Continuous Control
This report is organized based on [Akhiad Bercovich](https://github.com/akhiadber/DeepRL_Continuous_Control/blob/master/REPORT.md)'s report.


### Implementation Details

Apart from the `README.md` file this repository consists of the following files:

1. `model.py`: Actor and Critc Network classes;
1. `ddpg_agent.py`: Agent, ReplayBuffer and OUNoise classes; The Agent class makes use of the Actor and Critic classes from `model.py`, the ReplayBuffer class and the OUNoise class;
1. `run.py`: Script which will train the agent. Can be run directly from the terminal;
1. `checkpoint_actor.pth`: Contains the weights of successful Actor Networks;
1. `checkpoint_critic.pth`: Contains the weights of successful Critic Networks.

To train the model, simply run the following command
```bash
python run.py
```

### Learning Algorithm

The agent is trained using the DDPG algorithm.

References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Algorithm details: 

![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - Q-Learning is not straighforwardly applied to continuous tasks due to the argmax operation over infinite actions in the continuous domain. DDPG can be viewed as an extension of Q-learning to continuous tasks.

    - DDPG was introduced as an actor-critic algorithm, although the roles of the actor and critic here are a bit different then the classic actor-critic algorithms. Here, the actor implements a current policy to deterministically map states to a specific "best" action. The critic implemets the Q function, and is trained using the same paradigm as in Q-learning, with the next action in the Bellman equation given from the actor's output. The actor is trained by the gradient from maximizing the estimated Q-value from the critic, when the actor's best predicted action is used as input to the critic.
    
    - As in Deep Q-learning, DDPG also implements a replay buffer to gather experiences from the agent (or the multiple parallel agents in the 2nd version of the stated environment). 
    
    - In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions. I also needed to decay this noise using an epsilon hyperparameter to achieve best results.
    
    - Another fine detail is the use of soft updates (parameterized by tau below) to the target networks instead of hard updates as in the original DQN paper. 
    
6. Hyperparameters:

Parameter | Value
--- | ---
replay buffer size | int(1e5)
minibatch size | 128
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-4
learning rate critic | 1e-4
L2 weight decay | 0
UPDATE_EVERY | 20
NUM_UPDATES | 10

6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 128 units each, batch normalization and Relu activation function, with Tanh activation at the last layer.
    - Input and output layers sizes are determined by the state and action space.
    - Training time until solving the environment takes around 38 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the environment is solved after 105 episodes.

Screen output
```
Episode 100     Average Score: 28.36
Episode 105     Average Score: 30.14
Environment solved in 105 episodes!     Average Score: 30.14
```

Below shows the plot of rewards up to when the environment is solved.
![Plot of Rewards][image2]

Below shows the plot of rewards for 3000 episodes. There are two times when rewards plumped, but it was recovered and get stablized.
![Plot of Rewards for Training 3000 episodes][image3]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameter, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.

2. Solving the more challenging [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment using edited versions of these same algorithms. 

I'll at least do PPO and attempts to solve the Crawler environment after submission of this project (due to Udacity project submission rules).
