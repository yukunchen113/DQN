# DQN
Python Tensorflow Implementation for Deep Q Network

This repository focus on an implementation of [DQNs](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) Mnih et al. 2015
Also DDQN can be implemented (https://arxiv.org/abs/1509.06461)
### How it works

There are 2 parts to a dqn. 
- The first part plays the game
  - observes current state
  - performs action
  - observes reward, and new state
  - stores this experience in a set of N recent experiences
- second part learns the correct procedures
	- retrieve a batch size amount of experiences and learn the action value function according to bellman equation

- action value function is made with CNN.

-DDQN reduces the overoptimism from DQN, where both evaluation of the action and it's value is made.

- uses openai gym as environment

### Included Files:
util.py
	- stored useful functions. Doesn't affect concept of the code.

model.py
	- dqn/ddqn model

train.py
	- trains the model and logs information

Openai GYM:--------------
@misc{1606.01540,
        Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
        Title = {OpenAI Gym},
        Year = {2016},
        Eprint = {arXiv:1606.01540},
}
