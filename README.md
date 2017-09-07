# DQN
Python Tensorflow Implementation for Deep Q Network
###### Currently a work in progress
This repository focus on an implementation of [DQNs](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) Mnih et al. 2015

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

### Included Files:
util.py
	- stored useful functions. Doesn't affect concept of the code.

model.py
	- dqn model

train.py
	- trains the model and logs information
