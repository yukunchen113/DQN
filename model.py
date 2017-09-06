import tensorflow as tf 
import numpy as np 
import os
import gym
import util as ut
import random
import gym
import h5py
class Experience():
	#can:
	#	- get the experience (phi(t), a(t), r(t), phi(t+1),terminate)
	#		- terminate is a bool, which tells if the episode terminates at next step
	#	- return a minibatch of random past experiences
	def __init__(self, params):
		self.params = params
		self.state = np.empty([params['replay_cap']]+params['input_size']+[params['frame_stack']], 
			np.uint8)
		self.action = np.empty([params['replay_cap']], np.float32)
		self.reward = np.empty([params['replay_cap']], np.float32)
		self.done = np.empty([params['replay_cap']], np.bool)
		
		#self.next_state=np.zeros([params['replay_cap']]+params['input_size']+[params['frame_stack']], 
		#	np.uint8)

		self.step = 0#time step
		
	def add_exp(self, experience):
		#add in a circular queue 
		#experience should have state, action, reward, next_state values
		index = self.step%self.params['replay_cap']
		self.state[index],self.action[index], self.reward[index],self.state[(index+1)%self.params['replay_cap']],self.done[index]=experience
		self.step+=1

	def get_batch(self):
		if self.step <= self.params['replay_start']:
			return None
		max_index = min(self.step, self.params['replay_cap']-1)
		batch = [[],[],[],[],[]]
		for _ in range(self.params['batch_size']):
			index = random.randint(0,max_index)
			batch[0].append(self.state[index])
			batch[1].append(self.action[index])
			batch[2].append(self.reward[index])
			batch[3].append(self.state[(index+1)%self.params['replay_cap']])
			batch[4].append(self.done[index])
		return batch

class Environment():
	#OpenAI Gym environment, runs action until action is changed
	#has functions to change action and return k amount of frames after running
	#	that action
	def __init__(self,params):
		self.params = params
		self.k = params['frame_skip']
		self.env = gym.make(params['game_name'])

	def reset(self):
		observation = self.env.reset()
		return observation

	def run(self,action):#runs for k amount
		reward = 0.0
		for i in range(self.k):
			self.env.render()
			observation, r, done, _ = self.env.step(action)
			reward+=r
			if i == (self.k-2):
				last_observation = observation
		total_observation = np.maximum(last_observation,observation).tolist()
		return observation, reward, done


class dqn():
	#part 1: get and store data to environment-----------------
	def __init__(self, params):
		self.params = params
		self.environment = Environment(self.params)
		randact = params['rand_action']
		self.rand_act_prob = [randact[0], (randact[0]-randact[1])/float(randact[2])]

	def init_seq(self):
		self.sequence = np.zeros([self.params['frame_stack']]+self.params['orig_inp'],np.uint8)
	def add_seq(self, img):
		self.sequence[:-1]=self.sequence[1:]
		self.sequence[-1] = img

	def preprocess(self,sequence):
		#changes images to grayscale, and stacks last m frames in a sequence
		#	- if sequence contains less than m frames, use black frames instead
		out = sequence
		self.dred = sequence
		self.prep = sequence
		out = tf.image.resize_images(out,self.params['input_size'])
		out = tf.image.rgb_to_grayscale(out)
		out = tf.squeeze(out)
		out = tf.transpose(out)
		return out



	def Action_Value_Function(self, input, reuse=True, scope='Action_Value'):
	#two of these should be created, action value and target action value
		#computs action value of preprocessed image
		#Has two modes 
		# 	- takes in a batch of batch_size,
		# 	- takes in an image and makes it into a batch of 1
		with tf.variable_scope(scope,reuse=reuse):
			out = tf.cast(input,tf.float32)
			conv_params = self.params['conv_layers']
			for i in range(len(conv_params)):
				clp = conv_params[i]
				out = ut.conv2d(out,clp[0],clp[1],clp[2],'conv2d%d'%(i+1))
				out = tf.nn.relu(out)
			out = ut.fully_connected(out, self.params['FC_layer'],'fully_con1')
			out = tf.nn.relu(out)
			out = ut.fully_connected(out,self.params['n_output'],
				'fully_con2',False)
			#print out.get_shape().as_list()
			return out

	def get_action(self,preprocess,israndom):
		#chooses random action or action value function
		avf = self.Action_Value_Function(preprocess, False)
		avf = tf.squeeze(avf,0)
		out1 = tf.argmax(avf)
		cond = tf.cast(israndom,tf.bool)
		out2 = tf.random_uniform([],0,self.params['n_output'])
		out2 = tf.cast(out2,tf.int64)
		out = tf.cond(cond, lambda:out2, lambda:out1)
		self.test = out
		return out
		
	#part 2: minibatch and train---------------------
	def get_reward(self, batch):
		#gets reward with target action value or just reward if experience terminates at next step
		tavf = self.Action_Value_Function(batch[3],False,'Target_Action_Value')
		out = tf.multiply(self.params['disc_fact'],tavf)
		isdone = tf.reshape(tf.cast(batch[-1],tf.float32),[128,1])
		out = tf.reduce_max(out,1)
		out = out*isdone + batch[2]
		return out

	def loss(self, batch):
		#finds the loss between the outputs returned by the agent
		labels = self.get_reward(batch)
		logits = tf.reduce_max(self.Action_Value_Function(batch[0]),1)
		out = tf.subtract(labels, logits)
		print logits.get_shape().as_list()
		out = tf.clip_by_value(out,-1,1)
		out = tf.square(out)
		loss = tf.reduce_mean(out)
		return loss
	def switch_params(self):
		#assigns parameters from model avf to tavf
		avf_var = [var for var in tf.trainable_variables() if var.name.startswith('Act')]
		tavf_var = [var for var in tf.trainable_variables() if var.name.startswith('Tar')]
		assign_list = [tf.assign(avf_var[i] ,tavf_var[i]) for i in range(len(avf_var))]
		return assign_list
	
	def train(self, global_step,batch):
		#trains to minimize the loss, and switches parameters after C iterations
		loss = self.loss(batch)
		lr,decay,mom,ep=self.params['opt_params']
		optimizer = tf.train.RMSPropOptimizer(lr,decay,mom,ep).minimize(loss,global_step=global_step)
		with tf.control_dependencies([optimizer]):
			train_opt = tf.no_op(name='train')
			return train_opt



