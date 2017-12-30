#Deep q network for atari games
#I did not get the maximum between frames like the original DQN
#	- felt that it was unnecessary as sprites are not on every other frame
import tensorflow as tf 
import numpy as np 
import shutil
import os
import gym
import math
import random
import h5py
import util as ut
from PIL import Image
class Experience():
	'''
	Experience replay object
		- can store experience to experience replay
		- can return a batch of experiences
	'''
	def __init__(self,load_prev,input_size,frame_stack,max_epi,replay_start,exp_file):
		'''
		initializes experience data,
			- if load_prev and file exists, get past experience 
			- if not load_prev or not file exists, create new experience
			- states should be preprocessed and stacked
		'''
		self.exp_file = exp_file
		self.max_epi =max_epi
		self.replay_start = replay_start
		self.file = h5py.File(self.exp_file,'a')
			#create or open the h5py file
		past_exp = [int(x.split('experience')[-1]) for x in self.file]
		if not load_prev or past_exp == []:
			if not past_exp == []:
				index = max(past_exp)+1
			else:
				index = 0
		else:
			index = max(past_exp)
		self.grname = "experience%d"%index# group name
		self.group = self.file.require_group(self.grname)
		#experience ---------------
		self.state = self.group.require_dataset("state",tuple([max_epi]+input_size+[frame_stack]),dtype=np.uint8)
			#current state
		self.action = self.group.require_dataset("action",(max_epi,),dtype=np.int64)
			#action performed due to state
		self.reward = self.group.require_dataset("reward",(max_epi,),dtype=np.float32)
			#reward achieved after action
		self.next_state = self.group.require_dataset("next_state",tuple([max_epi]+input_size+[frame_stack]),dtype=np.uint8)
			#then resulting state after action
		self.done = self.group.require_dataset("done",(max_epi,),dtype=np.bool)
			#if episode is finished after
		#---------------------------------------
		self.step = self.group.require_dataset('step',(1,),dtype=np.uint64)
		self.step = 0
	def add_exp(self,experience):
		index = int(self.step%self.max_epi)
		self.state[index]=experience['state']
		self.action[index]=experience['action']
		self.reward[index]=experience['reward']
		self.next_state[index]=experience['new_state']
		self.done[index]=experience['done']
		self.step=self.step+1
	def get_batch(self,batch_size):
		
		ran = min(self.step-1,self.max_epi)
		if ran <self.replay_start:
			return None
		random_pick = np.random.randint(ran,size=(batch_size,))
		random_pick = random_pick.tolist()
		out = [[],[],[],[],[]]
		for index in random_pick:
			out[0].append(self.state[index])
			out[1].append(self.action[index])
			out[2].append(self.reward[index])
			out[3].append(self.next_state[index])
			out[4].append(self.done[index])
		return out


class Environment():
	'''
	Environment object
		- manages an openai gym environment (Atari)
	'''
	def __init__(self,frame_skip,game_name):
		'''
		Creates gym environment
		Args:
			frame_skip: number of frames to skip. for more human like motions
			game_name: name of the game to play
		'''
		self.k = frame_skip
		self.env = gym.make(game_name)
	def reset(self):
		#resets an enviroment (starts new one)
		#happens when game has finished or starting new game
		#returns inital state
		observation = self.env.reset()
		return observation
	def run(self, action):
		#runs an action in the environment
		'''
		Args:
			action: action to perform in environment
		Returns:
			observation: next state s
			reward: reward achieved
			done: bool, if episode has ended
		'''
		reward = 0.0 #pool the reward across k frame skips
		for i in range(self.k):
			self.env.render()
			observation,r,done,_ = self.env.step(action)
			reward+=r
		return observation, reward, done

class DQN():
	#Deep Q network model
	def __init__(self, params):
		'''
		initialize DQN model
		Args:
			params: parameters required for dqn model
		Return:
			None
		'''
		self.params = params
		randact = params['rand_action']#gets randon action settings
		self.rand_act_prob = [randact[0],(randact[0]-randact[1])/float(randact[2])]
			#creates a list of: [current rand act prob, change in prob per step]

	#----functions for stacking a group of k frames together-----
	def init_frame_stack(self):
		#initializes the stack of frames (start with a stack of black frames)
		self.stack = np.zeros([self.params['frame_stack']]+self.params['orig_inp'], np.uint8)
	def add_frame(self, img):
		#adds one frame to the stack
		#shifts indicies down and adds new image
		#change the stack of frames as a queue
		self.stack[:-1] = self.stack[1:]
		self.stack[-1] = img
	def get_stack(self):
		#returns stack of size [frame stack, orig input]
		return self.stack
	#---------

	#----preprocess a stack of frames-----------
	def preprocess(self,stack):
		'''
		preprocesses images, canges to grayscale, and stacks last frames as a sequence
		Args:
			stack: stack of frames to be preprocessed
		Returns:
			out: preprocessed input
		'''
		out = stack
		out = tf.image.resize_images(out, self.params['input_size'])
		out = tf.image.rgb_to_grayscale(out)
		out = tf.squeeze(out)
		out = tf.transpose(out)
		return out
	
	#-------------------------------------------

	#------action value function----------------
	def action_value_function(self,input,reuse=True,scope='Action_Value'):
		#should be created with invarianve to batch size
		#input should be a placeholder
		#outputs are actions shape [batch_size, num_actions]
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
			return out 

	def get_action(self, preprocess,reuse = False,israndom=False):
		'''
		get action for a batch of preprocessed images. 
		Used for 
			- online experience generation 
			- getting action component in batch portion
			#works with batch size of 1
			- could also give random action
		keep in mind that whether it be for DQN or DDQN, only the action
		value function is used for getting the correct action

		Args:
			preprocess: preprocessed image(s) in a batch, even a single
				preprocess should have a batch of 1
			israndom: if a random action should be returned
		Returns:
			out: batch of actions

		###to run this, israndom should be a placeholder when doing online

		'''
		avf = self.action_value_function(preprocess, reuse=reuse)
		out = tf.argmax(avf,axis=1)
		rand_cond = tf.cast(israndom,tf.bool)
		batch_size = out.get_shape()
		rand_out = tf.random_uniform(batch_size,0,self.params['n_output'])
		rand_out = tf.cast(rand_out,tf.int64)
		out = tf.cond(rand_cond, lambda:rand_out, lambda:out)
		return out
	#---------------


	#------Part 2: minibatch training---------------
	def compute_reward(self, batch):
		#batch should be [state, action, reward, next_state, done]
		#these should be placeholders
		state,action,reward,next_state,done = batch
		tavf = self.action_value_function(next_state,False,'Target_Action_Value')
			#create target action value function
			#tavf is all the possible action values
		if self.params["use_ddqn"]:#get yt+1 portion of target value
			#for DDQN
			actions = self.get_action(next_state,True,False)
			out = tf.diag_part(tf.gather(tavf,actions,axis=1))
				#get the value from TAVF, indexed by actions from AVF
				#becomes of shape [batch_size]
		else:
			#for DQN
			out = tf.reduce_max(tavf,axis=1)

		out = tf.multiply(out, tf.cast(self.params['disc_fact'],tf.float32))
		just_r_out = tf.cast(reward,tf.float32)
		just_r_out = tf.clip_by_value(just_r_out,-1,1)
			#reward clipping
		out = tf.add(out,just_r_out)
		isdone = tf.cast(done,tf.bool)
		out = tf.where(isdone, just_r_out, out)
		return out

	def compute_loss(self, batch):
		#batch should be [state, action, reward, next_state, done]
		state,action,reward,next_state,done = batch
		labels = self.compute_reward(batch)
		avf = self.action_value_function(state,True)
		logits = tf.diag_part(tf.gather(avf,action,axis=1))
			#get the value from AVF, indexed by previous actions
			#becomes of shape [batch_size]
		out = tf.subtract(labels, logits)
		out = tf.square(out)
		out = tf.reduce_mean(out)
		out = tf.clip_by_value(out,-1,1)
			#L2 Loss with clip
		return out
	def switch_params(self):
		avf_var = [var for var in tf.trainable_variables() if var.name.startswith('Action')]
		tavf_var = [var for var in tf.trainable_variables() if var.name.startswith('Target')]
		assign_list = [tf.assign(tavf_var[i], avf_var[i]) for i in range(len(avf_var))]
		return assign_list

	def train(self,global_step,batch):
		loss = self.compute_loss(batch)
		lr,decay,mom,ep = self.params['opt_params']
		optimizer = tf.train.RMSPropOptimizer(lr,decay,mom,ep).minimize(loss, global_step=global_step)
		with tf.control_dependencies([optimizer]):
			train_opt = tf.no_op(name='train')
			return train_opt


def test_dqn1(batch,reuse = False):

	#tests __init__, frame stack methods, preprocess, avf and get_action
	params = {
	'input_size':[84,84], #resize images to this
	'orig_inp':[210,160,3],# size of the original observation
	'conv_layers':[[32,8,4],[64,4,2],[64,3,1]],
	'n_output':4,
	'FC_layer':512,#fully connected layer depth
	'rand_action':[1.0,0.1,1000000],
	'frame_stack':4}

	#testing frame_stack
	mod = DQN(params)
	mod.init_frame_stack()
	#testing preprocess
	prep = mod.preprocess(mod.get_stack())
	if not prep.get_shape().as_list() == params['input_size']+[params['frame_stack']]:
		return False
	#testing get_action
	if not batch:
		mod = DQN(params)
		item = mod.get_action([prep],reuse)
		if not item.get_shape().as_list() == [1]:
			return False
	else:
		mod = DQN(params)
		prep = np.zeros((32,84,84,4),dtype=np.uint8)
		prep = tf.cast(prep, tf.uint8)
		item = mod.get_action(prep,reuse)
		if not item.get_shape().as_list() == [32]:
			return False
	return True

def test_environment():
	env = Environment(4,'Breakout-v0')
	a = 0
	o = env.reset()
	no,r,d = env.run(a)
	return True

def test_experience(old_file,load_prev):
	#--testing __init__
	exp_file = 'test.hdf5'
	if old_file:
		try:
			os.rmdir(exp_file)
		except OSError:
			pass
	max_epi = 5
	replay_start = 2
	input_size = [84,84]
	frame_stack = 4
	exp = Experience(load_prev,input_size,frame_stack,max_epi,replay_start,exp_file)

	#---testing add_exp 
	state = np.ones((84,84,4))
	action = 0
	reward = 3.456
	done = False
	new_state = np.ones((84,84,4))
	action_test = [0 for i in range(max_epi)]
	for i in range(8):
		experience = {'state':state,'action':action,
		'reward':reward,'done':done,'new_state':new_state}
		exp.add_exp(experience)
		action+=1
		action_test[i%max_epi] = i
	if not list(exp.action[()]) == action_test:
		return False
		#return False

	#testing get_batch
	exp.get_batch(4)

	return True

def main():
	#used for testing

	#testing experience-----------------------------------
	t1 = test_experience(False, False)#creates a file
	t2 = test_experience(True, False)#create new group
	t3 = test_experience(True, True)#use old file, use old experience
	if t1 and t2 and t3:
		print "experience is working"
	else:
		print "experience failure"
	if test_environment():
		print "environment is working"
	else:
		print "environment failure"
	t1 = test_dqn1(False)
	t2 = test_dqn1(True,True)
	tf.reset_default_graph()
	if t1 and t2:
		print "dqn part 1 is working"
	else:
		print "dqn part 1 failure"


if __name__ == '__main__':
	main()