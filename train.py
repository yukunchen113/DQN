import tensorflow as tf 
import numpy as np 
import shutil
import os
import gym
import math
import random
import h5py
from PIL import Image
import time
import util as ut
import model as md 
def train(params):
	global_step = tf.contrib.framework.get_or_create_global_step()
	#initialize model, environment and experience---------
	env = md.Environment(params['frame_skip'],params['game_name'])
	mod = md.DQN(params)
	args=[
		params['load_prev'],
		params['input_size'],
		params['frame_stack'],
		params['max_epi'],
		params['replay_start'], 
		params['exp_file']]
	exp = md.Experience(*args)#get all arguments from args list
	#-----------------------------------------------------

	#-----Part 1---------
	frame_stack_ph = tf.placeholder(tf.uint8,[params['frame_stack']]+params['orig_inp'])
		#frame stack placeholder
	preprocess = mod.preprocess(frame_stack_ph)
		#preprocessed input
	israndom_ph = tf.placeholder(tf.bool)
		#placeholder for getting random action
	action = mod.get_action([preprocess],israndom=israndom_ph)
		#keep in mind that action is of size [1]
	#here, should run action and store experience into Experience data

	#----Part 2----------
	#get batch of state,action,reward,new state,done
	state_shape = [params['batch_size']]+params['input_size']+[params['frame_stack']]
		#shape that state (and new state) is in
	state_ph = tf.placeholder(tf.uint8,shape=state_shape)#state placeholder
	action_ph = tf.placeholder(tf.int64,shape=[params['batch_size']])
	reward_ph = tf.placeholder(tf.float32,shape=[params['batch_size']])
	new_state_ph = tf.placeholder(tf.uint8,shape=state_shape)
	done_ph = tf.placeholder(tf.bool,shape=[params['batch_size']])
	batch_ph = [state_ph,action_ph,reward_ph,new_state_ph,done_ph]
		#batch_ph is not a placeholder itself, but a collection of placeholders
	train_opt = mod.train(global_step,batch_ph)	
	assign_list = mod.switch_params()

	#------training session-----------
	if params['load_prev']:
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(params['checkpoint_dir'])
	with tf.train.MonitoredTrainingSession(checkpoint_dir=params['checkpoint_dir']) as sess:
		if params['load_prev']:
			saver.restore(sess,ckpt.model_checkpoint_path)
		#document steps
		eps_step = 0#number of episodes that passed
		time_step = 0#steps after experience replay has started
		total_start_time = time.time()
		total_step = 0#steps in total
		while eps_step <= params['step_cap']:
			mod.init_frame_stack()
				#initialize frame stack
			x1 = env.reset()#start the environment and get initial observation
			eps_run_time = time.time()#start the runtime of an episode
			step_in_ep=0#steps passed in current episode
			mod.add_frame(x1)#add initial observation into the stack
			total_r = 0
			while True:
				#part 1---------
				experience_dict={}
				israndom_val=random.random()<=mod.rand_act_prob[0]
					#get a random bool
				experience_dict['state'],[experience_dict['action']]=sess.run([preprocess,action],feed_dict={frame_stack_ph:mod.get_stack(), israndom_ph:israndom_val}) 
					#get state and action values
				if mod.rand_act_prob>params['rand_action'][1]:
					mod.rand_act_prob[0]-=mod.rand_act_prob[1]
				new_unprocessed_state_val,experience_dict['reward'],experience_dict['done']=env.run(experience_dict['action'])
				mod.add_frame(new_unprocessed_state_val)
				experience_dict['new_state'] = sess.run(preprocess,feed_dict={frame_stack_ph:mod.get_stack()})
				exp.add_exp(experience_dict)
					#add experience
				#part 2---------
				batch_val = exp.get_batch(params['batch_size'])
				if not batch_val is None:
					sess.run([train_opt],feed_dict={batch_ph[i]:batch_val[i] for i in range(len(batch_ph))})
					if not time_step%params['target_update']:
						sess.run(assign_list)
					time_step+=1
				total_step+=1
				step_in_ep+=1
				total_r+=experience_dict['reward']
				if experience_dict['done']:
					cur_eps_run_time = ut.timer(time.time()-eps_run_time)
					total_run_time = ut.timer(time.time()-total_start_time)
					string="episodes ran: %d,steps ran in episode: %d, Total steps taken: %d,reward: %.4f,episode run time:%s,total run time:%s"
					print string%(eps_step,step_in_ep,total_step,total_r,cur_eps_run_time,total_run_time)
					break
			eps_step+=1


def main(argv=None):
	params = {
		'batch_size':128, #batch_size
		'target_update':10000, #number of iterations until target q update (C)
		'game_name':'Breakout-v0',# name of game environment
		'input_size':[84,84], #resize images to this
		'orig_inp':[210,160,3],# size of the original observation
		'conv_layers':[[32,8,4],[64,4,2],[64,3,1]], #convolutional layer parameters [depth,k,s]
		'FC_layer':512,#fully connected layer depth
		'n_output':4,#number of possible actions to take
		'step_cap':10000000000,#end training after this many steps
		'rand_action':[1.0,0.1,1000000],#epsilon, probability of taking random action, linearly decreasing from a to b over c steps with [a,b,c]
		'frame_skip':4,#number of frames to skip (k)
		'frame_stack':4,#number of past frames to stack in sequence
		'max_epi':1000000,#number of experience to keep
		'disc_fact':0.99,#discount factor for target action value function
		'replay_start':50000,#number of framess until replay starts
		'opt_params':[0.00025,0.95,0.95,0.01],#learning rate, decay,momentum, epsilon
		'main_dir':'/media/yukun/Barracuda Hard Drive 2TB/Data/DQN',#main directory for saving
		'load_prev':True,#load previous model
		'exp_file':'/media/yukun/Barracuda Hard Drive 2TB/Data/DQN/ModelExperience/experience.hdf5',
		'exp_dir':'/media/yukun/Barracuda Hard Drive 2TB/Data/DQN/ModelExperience/',
		'use_ddqn':True
	}
	params['checkpoint_dir'] = os.path.join(params['main_dir'],'ModelCheckpoint')
	if not tf.gfile.Exists(params['checkpoint_dir']):
		time.sleep(0.1)
		params['load_prev'] = False
	else:
		if not params['load_prev']:
			tf.gfile.DeleteRecursively(params['checkpoint_dir'])
			time.sleep(0.1)
	if not params['load_prev']:
		os.mkdir(params['checkpoint_dir'])
		time.sleep(0.1)
	if not tf.gfile.Exists(params['exp_dir']):
		os.mkdir(params['exp_dir'])

	#------------------------
	train(params)

if __name__ == '__main__':
	tf.app.run()
