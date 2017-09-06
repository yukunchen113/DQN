import tensorflow as tf 
import numpy as np 
import math
import os
import random
from PIL import Image
import model as md 
import util as ut
import time

def train(params):
	global_step = tf.contrib.framework.get_or_create_global_step()
	environment = md.Environment(params)
	
	model = md.dqn(params)
	
	seq = tf.placeholder(tf.uint8,shape=[params['frame_stack']]+params['orig_inp'])
	preprocess = model.preprocess(seq)
	israndom_ph = tf.placeholder(tf.bool)
	action = model.get_action([preprocess],israndom_ph)#gets phi(t) and a(t)

	exp = md.Experience(params)
	
	#-----Part2-----------
	img_ph = tf.placeholder(tf.uint8, 
		shape=[params['batch_size']]+params['input_size']+[params['frame_stack']])
	act_ph = tf.placeholder(tf.float32, shape=[params['batch_size']])
	rew_ph = tf.placeholder(tf.float32, shape=[params['batch_size']])
	new_img_ph = tf.placeholder(tf.uint8, 
		shape=[params['batch_size']]+params['input_size']+[params['frame_stack']])
	done_ph = tf.placeholder(tf.bool,shape = [params['batch_size']])
	ph_list = [img_ph,act_ph,rew_ph,new_img_ph,done_ph]
	train_opt = model.train(global_step,ph_list)
	assign_list = model.switch_params()
	if params['load_prev']:
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(params['checkpoint_dir'])
	with tf.train.MonitoredTrainingSession(checkpoint_dir=params['checkpoint_dir']) as sess:
		if params['load_prev']:
			saver.restore(sess, ckpt.model_checkpoint_path)
		eps_step = 0
		time_step = 0
		total_start_time = time.time()
		total_step = 0
		while eps_step <= params['last_step']:
			model.init_seq()
			x1 = environment.reset()
			eps_run_time = time.time()
			stepinep = 0
			model.add_seq(x1)
			total_r = 0
			while True:
				#---Part1--------------
				israndom_val = random.random()<=model.rand_act_prob[0]
				phi, a = sess.run([preprocess,action],feed_dict = {seq:model.sequence, israndom_ph:israndom_val})
				if model.rand_act_prob[0] > params['rand_action'][1]:
					model.rand_act_prob[0] -= model.rand_act_prob[1]
				#assert type(a) == np.int64
				#print sess.run(model.test,feed_dict = {seq:model.sequence, israndom_ph:israndom_val})
				n_x, r, done = environment.run(a)
				model.add_seq(n_x)
				n_phi = sess.run(preprocess,feed_dict = {seq:model.sequence})
				exp.add_exp([phi,a,r,n_phi,done])
				#------------------
				batch_val = exp.get_batch()
				if not batch_val == None:
					sess.run([train_opt],feed_dict={ph_list[i]:batch_val[i] for i in range(len(ph_list))})
					if not time_step%params['target_update']:
						sess.run(assign_list)
					time_step+=1
				total_step +=1
				stepinep +=1
				total_r+=r
				#run values
				if done:
					cur_eps_run_time = ut.timer(time.time() - eps_run_time)
					total_run_time = ut.timer(time.time() - total_start_time)
					print 'episode steps: %d, steps in Episode: %d, Total Steps taken: %d, reward: %.4f, episode run time:%s, Total run time:%s'%(eps_step,stepinep, total_step, total_r,cur_eps_run_time,total_run_time)
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
		'replay_cap':100000,#amount of experience to store in replay
		'rand_action':[1.0,0.1,1000000],#epsilon, probability of taking random action, linearly decreasing from a to b over c steps with [a,b,c]
		'frame_skip':4,#number of frames to skip (k)
		'frame_stack':4,#number of past frames to stack in sequence
		'last_step':10000000,#number of episodes to run
		'disc_fact':0.99,#discount factor for target action value function
		'replay_start':5000,#number of framess until replay starts
		'opt_params':[0.00025,0.95,0.95,0.01],#learning rate, decay,momentum, epsilon
		'main_dir':'/media/yukun/Barracuda Hard Drive 2TB/Data/DQN',#main directory for saving
		'load_prev':True#load previous model
	}
	#----------parameter setup
	params['checkpoint_dir'] = os.path.join(params['main_dir'],'ModelCheckpoint')
	params['experience_dir'] = os.path.join(params['main_dir'],'ModelExperience')
	if not tf.gfile.Exists(params['checkpoint_dir']):
		time.sleep(0.1)
		params['load_prev'] = False
	else:
		if not params['load_prev']:
			tf.gfile.DeleteRecursively(params['checkpoint_dir'])
			time.sleep(0.1)
			tf.gfile.DeleteRecursively(params['experience_dir'])
			time.sleep(0.1)
	if not params['load_prev']:
		os.mkdir(params['checkpoint_dir'])
		time.sleep(0.1)
		os.mkdir(params['experience_dir'])
	#---------------------
	train(params)


if __name__ == '__main__':
	tf.app.run()
