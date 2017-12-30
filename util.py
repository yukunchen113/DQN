import tensorflow as tf 
import numpy as np 
import shutil, os, gym, math, random, h5py
from PIL import Image
def variable(name,shape,initializer,trainable=True, dtype=tf.float32):
	#general variable for use of other functions, (for convienience)
	with tf.device('/cpu:0'):#store all variable onto the cpu
		out = tf.get_variable(name, shape,dtype,initializer,trainable=trainable)
			#out gets the variable if it exists or initializes it with
			#initializer if doesnt exist
	return out
def lrelu(input, const=0.2):
	const = tf.cast(const, input.dtype)
	return tf.maximum(input, tf.multiply(input,const))

def batch_normalization(input,istrain,scope='batch_normalization',decay=0.999):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		beta = variable('biases', [n_in], tf.constant_initalizer(0.0))
		gamma = variable('weights',[n_in],tf.constant_initalizer(1.0))
		mean = variable('mean',[n_in],tf.constant_initalizer(1.0),False)
		var = variable('variance',[n_in],tf.constant_initalizer(0.0),False)
		mean_val, var_val = tf.nn.moments(input,[0,1,2])
		assign_mean = tf.assign(mean, mean_val)
		assign_var = tf.assign(var,var_val)
		variance_epsilon = 1e-8
		ema = tf.train.ExponentialMovingAverage(decay=decay)
		def train_ema():
			with tf.control_dependencies([assign_mean,assign_var]):
				averages_op = ema.apply([mean,var])
			with tf.control_dependencies([averages_op]):
				return tf.identity(mean), tf.identity(var)
		istrain = tf.cast(istrain, tf.bool)
		cur_mean, cur_var = tf.cond(istrain, train_ema, 
			lambda: (ema.average_name(mean), ema.average_name(var)))
		return tf.nn.batch_normalization(input,cur_mean,cur_var,beta,gamma,
			variance_epsilon)

def conv2d(input, n_out, k,s, scope='convolution',padding = 'SAME'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		stddev = math.sqrt(2.0/(n_in*k*k))
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		kernel = variable('weights',[k,k,n_in,n_out],initializer)
		strides=[1,s,s,1]
		out = tf.nn.conv2d(input,kernel,strides,padding)
		return out 
		

def fully_connected(input, n_out, scope='fully_connected', reshape = True):
	with tf.variable_scope(scope):
		if reshape:
			n_in = reduce(lambda x,y: x*y,input.get_shape().as_list()[1:])
			out = tf.reshape(input, [-1, n_in])
		else:
			out = input
			n_in = input.get_shape()[-1].value
		stddev = math.sqrt(2.0/float(n_in))
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		weight = variable('weights', [n_in,n_out],initializer)
		bias = variable('biases',[n_out],tf.constant_initializer(0.01))
		out = tf.matmul(out, weight)
		out = tf.add(out, bias)
		return tf.nn.relu(out)

def timer(total_seconds):
	#returns a readable string format of time
	ts = total_seconds%60
	tm = (total_seconds/60)%60
	th = (total_seconds/3600)%24
	td = (total_seconds/864000)%7
	tw = total_seconds/604800
	time_list = [tw,td,th,tm,ts]
	unit_list = ['w','d','h','m','s']
	string = ''
	for i in range(len(time_list)):
		if int(time_list[i]) or unit_list[i] == 's':
			string += ' %d'%time_list[i] + unit_list[i]
	return string