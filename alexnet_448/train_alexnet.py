###################################################
###########This is just the first version########### 
###########The codes will be updated in the future###
####################################Queenie.Liu#####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import division

import argparse
import os.path
import sys
import time
import numpy as np
import datetime
import os
import re

import tensorflow as tf

import global_define as gd
#import Alexnet
import PIL
#from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
#from sklearn import datasets, metrics, cross_validation
#import sklearn as sk
#from tensorflow.models.inception.inception.slim import scopes 
#from utils import tile_raster_images
from tensorflow.contrib.slim.python.slim.nets import alexnet_448
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'val.tfrecords'
#TEST_FILE='test.tfrecords'
os.environ["CUDA_VISIBLE_DEVICES"]=gd.CUDA_VISIBLE_DEVICES
flags = tf.app.flags
FLAGS = flags.FLAGS

time_value=re.sub(r'[^0-7]','',str(datetime.datetime.now()))
flags.DEFINE_string('tfrecord_dir', 
	gd.TFRECORD_PATH, 'Directory to put the training data.')
#flags.DEFINE_string('filename', 'train.tfrecords', 'Directory to put he training data.')
flags.DEFINE_integer('batch_size',gd.BATCH_SIZE, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('learning_rate', gd.LEARNING_RATE,'balabala')
flags.DEFINE_integer('max_steps', 50000,'balabala')
flags.DEFINE_string('model_dir','Modal/model'+str(time_value)+'/','balabala')
flags.DEFINE_string('tensorevents_dir','tensorboard_event/event_wth'+str(time_value)+'/','balabala')
flags.DEFINE_string('log_dir','Log_data/'+str(gd.DESCRIPTION)+'_'+str(time_value)+'/','balabala')
flags.DEFINE_string('pic_dir','Pic/Pictures_input'+str(time_value)+'/','balabala')

if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

log_name=FLAGS.log_dir+'/'+'log.txt'
f=open(log_name,'w')
f.close()
# if not os.path.exists(FLAGS.tensorevents_dir):
#   os.makedirs(FLAGS.tensorevents_dir)

# if not os.path.exists(FLAGS.model_dir):
# 	os.makedirs(FLAGS.model_dir)



# if not os.path.exists(FLAGS.pic_dir):
# 	os.makedirs(FLAGS.pic_dir)

def read_and_decode(filename_queue):

	reader=tf.TFRecordReader()
	_,serialized_exampe=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_exampe,
		features={
		'image_raw':tf.FixedLenFeature([],tf.string),
		'height':tf.FixedLenFeature([],tf.int64),
		'width':tf.FixedLenFeature([],tf.int64),
		'depth':tf.FixedLenFeature([],tf.int64),
		'label':tf.FixedLenFeature([],tf.int64)
		})
	image=tf.decode_raw(features['image_raw'],tf.uint8)
	print("shape:")
	print(tf.shape(image))
	#tf.reshape(image,[224,224,3])
	#print("decode images after set_shape:")
	#print(str(tf.shape(image)))
	#image.set_shape([gd.IMAGE_PIXELS])
	image.set_shape([gd.IMAGE_PIXELS*3])
	print("read and decode:")
	print(image)
	image=tf.cast(image,tf.float32)*(1./255)-0.5
	label=tf.cast(features['label'],tf.int32)
	return image,label

def do_eval(sess,eval_correct,log_name,mode):
	true_count=0
	#for step in xrange(FLAGS.batch_size):
	true_count+=sess.run(eval_correct)
        if mode=="train":
                eval_batch_size=gd.BATCH_SIZE
        else:
                eval_batch_size=gd.BATCH_SIZE_VAL

	precision=float(true_count)/eval_batch_size
	# print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
 #            (FLAGS.batch_size, true_count, precision))
 	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n' %
            (eval_batch_size, true_count, precision))
	logfile=open(log_name,'a')
	logfile.write('  Num examples: %d  Num correct: %d  Precision : %0.04f\n' %    (eval_batch_size, true_count, precision))
	
	logfile.close()
	return precision

def calc_loss(logits,labels):
	batch_size=tf.size(labels)
	labels=tf.expand_dims(labels,1)
	indices=tf.expand_dims(tf.range(0,batch_size),1)

	concated=tf.concat([indices,labels],1)
	#print("sta:")
	#print(concated)
	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)
	print('onehot_labels:')
	print(onehot_labels)
	print('logits:')
	print(logits)

	flat_logits=logits
	flat_labels=onehot_labels
	eps=tf.constant(value=1e-10)
	flat_logits=flat_logits+eps

	softmax=tf.nn.softmax(flat_logits)

	#back_freq=998/1328
	#road_freq=330/1328


	#coeffs=tf.constant([(0.5/back_freq),(0.5/road_freq)])
	coeffs=tf.constant(gd.COEFFS)
	cross_entropy=-tf.reduce_sum(tf.multiply(flat_labels*tf.log(softmax+eps),coeffs),reduction_indices=[1])
	#loss=tf.reduce_mean(cross_entropy,name="xentropy_mean")
	#tf.summary.scalar('xentropy_mean',loss)
	
	#loss_L2=tf.add_n([tf.nn.l2_loss()])

	#cross_entropy=slim.losses.softmax_cross_entropy(logits,onehot_labels)
	
	#print("cross_entropy:")
	#print(cross_entropy)
	#print(tf.shape(cross_entropy))
	# error=tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=logits)
	# print("error:")
	# print(error)

	# weights=tf.constant([1.0,2.0])
	# loss = tf.contrib.losses.compute_weighted_loss(cross_entropy, weights)
	# #loss=tf.reduce_mean(cross_entropy,name='xentropy_mean')
	
	#loss=error
	return cross_entropy

def evaluation(logits_in,labels,mode):
	logits_in=tf.squeeze(logits_in)
	correct=tf.nn.in_top_k(logits_in,labels,1)
	if mode=='train':
		tf.summary.scalar('accuracy_train',tf.reduce_sum(tf.cast(correct,tf.int32))/gd.BATCH_SIZE)
	elif mode == 'val':
		tf.summary.scalar('accuracy_val',tf.reduce_sum(tf.cast(correct,tf.int32))/gd.BATCH_SIZE_VAL)
	return tf.reduce_sum(tf.cast(correct,tf.int32))


def inputs(train,batch_size,num_epochs):
	if not num_epochs:
		num_epochs=None
	if train=='train':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
	elif train=='val':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
	# else:
	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)

	with tf.name_scope('input'):
		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
		print(filename)
		image,label=read_and_decode(filename_queue)
		print("input image:")
		print(image)
		images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=6,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)
	return images, sparse_labels

def run_training():

	with tf.Graph().as_default():
		with slim.arg_scope(alexnet_448.alexnet_v2_arg_scope()):
		#with slim.arg_scope(resnet_v2.resnet_arg_scope()):
			images,labels=inputs(train='train',batch_size=gd.BATCH_SIZE,num_epochs=FLAGS.num_epochs)
			
			images_val,labels_val=inputs(train='val',batch_size=gd.BATCH_SIZE_VAL,num_epochs=FLAGS.num_epochs)
			
			images=tf.reshape(images,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			# print("images:")
			# print(images)

			#logits,description=resnet_v2.resnet_v2_101(images,4,is_training=True)
			logits,description=alexnet_448.alexnet_v2(images,num_classes=gd.NUM_CLASSES,is_training=True)
			print('logits:')
			print(logits)
			print('description:')
			print(description)
			#loss=slim.losses.softmax_cross_entropy(logits, labels)
			tf.get_variable_scope().reuse_variables()
			
			images_val=tf.reshape(images_val,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			#loss=slim.losses.softmax_cross_entropy(logits, labels)		
                	logits_val,_=alexnet_448.alexnet_v2(images_val,num_classes=gd.NUM_CLASSES,is_training=True)
			
			cross_entropy=calc_loss(logits,labels)
			tf.summary.scalar('entropy_mean',tf.reduce_mean(cross_entropy))

			print("cross_entropy:")
			print(cross_entropy)
			loss_beta=0.001

			#variable_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

			weight_variable_list=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
			#print("variable_list:")
			#print(variable_list)
			print("weight_list:")
			print(weight_variable_list)

			l2_loss=tf.add_n([tf.nn.l2_loss(v) for v in weight_variable_list])*loss_beta
			tf.summary.scalar('l2_loss',tf.reduce_mean(l2_loss))
			print("l2_loss:")
			print(l2_loss)

			print("add result:")
			print(tf.add(cross_entropy,l2_loss))

			loss_total=tf.reduce_mean(tf.add(cross_entropy,l2_loss),name="total_loss")

			tf.summary.scalar('total_loss',loss_total)

			#l2_loss=tf.add_n([ tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) ]) * loss_beta
			
			optimizer=tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

			global_step=tf.Variable(0,name='global_step',trainable=False)

			train_op=optimizer.minimize(loss_total,global_step=global_step)			
			
			eval_correct=evaluation(logits,labels,'train')

			eval_correct_eval=evaluation(logits_val,labels_val,'val')
			# train_op=slim.learning.create_train_op(loss,optimizer)

			# logdir=FLAGS.log_dir

			# slim.learning.train(train_op,logdir,number_of_steps=1000,
			# 	save_summaries_secs=300,save_interval_secs=600)

			summary_op=tf.summary.merge_all()

		init_op=tf.initialize_all_variables()

		saver=tf.train.Saver()
		config=tf.ConfigProto()
		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			summary_writer=tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
			coord=tf.train.Coordinator()

			threads=tf.train.start_queue_runners(sess=sess,coord=coord)

			try:
				step=0
				while not coord.should_stop():
					start_time=time.time()
					_,loss_value=sess.run([train_op,loss_total])
					if step%10 == 0:
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)
						# print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
						#                                  duration))
						print('step %d : loss = %.4f' %(step,loss_value))
						precision_test=do_eval(sess,eval_correct_eval,log_name,'val')
                                                logfile=open(log_name,'a')
						logfile.write('Step %d: loss = %.4f \n' % (step, loss_value))
						logfile.close()

					if step%100 == 0 or step == FLAGS.max_steps:
						logfile=open(log_name,'a')
						logfile.write('Train:\n')
						logfile.close()

						print('Train:')
						do_eval(sess,eval_correct,log_name,'train')

						logfile=open(log_name,'a')
						logfile.write('Val:\n')
						logfile.close()
						print('Val:')
						#precision_test=do_eval(sess,eval_correct_eval,log_name,"val")
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)

       					#if step%10000 == 0 or step == FLAGS.max_steps:
					if step%2000 == 0 and precision_test > 0.98:
                                          	checkpoint_file=FLAGS.log_dir+'/'+"alexnet_model_"+str(step)+'_'+str(precision_test)
						saver.save(sess,checkpoint_file)
 
                                        if step%10000 == 0 or step == FLAGS.max_steps:
                                                checkpoint_file=FLAGS.log_dir+'/'+"alexnet_model_"+str(step)

                                                saver.save(sess,checkpoint_file)

                                                

					step+=1
			except tf.errors.OutOfRangeError:
				f=open(log_name,'a')
				f.write('Done training for  epochs,steps.\n' )
				f.close()
			finally:
				coord.request_stop()

			coord.join(threads)

if __name__=="__main__":
	run_training()


