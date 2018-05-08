####################################################
###########This is just the first version########### 
###########The codes will be updated in the future###
####################################Queenie.Liu#####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os.path
import sys
import time
import numpy as np
import datetime
import os
import re
import os
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
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

from tensorflow.contrib.slim.python.slim.nets import vgg
#import alexnet_input_512 as alexnet_512
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=gd.CUDA_VISIBLE_DEVICES

time_value=re.sub(r'[^0-7]','',str(datetime.datetime.now()))

flags.DEFINE_integer('batch_size',1, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('tfrecord_dir', 
	gd.TFRECORD_PATH, 'Directory to put the training data.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('log_dir','Log_data/log'+str(time_value)+'/','balabala')


checkpoint_file=tf.train.latest_checkpoint("./Log_data/12cls_alexnet_20171116_sensi_l2_2017111611641142001")
print('checkpoint:')
print(checkpoint_file)
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

log_name=FLAGS.log_dir+'/'+'test_log.txt'
f=open(log_name,'w')
f.close()



def do_eval(sess,eval_correct,log_name):
	true_count=0
	#for step in xrange(FLAGS.batch_size):
	true_count+=sess.run(eval_correct)

	precision=float(true_count)/FLAGS.batch_size
	# print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
 #            (FLAGS.batch_size, true_count, precision))
 	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n' %
            (FLAGS.batch_size, true_count, precision))
	logfile=open(log_name,'a')
	logfile.write('  Num examples: %d  Num correct: %d  Precision : %0.04f\n' %
            (FLAGS.batch_size, true_count, precision))
	
	logfile.close()
	return precision


def evaluation(logits_in,labels):
	correct=tf.nn.in_top_k(logits_in,labels,1)
	return tf.reduce_sum(tf.cast(correct,tf.int32))

def read_and_decode(filename_queue):

	reader=tf.TFRecordReader()
	_,serialized_exampe=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_exampe,
		features={
		'image_raw':tf.FixedLenFeature([],tf.string),
		'height':tf.FixedLenFeature([],tf.int64),
		'width':tf.FixedLenFeature([],tf.int64),
		'depth':tf.FixedLenFeature([],tf.int64),
		'label':tf.FixedLenFeature([],tf.int64),
                'filename':tf.FixedLenFeature([],tf.string)
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
        filename=tf.cast(features['filename'],tf.string)
	return image,label,filename

# def inputs(batch_size,num_epochs):
# 	if not num_epochs:
# 		num_epochs=None
# 	# if train=='train':
# 	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
# 	# elif train=='val':
# 	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
# 	# else:
# 	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)
# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.CLASS_FILE)

# 	with tf.name_scope('input'):
# 		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
# 		print(filename)
# 		image,label=read_and_decode(filename_queue)
# 		print("input image:")
# 		print(image)
# 		images, sparse_labels = tf.train.shuffle_batch(
#         [image, label], batch_size=FLAGS.batch_size, num_threads=6,
#         capacity=1000 + 3 * batch_size,
#         min_after_dequeue=1000)
# 	return images, sparse_labels

def inputs(batch_size,num_epochs):
	if not num_epochs:
		num_epochs=None
	# if train=='train':
	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
	# elif train=='val':
	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
	# else:
	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)
	filename=os.path.join(FLAGS.tfrecord_dir,gd.CLASS_FILE)

	with tf.name_scope('input'):
		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
		print(filename)
		image,label,filename=read_and_decode(filename_queue)
		print("input image:")
		print(image)
		images, sparse_labels,filenames = tf.train.batch(
		[image, label,filename], batch_size=FLAGS.batch_size, num_threads=1,
		capacity=1000 + 3 * batch_size)
	return images, sparse_labels,filenames

def details_accuray(labels,predictions,num_of_class):
	count_label=[0 for i in range(num_of_class)]
	count_prediction=[0 for i in range(num_of_class)]
	#print(count_label)
	for i in range(len(labels)):
		count_label[labels[i]]+=1
		if labels[i]==predictions[i]:
			count_prediction[predictions[i]]+=1

	print(count_label)
	print(count_prediction)

num_of_class=gd.NUM_CLASSES
count_label=[0 for i in range(num_of_class)]
count_prediction=[0 for i in range(num_of_class)]
confusion_matrix=[[0 for col in range(num_of_class)] for row in range(num_of_class)]






def run_testing():


	with tf.Graph().as_default():

		with slim.arg_scope(vgg.vgg_arg_scope()):

			images,labels,filenames=inputs(FLAGS.batch_size,FLAGS.num_epochs)
			
			images=tf.reshape(images,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			logits,end_points=alexnet.alexnet_v2(images,num_classes=gd.NUM_CLASSES,is_training=False)

			print(labels)

			print(logits)
			
			eps=tf.constant(value=1e-10)
			
			flat_logits=logits+eps

			softmax=tf.nn.softmax(flat_logits)

			probability=tf.reduce_max(softmax,axis=1)
			ll=tf.argmax(logits,axis=1)
			print(ll)
			variables_to_restore=slim.get_variables_to_restore()

			saver=tf.train.Saver(variables_to_restore)

			eval_correct=evaluation(logits,labels)

			
		config=tf.ConfigProto()

		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:

			saver.restore(sess,checkpoint_file)

			coord=tf.train.Coordinator()

			threads=tf.train.start_queue_runners(sess=sess,coord=coord)

			
			step=0

			if not os.path.exists(gd.DIR_DESCRIPTION):
				os.makedirs(gd.DIR_DESCRIPTION)

                        csvfile=open(gd.DIR_DESCRIPTION+"/12cls_2017-11-16_alexnet_sensi_color_change_wrongprediction.csv","a")
                        writer=csv.writer(csvfile)
                        writer.writerow(['labels','prediction','filename'])

			file_name2="/detail_result.csv"
			csvfile2=open(gd.DIR_DESCRIPTION+file_name2,"wb")
			writer2=csv.writer(csvfile2)
			writer2.writerow(['labels','prediction','probability'])

			for step in range(gd.TOTAL):
			#while not coord.should_stop():
				#accuracy=do_eval(sess,eval_correct,log_name)

				labels_out,prediction_out,filename,softmax_out,probability_max=sess.run([labels,ll,filenames,softmax,probability])
				print ("%d : %d ,%d ,max_probability: %f" %(step,labels_out[0],prediction_out[0],probability_max[0]))
				
				writer2.writerow([labels_out[0],prediction_out[0],probability_max[0]])
				#print(labels_out[0])

				#print(prediction_out[0])
                                
				count_label[labels_out[0]]+=1

				if labels_out[0]==prediction_out[0]:
					count_prediction[prediction_out[0]]+=1
                                else:
                                        writer.writerow([labels_out[0],prediction_out[0],filename[0]])
				confusion_matrix[labels_out[0]][prediction_out[0]]+=1
				#details_accuray(labels_out,prediction_out,gd.NUM_CLASSES)
                        csvfile.close()
		print(count_label)
		print(count_prediction)
		print(confusion_matrix)
		print('\n')
		for i in range(num_of_class):
			print(confusion_matrix[i])
		precision_result=[0 for i in range(num_of_class)]
		recall_result=[0 for i in range(num_of_class)]
		#for i in range(num_of_class):
		#	precision_result[i]=confusion_matrix[i][i]/
		precision_sum=map(sum,zip(*confusion_matrix))
		
		print("precision_sum:")
		print(precision_sum)
		for i in range(num_of_class):
			precision_result[i]=confusion_matrix[i][i]/precision_sum[i]
		
		print("average_precision:")
		print(precision_result)
		
		print("mean_average_precision:")
		print(sum(precision_result)/num_of_class)
		
		print("recall_sum:")
		recall_sum=map(sum,confusion_matrix)
		print(recall_sum)
		
		for i in range(num_of_class):
			recall_result[i]=confusion_matrix[i][i]/recall_sum[i]
		print("recall:")
		print(recall_result)
		
		print("mean_recall:")
		print(sum(recall_result)/num_of_class)
		
		print("accuracy:%d/%d" %(sum(count_prediction),sum(count_label)))
		#print(sum(count_prediction))
		#print(count_prediction)
		#print(sum(count_label))
		print(sum(count_prediction)/sum(count_label))
			
if __name__=='__main__':
	run_testing()


