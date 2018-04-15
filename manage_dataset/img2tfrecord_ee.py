import os
import random
import shutil
import PIL
import cv2
import numpy as np
import csv
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="6"

def GetFileNameAndExt(filename):
	(filepath,tempfilename) = os.path.split(filename);
	(shotname,extension) = os.path.splitext(tempfilename);
	return filepath,shotname,extension

def translate(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted



def image_flip(dirpath):
	print os.listdir(dirpath)
	for dirc in os.listdir(dirpath):
		#if dir
		subdir=os.listdir(dirpath+dirc)
		print "%s : %d" %(str(dirpath+dirc),len(subdir))
		for subdirc in subdir:
			#print(subdirc)
			filename=dirpath+dirc+'/'+str(subdirc)
			#print(filename)
			filepath,shotname,extension=GetFileNameAndExt(filename)
			tempimg=cv2.imread(filename)
			fliped_img=cv2.flip(tempimg,1)
			newname=filepath+'/'+shotname+"_1"+extension
			print(newname)
			cv2.imwrite(newname,fliped_img)
						


def image_shift(class_dir):
	print os.listdir(class_dir)
	for dirc in os.listdir(class_dir):
		filename=class_dir+str(dirc)
		print(filename)

		image=cv2.imread(filename)
		# cv2.imshow("Origin",image)
		# cv2.waitKey(0)

		filepath,shotname,extension=GetFileNameAndExt(filename)

		#shifted_1_name=filepath+'/'+shotname+"_shifted_1"+extension

		shifted_1_img=translate(image,0,100)
		shifted_1_name=filepath+'/'+shotname+"_shifted_1"+extension
		
		#print(shifted_1_name)
		cv2.imwrite(shifted_1_name,shifted_1_img)

		shifted_2_img=translate(image,0,-100)
		shifted_2_name=filepath+'/'+shotname+"_shifted_2"+extension
		
		#print(shifted_2_name)
		cv2.imwrite(shifted_2_name,shifted_2_img)

		shifted_3_img=translate(image,50,0)
		shifted_3_name=filepath+'/'+shotname+"_shifted_3"+extension
		
		#print(shifted_3_name)
		cv2.imwrite(shifted_3_name,shifted_3_img)

		shifted_4_img=translate(image,0,50)
		shifted_4_name=filepath+'/'+shotname+"_shifted_4"+extension
		
		#print(shifted_4_name)
		cv2.imwrite(shifted_4_name,shifted_4_img)

def ReadeachFIle(filepath,writer):
	print(filepath)
	pathdir=os.listdir(filepath)
	print(pathdir)
	for subdirc in pathdir:
		subdir=filepath+subdirc
		print(subdirc)
		if not os.path.isfile(subdir):
			#subdir=filepath+subdirc
			filelist=os.listdir(subdir)
			for filec in filelist:
				filename=subdir+'/'+filec

				label=int(subdirc)

				print('label:'+str(label))

			
				writer.writerow([filename,label])

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  

def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def load_file(examples_file):
	with open(examples_file) as csvfile:
		reader=csv.reader(csvfile)
		examples=[]
		labels=[]
		cnt=0
		for row in reader:
			examples.append(row[0])
			labels.append(row[1])
			cnt+=1
	
	return examples,labels,cnt

def extract_image(filename,resize_height,resize_width):
	image=cv2.imread(filename)
	image=cv2.resize(image,(resize_height,resize_width))

	rgb_image=image

	#b,g,r=cv2.split(image)
	#rgb_image=cv2.merge([b,g,r])

	return rgb_image

def transform2tfrecord(train_file, name, output_directory, resize_height, resize_width):  
	if not os.path.exists(output_directory) or os.path.isfile(output_directory):  
		os.makedirs(output_directory) 
	print(train_file) 
	_imgdirs, _labels, examples_num = load_file(train_file) 
	print("###########################")
	print(_imgdirs)
	print("#############################") 
	filename = output_directory + "/" + name + '.tfrecords'  
	writer = tf.python_io.TFRecordWriter(filename)  
	for i, [imgdir, label] in enumerate(zip(_imgdirs, _labels)):  
		print('No.%d' % (i))  
		print([imgdir,label])
		image = extract_image(imgdir, resize_height, resize_width)  
		#print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))  
		print('shape: %s, %s, %s, label: %s' % (image.shape[0], image.shape[1], image.shape[2], label))  
		#winlabel="image:"+str(label)
		#cv2.imshow(winlabel,image)
		#cv2.waitKey(0)
		image_raw = image.tostring()  
		example = tf.train.Example(features=tf.train.Features(feature={  
			'image_raw': _bytes_feature(image_raw),  
			'height': _int64_feature(image.shape[0]),  
			'width': _int64_feature(image.shape[1]),  
			'depth': _int64_feature(image.shape[2]),  
			'label': _int64_feature(int(label)),  
			'filename':_bytes_feature(str(imgdir))
		}))  
		writer.write(example.SerializeToString())  
	writer.close() 



if __name__=="__main__":

	train_path="/home/ed/PPM_2cls/train/"
	
	val_path="/home/ed/PPM_2cls/val/"
	md_path="/home/ed/PPM_2cls/tfrecord/"
	if not os.path.exists(md_path):
		os.makedirs(md_path)
	dirpath_train=train_path
	csvfile_train=file(md_path+"/train.csv","wb")

	dirpath_val=val_path
	csvfile_val=file(md_path+"/val.csv","wb")

	#image_flip(train_path)
	#for dirc in os.listdir(train_path):
	#	class_dir=train_path+dirc+'/'
	#	if not os.path.isfile(class_dir):
	#		print(class_dir)
	#		image_shift(class_dir)

	
	writer=csv.writer(csvfile_train)
	ReadeachFIle(dirpath_train,writer)
	csvfile_train.close()


	writer=csv.writer(csvfile_val)
	ReadeachFIle(dirpath_val,writer)
	csvfile_val.close()
	
	csvfile1=open(md_path+'/train.csv','r')
	rows=csvfile1.readlines()
	csvfile1.close()
	random.shuffle(rows)

	csvfile2=open(md_path+"/train2.csv","w")
	csvfile2.writelines(rows)
	csvfile2.close()


	train_file=md_path+"/train2.csv"
	val_file=md_path+"val.csv"
	train_name="train"
	val_name="val"
	output_dir=md_path

	
	resize_height=512
	resize_width=512


	transform2tfrecord(train_file,train_name,output_dir,resize_height,resize_width)
	transform2tfrecord(val_file,val_name,output_dir,resize_height,resize_width)
				



