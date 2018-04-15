import cv2
import numpy as np
import os

#pic_name="/home/goerlab/Welder_detection/dataset/20171128/GVS2SP2017/trunk/solder_pad/Data/train/09/GVS2SP2017_train_009_0000020.JPG"

def seg_crop(pic_name):
	image = cv2.imread(pic_name)
	#cv2.imshow("origin",image)
	#image=cv2.resize(image,(512,512))
	#cv2.imshow("origin",image)
	image2=np.zeros(image.shape,np.uint8)
	image2=image.copy()
	shape=image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gradX = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)


	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)



	blurred = cv2.blur(gradient, (9, 9))

	(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)



	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	closed = cv2.erode(closed, None, iterations=4)
	closed = cv2.dilate(closed, None, iterations=4)


	(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	print(len(cnts))

	M=cv2.moments(c)
	Mx=int(M['m10']/M['m00'])
	My=int(M['m01']/M['m00'])

	#cv2.circle(image,(Mx,My-75),260,(0,0,0),-1)

	#cv2.ellipse(image,(Mx,My-40),(260,360),0,360,0,(0,0,0),-1)
	#cv2.ellipse(image, (Mx, My - 40), (260, 380), 0, 360, 0, (0, 0, 0), -1)
	stencil = np.zeros(image.shape).astype(image.dtype)
	color=[255,255,255]
	#cv2.fillConvexPoly(stencil,c,color)
	cv2.fillConvexPoly(stencil,c,color)
	#print(c)
	#cv2.polylines(stencil,c,1,(255,255,255))
	#cv2.imshow("origin",image2)
	#cv2.imshow("stencil",stencil)

	result=cv2.bitwise_and(image2,stencil)
	#cv2.imshow("seg_result",result)
	#cv2.imwrite("seg_"+pic_name,result)
	#cv2.circle(result,(Mx,My-75),260,(0,0,0),-1)
	#cv2.ellipse(result,(Mx,My-40),(280,390),0,360,0,(0,0,0),-1)

	x,y,w,h=cv2.boundingRect(c)
	segImg=result[y:y+h,x:x+w]
	cropImg = image[y:y+h, x:x+w]
	# cv2.imshow("segImg",segImg)
	# cv2.imshow("cropImg",cropImg)
	# cv2.waitKey()

	return segImg,cropImg


	# cv2.imshow("seg_rec",segImg)
	# cv2.imwrite("seg1.jpg",segImg)

	# cv2.imshow("crop_rec",cropImg)
	# cv2.imwrite("crop1.jpg",cropImg)

	# cv2.waitKey()
if __name__=="__main__":
	src_dir="/home/goerlab/Welder_detection/dataset/20171214/val/01/"
	#seg_dst_dir="/media/goerlab/My Passport/Welder_detection/dataset/20171208/val_large/09_seg_ellipse/"
	crop_dst_dir="/home/goerlab/Welder_detection/dataset/20171214/val/01_crop3/"
	if not os.path.exists(crop_dst_dir):
		os.makedirs(crop_dst_dir)
	# filename="GVS2SP2017_train_100_0000000.JPG"
	# segImg,cropImg=seg_crop(filename)
	# cv2.imwrite("seg_"+filename,segImg)
	# cv2.imwrite("crop_"+filename,cropImg)
	for dirc in os.listdir(src_dir):
		print("dirc: %s" %(dirc))
		filename=src_dir+dirc
		segImg,cropImg=seg_crop(filename)
		#cv2.imwrite(seg_dst_dir+"seg_"+dirc,segImg,[int(cv2.IMWRITE_JPEG_QUALITY),100])
		cv2.imwrite(crop_dst_dir+"crop_"+dirc,cropImg,[int(cv2.IMWRITE_JPEG_QUALITY),100])
		# break
		# cv2.imwrite(dst_dir+"crop_"+dirc,cropImg)
