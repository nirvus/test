import cv2
import numpy as np
import cv2
import numpy as np
import os


def seg_crop(pic_name):
	image = cv2.imread(pic_name)
	image2=np.zeros(image.shape,np.uint8)
	image2=image.copy()
	shape=image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	blurred = cv2.blur(gradient, (9, 9))

	(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)


	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


	closed = cv2.erode(closed, None, iterations=4)
	closed = cv2.dilate(closed, None, iterations=4)



	(_,cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	#print(len(cnts))

	stencil = np.zeros(image.shape).astype(image.dtype)
	color=[255,255,255]

	cv2.fillConvexPoly(stencil,c,color)

	result=cv2.bitwise_and(image2,stencil)


	x,y,w,h=cv2.boundingRect(c)
	cropImg = image[y:y+h, x:x+w]
	return cropImg
	#cv2.imshow("crop_rec",cropImg)
	#cv2.imwrite("crop_"+pic_name,cropImg)

if __name__=="__main__":
	src_dir="/home/th/data2/Welder_detection/code/20171221/Data/val/"
	dst_dir="/home/th/data2/Welder_detection/code/20171221/Data/val_crop/"
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)
	for dirc in os.listdir(src_dir):
		print("dirc: %s" %(dirc))
		subdir=src_dir+dirc+'/'
		dst_subdir=dst_dir+dirc+'/'
		if not os.path.exists(dst_subdir):
			os.makedirs(dst_subdir)
		for subdirc in os.listdir(subdir):
			print("filename: %s" %(subdirc))

			filename=subdir+subdirc

			cropImg=seg_crop(filename)
			cv2.imwrite(dst_subdir+"crop_"+subdirc,cropImg)
