# -- coding: utf-8 --
import cv2
import os
import numpy as np

origin_dir="/home/ed/桌面/pix2pix-tensorflow-master/20180319/裁伤_512/"
dst_dir="/home/ed/桌面/pix2pix-tensorflow-master/20180319/裁伤_512_blank/"
for file in os.listdir(origin_dir):
    new_name=dst_dir+file
    img=np.zeros((512,512,3),np.uint8)
    img.fill(255)
    cv2.imwrite(new_name,img)
print("Done")