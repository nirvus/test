from PIL import Image
import os

dir="/home/ed/PPM_2cls/val/00/"
dir2="/home/ed/PPM_2cls/val/000/"
for dirc in os.listdir(dir):
    print dirc

    filename = dir + str(dirc)
    print(filename)
    f1=Image.open(filename)
    f11=f1.convert("RGB")
    # f11.show()
    f11.save(dir2+dirc.split('.')[-2]+".jpg")
