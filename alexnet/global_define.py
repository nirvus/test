NUM_CLASSES=2
IMAGE_SIZE=224
IMAGE_PIXELS=224*224
INPUT_SIZE=224
TRAIN_FILE='train.tfrecords'
VALIDATION_FILE='val.tfrecords'

TOTAL=182

CLASS_FILE='val.tfrecords'
TRAIN_TOTAL=540
#TRAIN_NUMS=[5770*0.2,450,370,10*10,1520,1000,370,210*2,610,270,170,70*5]
#COEFFS=[0.5/(1.0*TRAIN_TOTAL/i) for i in TRAIN_NUMS]
#TRAIN_NUMS=[5120,560*3,440*3,40*10,1920,1260,470*3,220*3,780,200*3,190*3,60*10]
#COEFFS=[0.5/(1.0*i/TRAIN_TOTAL) for i in TRAIN_NUMS]
#TRAIN_NUMS=[5770,450*5,370*5,10*5,1520,1000,370*3,210*5,610*2,27*50,170*6,70*5]
#COEFFS=[0.5/(1.0*i/TRAIN_TOTAL) for i in TRAIN_NUMS]
TRAIN_NUMS=[300,240]
COEFFS=[0.25/(1.0*i/TRAIN_TOTAL) for i in TRAIN_NUMS]

print(COEFFS)
#TEST_FILE='test.tfrecords'
TFRECORD_PATH="/home/th/data/Welder_detection/code/20171116/train_alexnet/dataset/tfrecord_224_182/"
CUDA_VISIBLE_DEVICES='1'
LEARNING_RATE=0.001
BATCH_SIZE=50
BATCH_SIZE_VAL=50
#CHECKPOINT_PATH="/home/th/data/Welder_detection/model_weight/vgg_16/vgg_16.ckpt"
DESCRIPTION="2cls_alexnet_20180322_sensi_1111_l2"
DIR_DESCRIPTION="result_half"
