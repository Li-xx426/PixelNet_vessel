# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys

caffe_root = '/home/lixiaoxing/code/PixelNet/tools/caffe'
sys.path.insert(0, caffe_root+'python/')
import caffe
# Use GPU?
use_gpu = 1;
gpu_id = 3;
net_struct = '/home/lixiaoxing/code/PixelNet/experiments/train/data/config/deploy.prototxt'
data_source = 'img.lst'
data_root = 'image_crop/'
save_root = 'results_py/'
with open(data_source) as f:
    imnames = f.readlines()
test_lst = [data_root + x.strip() for x in imnames]
if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

# load net
net = caffe.Net(net_struct,'/home/lixiaoxing/code/PixelNet/experiments/train/cachedir/seg/(06).caffemodel', caffe.TEST);
for idx in range(0,len(test_lst)):
    print("Scoring snet for image " + data_root + imnames[idx][:-1])

    #Read and preprocess data
    input_data=np.zeros((3,424,424))
    im = Image.open(test_lst[idx])
    im = im.resize((224, 224))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((23.4366,60.5326,124.6872)) # ophtha
    #in_ -= np.array((88.9513,88.9512,88.9513)) # ophtha
    in_ = in_.transpose((2,0,1))
    input_data[:,100:324,100:324]=in_

    #Reshape data layer
    net.blobs['data'].reshape(1, *input_data.shape)
    net.blobs['data'].data[...] = input_data
    xs=np.matlib.repmat(np.arange(224).reshape(224,1),1,224).reshape((1,224*224))[0]+100
    ys=np.matlib.repmat(np.arange(224),1,224).reshape((1,224*224))[0]+100
    print(xs)
    print(ys)
    input_index=np.zeros((3,224*224))
    input_index[0,:]=0
    input_index[1,:]=xs
    input_index[2,:]=ys
    input_index=input_index.transpose(1,0)
    net.blobs['pixels'].reshape(*input_index.shape)
    net.blobs['pixels'].data[...]=input_index

    net.forward()
    result= net.blobs['f_score'].data.argmax(1).reshape((224,224))
    print(result.max())
    scipy.misc.imsave(save_root+imnames[idx][:-1][0:-4]+'.png', result*128)
