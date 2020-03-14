% this is an example code for training a PixelNet model using caffe
% here we consider a 224x224 image 
% uniform sampling of pixels in an image, used for segmentation

% add the list of the paths for the startup
%addpath(genpath('~/caffe-master/matlab/'));
addpath(genpath('../../tools/caffe'));
addpath(genpath('./'));
addpath('../../experiments/test/');

% following are the set of options that need to be 
% set to use the train code --
options.cachepath = ['./cachedir_zpd/'];
options.datapath = ['./data/'];
options.solverpath = ['./data/config/solver.prototxt'];
options.initmodelpath = ['./data/config/VGG16_ILSVRC_16_layers.caffemodel'];
%options.initmodelpath = ['./data/snapshot/mlp_iter_10000.caffemodel'];
options.cnn_input_size = 224;
options.segimbatch = 1;
options.segsamplesize = 4000;
options.segbatchsize = (options.segimbatch)*(options.segsamplesize);
options.trainFlip = 1;
options.seed = 2020;
options.segepoch = 100;
options.saveEpoch = 1;
options.meanvalue = [48.6451, 89.1304, 160.0611];
%options.meanvalue = [102.9801, 115.9465, 122.7717];

gpuid=0;
trainSeg(gpuid,options)
