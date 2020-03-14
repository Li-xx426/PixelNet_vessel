% NOTE - this demo code is for a single scale only --
% a minor modification is required for multi-scale
clc; clear all;

addpath(genpath('../../tools/caffe'));
conv_cache = ['image_crop_results/'];
if(~isdir(conv_cache))
        mkdir(conv_cache);
end

% initialize caffe
net_file     = '../train/cachedir/seg/(06).caffemodel';
deploy_file  = '../train/data/config/deploy.prototxt'; 
load('colormap.mat');

% set the gpu --
% if not using GPU, set it to CPU mode.
gpu_id = 0;
caffe.reset_all;
caffe.set_device(gpu_id);
caffe.set_mode_cpu;
net = caffe.Net(deploy_file, net_file, 'test');

cnn_input_size = 224;
crop_height = 224; crop_width = 224;
%image_mean = cat(3,  103.9390*ones(cnn_input_size),...
%		     116.7700*ones(cnn_input_size),...
%		     123.6800*ones(cnn_input_size));
         
image_mean = cat(3,  23.4366*ones(cnn_input_size),...  %三维数组(每层大小224*224)
		      60.5326*ones(cnn_input_size),...
		     124.6872*ones(cnn_input_size));
         
         
% read the image set -- taking random examples here
egg_direction = ['img/'];%测试图片路径
img_data=dir(fullfile(egg_direction,'*.jpg'));
img_data = struct2cell(img_data);
img_data=img_data(1,:);  %所有图像
%img_data = {'005 (2).bmp','006 (2).bmp'};

% for each image in the img_set
for i = 1:length(img_data)

	display(['Image : ', img_data{i}]);   %显示图像1张
	%ith_Img = im2uint8(imread(['/home/iva/PixelNet/experiments/demo/', img_data{i}]));
    ith_Img = im2uint8(imread([egg_direction, img_data{i}]));%读取图像1张
	%
        save_file_name = [conv_cache, img_data{i}];
        if(exist([save_file_name], 'file'))
                continue;
        end
	 
        j_ims = single(ith_Img(:,:,[3 2 1]));%rgb-bgr(3通道1通道变换)
   
        j_tmp = imresize(j_ims, [cnn_input_size, cnn_input_size], ...%图像大小imresize为224*224
                           'bilinear', 'antialiasing', false);
        j_tmp = j_tmp - image_mean;%预处理(减均值)
        ims(:,:,:,1) = permute(j_tmp, [2 1 3]);	%重置矩阵使图像按照第二维，第一维，第三维顺寻排列，取第一通道，ims是四维 然后最后一维存了个三维矩阵


        %
        net.blobs('data').reshape([crop_height+200, crop_width+200, 3, 1]); %行列3通道1张图
	    net.blobs('pixels').reshape([3,crop_height*crop_width]);%pixels
        h = crop_height;%h=224
        w = crop_width;%h=224
        hw = h * w;%hw=224*224

        xs = reshape(repmat(0:w-1,h,1), 1, hw) + 100; %xs:[1,224],100:323
        ys = reshape(repmat(0:h-1,w,1)', 1, hw)+ 100; %ys:[1,224],100:323


	% set the image data --输入图像数据
        input_data = zeros(crop_height+200,crop_width+200,3,1);%4维 424*424*3*1
        input_data(101:crop_width+100, 101:crop_width+100, :, 1) = ims;
        net.blobs('data').set_data(input_data);%初始化data：4维 424*424*3*1 行列前224行赋值为101：324，第三层3，第四层为图像值
	
	% set the pixels --输入pixels数据
        input_index = zeros(3, 224*224);%初始化input_index[3,224*224]
        input_index(1,:) = 0;%第一行赋值为0
        input_index(2,:) = xs;%第二行赋值为xs xs:[1,224],100:323
        input_index(3,:) = ys;%第三行赋值为ys ys:[1,224],100:323
        net.blobs('pixels').set_data(input_index); %放入pixels中

	% feed forward the values --
        tic;
        net.forward_prefilled();%前向计算 测试结果
        tim = toc;
        fprintf('version: tic --toc : the for statements use time %15.5f.\n',tim);
        
        out = net.blobs('f_score').get_data();%分类
        
        % reshape the data --
        f2 =  out';
        [~, f2] = max(f2, [], 2);%就是返回f2矩阵中每行的最大值，和最大值所在的列数，y就是每行的最大值，i最大值所在的列数 
        f2 = f2 -1;
        f2 = reshape(f2, [224, 224,1]);
        f2 = permute(f2, [2,1,3]);

        % resize to the original size of image
       predns = imresize(f2, [size(ith_Img,1), size(ith_Img,2)],...%将图返回到原图大小
                                'nearest');
%         predns =f2;
        predns = uint8(predns);
        imwrite(predns,colormap, save_file_name);

end

% reset caffe
caffe.reset_all;

