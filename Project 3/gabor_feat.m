close all; clear; clc;
file = load('Miv.mat');
file = file.Miv;

for i = 1:size(file,1)
    for m = 1:size(file,2)
        varinece(i,m) = var(file{i,m});
        avr(i,m) = mean(file{i,m});
        skew(i,m) = skewness(file{i,m});
        kurt(i,m) = kurtosis(file{i,m});
    end
end

features = [varinece, avr, skew, kurt];
num_feat = 4;
norm_feat = normalize(features(:,1:num_feat*size(file,2)), 'range');
%save('norm_feat_gabor.mat', 'norm_feat', '-v7.3')

dir = 'D:/PhD/OneDrive - Oklahoma A and M System/CM/Project 3/Samples/';

img = {};
for i = 1:59
    img{i,1} = imread(sprintf('%s/D%d.bmp', dir, i));
end

nrows = 10; 
ncols = 10;

image_blocks = {};
for i = 1:59
    subimages = mat2cell(img{i,1}, ones(1, nrows) * size(img{i,1}, 1)/nrows, ones(1, ncols) * size(img{i,1}, 2)/ncols, 1);
    image_blocks{i,1} = reshape(subimages, 100, 1);
end



