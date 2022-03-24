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
num_feat = input("Enter the number of features: \n");
norm_feat = normalize(features(:,1:num_feat*size(file,2)), 'range');
%save('norm_feat_gabor.mat', 'norm_feat', '-v7.3')

dir = '/home/admin1/Documents/Atik/CM/Project 3/Samples/';

img = {};
for i = 1:59
    img{i,1} = imread(sprintf('%s/D%d.bmp', dir, i));
end

nrows = 10; 
ncols = 10;
Ns=4; No=6;	

image_blocks = {};
for i = 1:59
    subimages = mat2cell(img{i,1}, ones(1, nrows) * size(img{i,1}, 1)/nrows, ones(1, ncols) * size(img{i,1}, 2)/ncols, 1);
    image_blocks{i,1} = reshape(subimages, 100, 1);
end

file2 = load('Miv_blocks.mat');
Miv2 = file2.Miv2;

varinece_n = {};
avr_n = {};
skew_n = {};
kurt_n = {};
for m = 1:59
    for i = 1:nrows*ncols
        for j = 1: Ns*No
            a = Miv2{m,1}{i,j};
            varinece_n{m,1}{i,j} = var(a);
            avr_n{m,1}{i,j} = mean(a);
            skew_n{m,1}{i,j} = skewness(a);
            kurt_n{m,1}{i,j} = kurtosis(a);
        end
    end
end

norm_feat_n = {};
for m = 1:59
    feat_1 = cell2mat(varinece_n{m, 1});
    feat_2 = cell2mat(avr_n{m, 1});
    feat_3 = cell2mat(skew_n{m, 1});
    feat_4 = cell2mat(kurt_n{m, 1});
    feat_n = [feat_1, feat_2, feat_3, feat_4];

    norm_feat_n{m,1} = normalize(feat_n(:,1:num_feat*Ns*No), 'range');
end

err = {};
for l = 1:59
    for m = 1:59
        for i = 1:nrows*ncols
            err{l,1}{i,m} = sqrt(sum((norm_feat(m,:) - norm_feat_n{l,1}(i,:)).^2));
        end
    end
end

error = {};
index = {};
for i = 1:59
    block = cell2mat(err{i,1});
    for j = 1:size(block,1)
        [min_error(j,1), min_ind(j,1)] = min(block(j,:));
        error{i,1} = min_error;
        index{i,1} = min_ind;
    end   
    count(i,1) = sum((index{i, 1} == i));
end

avr_pcc = sum(count)/59;
fprintf("The average pcc is: %.2f%% \n", avr_pcc);
