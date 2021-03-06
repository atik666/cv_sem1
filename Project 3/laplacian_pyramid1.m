% Laplacian pyramid
close all; clear; clc;

dir = 'D:/PhD/OneDrive - Oklahoma A and M System/CM/Project 3/Samples/';

img = {};
for i = 1:59
    img{i,1} = imread(sprintf('%s/D%d.bmp', dir, i));
end

level = 5;

feat = {};
norm_feat = {};
img_blocks = {};
num_sample = 59;
varinece = {};

nrows = 10; 
ncols = 10;

for i = 1:num_sample
    l_pyramid = L_pyramid(img{i,1},level);
    for m = 1:level
        varinece{i,m} = var(l_pyramid{1,m},[],[1 2]);
    end
end

norm_feat = normalize(cell2mat(varinece)); % Feat from original samples

image_blocks = {};
for i = 1:num_sample
    subimages = mat2cell(img{i,1}, ones(1, nrows) * size(img{i,1}, 1)/nrows, ones(1, ncols) * size(img{i,1}, 2)/ncols, 1);
    image_blocks{i,1} = reshape(subimages, 100, 1);
end

l_pyramid_n = {};
varinece_n = {};
for m = 1:59
    for i = 1:100
        l_pyramid_n{m,1}{i, 1} = L_pyramid(image_blocks{m, 1}{i, 1},level);
        for n = 1:level
            varinece_n{m,1}{i, 1}{1,n} = var(l_pyramid_n{m, 1}{i, 1}{1, n},[],[1 2]);
        end
    end
end 

var_1 = {};
for m = 1:59
    for i = 1:100
        var_1{m,1}{i,1} = normalize(cell2mat(varinece_n{m, 1}{i, 1}));
    end
end

err = {};
for j = 1:59
    for m = 1:59
        for i = 1:100
            err{j,1}{i,m} = sqrt(sum((norm_feat(m,:) - var_1{j, 1}{i, 1}).^ 2));
        end
    end
end

error = {};
index = {};
for i = 1:59
    block = cell2mat(err{i,1});
    for j = 1:length(block)
    [min_error(j,1), min_ind(j,1)] = min(block(j,:));
    error{i,1} = min_error;
    index{i,1} = min_ind;
    end   
end

for i = 1:59
    count(i,1) = sum((index{i, 1} == i));
end

avr_pcc = sum(count)/59;
fprintf("The average pcc is: %.2f%% \n", avr_pcc);
