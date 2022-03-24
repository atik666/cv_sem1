% Laplacian pyramid
close all; clear; clc;

dir = 'D:/PhD/OneDrive - Oklahoma A and M System/CM/Project 3/Samples/';

img = {};
level = 4;
feat = {};
img_blocks = {};
num_sample = 59;
varinece = {};

nrows = 10; 
ncols = 10;

for i = 1:59
    img{i,1} = imread(sprintf('%s/D%d.bmp', dir, i));
    mrp = multiresolutionPyramid(im2double(img{i,1}), level);

    lapp = cell(size(mrp));
    num_levels = numel(mrp);
    lapp{num_levels} = mrp{num_levels};
    for k = 1:(num_levels - 1)
       A = mrp{k};
       B = imresize(mrp{k+1},2,'lanczos3');
       [M,N,~] = size(A);
       lapp{k} = A - B(1:M,1:N,:);
    end
    lapp{end} = mrp{end};
    for m = 1:level
        shape = reshape(lapp{1,m},1,[])';
        varinece{i,m} = var(shape);
        avr{i,m} = mean(shape);
        skew{i,m} = skewness(shape);
        kurt{i,m} = kurtosis(shape);
    end
end

features = [cell2mat(varinece), cell2mat(avr), cell2mat(skew), cell2mat(kurt)];

num_feat = input("Enter the number of features: \n");
norm_feat = normalize(features(:,1:num_feat*level), 'range'); % Feat from original samples

image_blocks = {};
for i = 1:num_sample
    subimages = mat2cell(img{i,1}, ones(1, nrows) * size(img{i,1}, 1)/nrows, ones(1, ncols) * size(img{i,1}, 2)/ncols, 1);
    image_blocks{i,1} = reshape(subimages, 100, 1);
end

l_pyramid_n = {};
varinece_n = {};
avr_n = {};
skew_n = {};
kurt_n = {};
for m = 1:59
    for i = 1:100
        mrp = multiresolutionPyramid(image_blocks{m, 1}{i, 1},level);
        lapp = cell(size(mrp));
        num_levels = numel(mrp);
        lapp{num_levels} = mrp{num_levels};
        for k = 1:(num_levels - 1)
           A = mrp{k};
           B = imresize(mrp{k+1},2,'lanczos3');
           [M,N,~] = size(A);
           lapp{k} = A - B(1:M,1:N,:);
        end
        lapp{end} = mrp{end};
    
        for n = 1:level
            shape_n = reshape(lapp{1, n},1,[])';
            varinece_n{m,1}{i, 1}{1,n} = var(shape_n);
            avr_n{m,1}{i, 1}{1,n} = mean(shape_n);
            skew_n{m,1}{i, 1}{1,n} = skewness(shape_n);
            kurt_n{m,1}{i, 1}{1,n} = kurtosis(shape_n);
        end
    end
end 

norm_feat_n = {};
for m = 1:59
    for i = 1:100
        feat_1 = cell2mat(varinece_n{m, 1}{i, 1});
        feat_2 = cell2mat(avr_n{m, 1}{i, 1});
        feat_3 = cell2mat(skew_n{m, 1}{i, 1});
        feat_4 = cell2mat(kurt_n{m, 1}{i, 1});
        feat_n = [feat_1, feat_2, feat_3, feat_4];
        
        norm_feat_n{m,1}{i,1} = normalize(feat_n(:,1:num_feat*level), 'range');
    end
end

err = {};
for j = 1:59
    for m = 1:59
        for i = 1:100
            err{j,1}{i,m} = sqrt(sum((norm_feat(m,:) - norm_feat_n{j, 1}{i, 1}).^ 2));
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