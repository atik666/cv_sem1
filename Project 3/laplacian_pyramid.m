% Laplacian pyramid
close all; clear; clc;

dir = 'D:/PhD/OneDrive - Oklahoma A and M System/CM/Project 3/Samples/';

img = {};
for i = 1:59
    img{i,1} = imread(sprintf('%s/D%d.bmp', dir, i));
end

level = 5;
[norm_feat, img_blocks] = feat_pyramid(img, level);


for i = 1:100
    new = img_blocks{6, 1}{i,1};

    l_pyramid1 = L_pyramid(new,5);

    a1 = l_pyramid1{1,5};
    avr1 = mean2(a1);
    varinece1 = var(a1,[],[1 2]);
    skew1 = skewness(a1,[],[1 2]);
    kurt1 = kurtosis(a1,[],[1 2]);

    feat1 = [avr1, varinece1, skew1, kurt1];

    norm1{i,1} = normalize(feat1,2);
    
end

D = {};
for m = 1:59
    for i = 1:100
        D{i,m}  = sqrt(sum((norm_feat{m,1} - norm1{i,1}) .^ 2));
    end
end
sum_err = sum(cell2mat(D));
d = cell2mat(D);

a = d(56,1:end);

for i = 1:59
[min_error(i,1), min_ind(i,1)] = min(d(j,i));
end


aa = sqrt(sum((norm_feat{m,1} - norm1{i,1}) .^ 2)); 

