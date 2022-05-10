clear;close all;clc;

%k_means initialization
img = imread('mosaicB.bmp');
K = 2;    % no. of clusters
[h,w,d] = size(img);
img_seg = imnoise(img,'gaussian',0.001);
img = double(img_seg);

if d == 1
    array = img(:);
else
    ch1=img(:,:,1);
    ch2=img(:,:,2);

    ch3 = img(:,:,3);
    ar1 = ch1(:);
    ar2 = ch2(:);
    ar3 = ch3(:);
    array = [ar1 ar2 ar3];
end

n_size = size(array,1);

out_img = zeros(h,w,d);

[IDX,C] = kmeans(array,K,'Maxiter',100);

c_sum = zeros(K,1);
cluster = zeros(n_size,K,d);

v_mean = zeros(K,d);  % mean
sigma = zeros(d,d,K);    % standard devision

for i = 1:n_size
    c_sum(IDX(i)) = c_sum(IDX(i)) + 1;
    cluster(c_sum(IDX(i)),IDX(i),:) = array(i,:);
end

for i = 1:K
    v_mean(i,:) = sum(cluster(1:c_sum(i),i,:))./c_sum(i);
    if d == 1
        cluster_main = cluster(1:c_sum(i),i);
    else
        cluster1 = cluster(1:c_sum(i),i,1);
        cluster2 = cluster(1:c_sum(i),i,2);
        cluster3 = cluster(1:c_sum(i),i,3);
        cluster_main =[cluster1 cluster2 cluster3];
    end
    sigma(:,:,i) = cov(cluster_main);
end

