clc;
close all;
% start creating alpha (primary) kernel for matched filter
% spliting the kernel equation into three terms t1,t2,m0
% before that define standard deviation variable d

d = input("Enter the value of standard deviation");
%k  = input("Enter the size of kernel");
k = 21;
c = (k+1)/2;
t1 = 1/(sqrt(2*pi*(d^2)));
sum = 0;

%1/sqrt(2*pi*s)*exp(-t.^2/(2*s.^2))
%G0 = repmat(x,L,1);


for i1 = 1:k
    for j1 = 1:k
        t2 = exp((-(j1-c)^2)/2*(d^2));
        G0(i1,j1) = -(t1*t2);
        sum = sum + G0(i1,j1);
    end
end

% defining value of m0 which helps to make average of the kernel as zero
m0 = mean(G0,'all');%sum/(k*k);
G = G0 - m0;
% G0 is guassian primary kernel for matched filter

L = imread("retina1.jpg");
L1= double(L(:,:,2))/255;        %read only green channel

% producing matched filters set
%doing with loop
angle = 0;
W =12;
G1 = cell(W,1);
for i=1:W
    G1{i} = imrotate(G,angle,"bicubic","crop");
    angle = angle+15;
end

% resulted set of filtered images

I = cell(W,1);

for i=1:W
    I{i} = conv2(L1,G1{i},"same");
end

% fusing of all filtered images into a single image by selecting maximum
%pixel value of all the images

[X,Y]=size(L1);
 
Isample= cat(3,I{1},I{2},I{3},I{4},I{5},I{6},I{7},I{8},I{9},I{10},I{11},I{12});
Ifusion = max(Isample,[],3);

% Imax is new fusioned filtered image
Inew = Ifusion;  %duplicating of the image
% tresholding the image and converting the treshold image into binary
level= graythresh(Inew);
BW = imbinarize(Inew,level);
treshold = BW; % making a copy of the image
% length filtering

[V,NUM] = bwlabel(BW,4);

for i=1:NUM
    [r, c]=find(V==i);
    [rn, r2]=size(r);
    if rn<100              % the threshold of length filtering 
        BW(r,c)=0;               % remove edge contours with less 100 pixels
    end
end

%Thinning the edges of the image
Thin = bwmorph(BW,'thin');

% superimposing image
SI = imfuse(L,Thin,'blend');

figure(1);
subplot(2,2,1),imshow(L);    % the original image
hold on;
title("Original image");
subplot(2,2,2),imshow(BW);    % the matched filtered image 
hold on;
title("Match filtered image");
subplot(2,2,3),imshow(Thin);    % Thinning result
hold on;
title("Thinning Result");
subplot(2,2,4),imshow(SI);    % the superimposed result
hold on;
title("superimposed image")
figure(2);
subplot(2,1,1),
imshow(Inew);
title("Fusion image");
subplot(2,1,2);
imshow(treshold);
title("Treshold image");