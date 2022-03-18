%% matched filter2，测试程序
clc,close all,clear all;
img=imread('../retina2.jpg');
J(:,:)=img(:,:,2); 
if length(size(J))==3
    J=rgb2gray(J);
end

[g,bg]=matchedFilter2(J);

function [g,bg]=matchedFilter2(f)
 
f=double(f);
subplot(1,2,1);
A=uint8(f);
subplot(2,2,1);
imshow(A,[]);title('origin imag.png');     % after thining
imwrite(A,'origin imag.png')
% mean filter
f=medfilt2(f,[5,5]);
% f=medfilt2(f,[21 1]);
% f=medfilt2(f,[1,7]);
% 参数
os=12;  % 角度的个数
sigma=2;
tim=3;
L=9;
t=120; % 全局阈值,需要多次尝试
 
thetas=0:(os-1);
thetas=thetas.*(180/os);
N1=-tim*sigma:tim*sigma;
N1=-exp(-(N1.^2)/(2*sigma*sigma));
N=repmat(N1,[2*floor(L/2)+1,1]);
r2=floor(L/2);
c2=floor(tim*sigma);
[m,n]=size(f);
RNs=cell(1,os);  % rotated kernals
MFRs=cell(1,os); % filtered images
g1=f;
 
% matched filter
for i=1:os
    theta=thetas(i);
    RN=imrotate(N,theta);
    %去掉多余的0行和零列
    RN=RN(:,any(RN));
    RN=RN(any(RN'),:);
    meanN=mean2(RN);
    RN=RN-meanN;
    RNs{1,i}=RN;
    MFRs{1,i}=imfilter(f,RN,'conv','symmetric');
end
 
% get the max response
g=MFRs{1,1};
for j=2:os
    g=max(g,MFRs{1,j});
end

bg=g<t;
% bg = im2bw(bg);


subplot(2,2,2);
imshow(bg);
title('filterd image');


Z=g;
[P,Pn]=bwlabel(g,8);            % length filtering
for i=1:Pn
    [r, c]=find(P==i);
    [rn, r2]=size(r);
    if rn<20                      % the threshold of length filtering 
        Z(r,c)=0;               % remove edge contours with less 100 pixels
    end
end
subplot(2,2,3);
imshow(Z,[]);title('edge detection result after length filtering');     % the edge detection result after length filtering
imwrite(Z,'after length filtering.png')

% % Use Top-Hat transform
% % se = strel('disk',10);
% % im_top = imtophat(Z,se);  


level = graythresh(Z)
% BW = imbinarize(Z,level);
% imBw = im2bw(Z);  
imBw = imbinarize(Z,level);
% imshowpair(img,BW,'montage')

BW2 = bwmorph(imBw, 'thin');
    % Delete small connected component
% BW2 = bwareaopen(imBw,100,8);
subplot(2,2,4);
imshow(BW2,[]);title('after thining.png');     % after thining
imwrite(BW2,'after thining.png')
% function [img]=xueguanLianTongYu(J)
% if length(size(J))>2
%     J = rgb2gray(J);
% end
% if ~islogical(J)
%     imBw = im2bw(J);                        %转换为二值化图像
% else
%     imBw = J;
% end
% imBw = im2bw(J);                        %转换为二值化图像
% BW2 = bwmorph(imBw, 'thin');
% subplot(2,2,4);
% imshow(BW2,[]);title('after thining.png');     % after thining
% imwrite(BW2,'after thining.png')


end