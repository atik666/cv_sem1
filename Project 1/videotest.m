clear;clc;
T=imread('test_image.bmp');
imshow(T);

for i=1:100
L=T;
% some processing in L
Frame(:,:,1)=L/i*2;  % Red channel
Frame(:,:,2)=L/i*2;  % Blue channel
Frame(:,:,3)=L/i*2;  % Green channel
Mo(i)=im2frame(Frame);
end

image(Frame)

v = VideoWriter('Mo.avi','Archival');
open(v)
writeVideo(v,L);
