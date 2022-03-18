clc;clear;
%% Brief: Record video using writeVideo()
%% For the project, you can use patch() to refresh the new image
T = imread('test_image.bmp');

v = VideoWriter('ObjectMoving.avi');
open(v);

figure
imshow(T);

for i=1:100 
L=T;

% some processing in L
Frame(:,:,1)=L/i*2;  % Red channel
Frame(:,:,2)=L/i*2;  % Blue channel
Frame(:,:,3)=L/i*2;  % Green channel
Mo(i)=im2frame(Frame);
writeVideo(v, Mo)
end

% Mo = getframe;
% for tmp = 0:100
% writeVideo(v, Mo)
% end

close(v)

