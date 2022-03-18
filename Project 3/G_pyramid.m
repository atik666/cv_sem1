function [g_pyramid] = G_pyramid(img_bw,levels)

g_pyramid= cell(1,levels);
gauss_f= [1/256 4/256 6/256 4/256 1/256; 4/256 16/256 24/256 16/256 4/256; 6/256 24/256 36/256 24/256 6/256; 4/256 16/256 24/256 16/256 4/256; 1/256 4/256 6/256 4/256 1/256];%gaussian filter
g_pyramid{1}= conv2(double(img_bw),gauss_f,'same'); %convolving the original image for the first level
for i=2:levels
     g_pyramid{i}=conv2(double(g_pyramid{i-1}(2:2:end,2:2:end)),gauss_f,'same'); %computing higher levels, taking every second row and colum
end
