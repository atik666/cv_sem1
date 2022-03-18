function [l_pyramid] = L_pyramid(img_bw,levels)

gauss_f= [1/256 4/256 6/256 4/256 1/256; 4/256 16/256 24/256 16/256 4/256; 6/256 24/256 36/256 24/256 6/256; 4/256 16/256 24/256 16/256 4/256; 1/256 4/256 6/256 4/256 1/256];%gaussian filter
l_pyramid = cell(1,levels);
upsampled_imgs = cell(1,levels);
g_pyramid = G_pyramid(img_bw,levels+1);
for p=1:levels%creating an upsampled Gaussian pyramid level array
    upsampled_imgs{p}=zeros(size(g_pyramid{p},1),size(g_pyramid{p},2));
    for i=1:size(upsampled_imgs{p},1)
        for j=1:size(upsampled_imgs{p},2)
            if mod(i, 2) == 0 && mod(j,2) == 0
              upsampled_imgs{p}(i,j) = g_pyramid{p+1}(i/2,j/2);
            end
        end
    end
    upsampled_imgs{p} = conv2(double(upsampled_imgs{p}),gauss_f,'same');  
    upsampled_imgs{p} = upsampled_imgs{p} * 4;
    l_pyramid{p} = minus( g_pyramid{p}, upsampled_imgs{p}); %the laplacian pyramid level preserves the difference between the upsampled Gaussian pyramid level and the Gaussian pyramid level

end
    