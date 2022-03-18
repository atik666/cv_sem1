clear; clc;

I=imread('retina1.jpg');
img(:,:)=I(:,:,2);
imshow(img);

size = int(size) // 2
x, y = np.mgrid[-size:size+1, -size:size+1]
normal = 1 / (2.0 * np.pi * sigma**2)
g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal