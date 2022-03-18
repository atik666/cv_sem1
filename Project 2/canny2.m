clc; clear; 

% Edge filtering using upper and lower threshold
thres_high = 0.2;
thres_low = 0.1;
kernel_dim = 7;
pixels = 10;

% Loading the image
I = imread('retina1.jpg');
img(:,:) = I(:,:,2); % Take the green channel only
%img(:,:) = rgb2gray(I);

% Visualize the loaded image
%figure; imshow(img); title('A retinal image');

k = fspecial('gaussian',kernel_dim,2); % Initializing the gaussian kernel

k_image=imfilter(img, k); % Smoothing with filter
%figure; imshow(k_image); title('Smoothing with filter');

% Vertical and horizontal kernels
kx=[-1 0 1; -2 0 2; -1 0 1];
ky=[-1 -2 -1; 0 0 0; 1 2 1];

gx=imfilter(k_image, kx); % x gradient
gy=imfilter(k_image, ky); % y gradient

% figure; imshow(gx); title('x gradient');
% figure; imshow(gy); title('y gradient');

mag = sqrt(double(gx).^2+double(gy).^2); %  Magnitude of the Gradient

angle = atan(double(gy)./double(gx)); % Orientation of change

% Finding the gradient orientation
for i=1:1:size(angle,1)
    for j=1:1:size(angle,2)
        if(angle(i,j)>=0)
            %positive, clock-wise
            if(abs(angle(i,j))<(pi/8))
                angle(i,j)=0;
            elseif(abs(angle(i,j))<((pi/4)+(pi/8)))
                angle(i,j)=pi/4;
            elseif(abs(angle(i,j))<((pi/2)+(pi/8)))
                angle(i,j)=pi/2;
            elseif(abs(angle(i,j))<((3*pi/4)+(pi/8)))
                angle(i,j)=3*pi/4;
            else
                angle(i,j)=pi;
            end
        else
            %negative, counter clock-wise
            if(abs(angle(i,j))<(pi/8))
                angle(i,j)=0;
            elseif(abs(angle(i,j))<((pi/4)+(pi/8)))
                angle(i,j)=-pi/4;
            elseif(abs(angle(i,j))<((pi/2)+(pi/8)))
                angle(i,j)=-pi/2;
            elseif(abs(angle(i,j))<((3*pi/4)+(pi/8)))
                angle(i,j)=-3*pi/4;
            else
                angle(i,j)=-pi;
            end
        end
    end
end

mag_pad = padarray(mag,[1 1],0,'both'); % Pad the image boundaries

% Finding the 4 case of no maximun supression
for i=1+1:1:size(mag,1)+1
    for j=1+1:1:size(mag,2)+1
        if(angle(i-1,j-1)==0||(angle(i-1,j-1)==pi)||(angle(i-1,j-1)==-pi)||(angle(i-1,j-1)==2*pi))
            if((mag_pad(i,j)<mag_pad(i,j-1))||(mag_pad(i,j)<mag_pad(i,j+1)))
                mag(i-1,j-1)=0;
            end
        elseif((angle(i-1,j-1)==pi/4)||(angle(i-1,j-1)==(-3*pi/4)))
            if((mag_pad(i,j)<mag_pad(i-1,j-1))||(mag_pad(i,j)<mag_pad(i+1,j+1)))
                mag(i-1,j-1)=0; 
            end
        elseif(angle(i-1,j-1)==(pi/2)||(angle(i-1,j-1)==(-pi/2)))
            if((mag_pad(i,j)<mag_pad(i-1,j))||(mag_pad(i,j)<mag_pad(i+1,j)))
                mag(i-1,j-1)=0;
            end
        elseif(angle(i-1,j-1)==(3*pi/4)||(angle(i-1,j-1)==(-pi/4)))
            if((mag_pad(i,j)<mag_pad(i+1,j-1))||(mag_pad(i,j)<mag_pad(i-1,j+1)))
                mag(i-1,j-1)=0; 
            end
        end
    end
end

%Normalizing the gradient magnitude
mag_norm = mag - min(mag(:));
mag_norm = mag_norm ./ max(mag_norm(:));
% High and low threshold 
high_edge=max(mag_norm(:));
low_edge=median(max(mag_norm));    

%double thresholding and edge tracking
for i=1:1:size(mag_norm,1)
    for j=1:1:size(mag_norm,2)
        if(mag_norm(i,j)<thres_low)
            mag_norm(i,j)=0;
        elseif(mag_norm(i,j)<thres_high)
            mag_norm(i,j)=low_edge;
        else
            mag_norm(i,j)=high_edge;
        end
    end
end

%figure; imshow(mag_norm); title('Thresholding and edge tracking');

% Clearing the clutter
for i=1+1:1:size(mag_norm,1)-1
    for j=1+1:1:size(mag_norm,2)-1
        if(mag_norm(i,j)==low_edge)
            if((mag_norm(i+1,j+1)==high_edge)||(mag_norm(i,j+1)==high_edge)||(mag_norm(i+1,j)==high_edge)||(mag_norm(i-1,j-1)==high_edge)||(mag_norm(i,j-1)==high_edge)||(mag_norm(i-1,j)==high_edge)||(mag_norm(i-1,j+1)==high_edge)||(mag_norm(i+1,j-1)==high_edge))
                mag_norm(i,j)=high_edge;
            else
                mag_norm(i,j)=0;
            end
        end
    end
end

figure; imshow(mag_norm); title('Canny detection');

% Thining the edges
norm_img = mag_norm;

[L,n]=bwlabel(norm_img,8);            % length filtering
for i=1:n
    [r, c]=find(L==i);
    [rn, r2]=size(r);
    if rn<pixels                      % the threshold of length filtering 
        norm_img(r,c)=0;               % remove edge contours with less pixels
    end
end

level = graythresh(norm_img); % Threshold level
imBw = imbinarize(norm_img,level); % Binary image
morph = bwmorph(imBw, 'thin');

figure; imshow(morph); title('Thinning result');

output=mag_norm;  

% Superimposed result
% C = imfuse(morph,I, 'blend');
% figure; imshow(C); title('Superimposed result');

% subplot(2,2,1);
% imshow(I); title('A retinal image');
% subplot(2,2,2);
% imshow(mag_norm); title('Canny detection');
% subplot(2,2,3);
% imshow(morph); title('Thinning result');
% subplot(2,2,4);
% imshow(C); title('Superimposed result');