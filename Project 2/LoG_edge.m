clear; clc;
%LoG_kernel=[0 0 1 0 0; 0 1 2 1 0; 1 2 -16 2 1;0 1 2 1 0;0 0 1 0 0];

K=11;
C=(K+1)/2;
V=1.5;          %the standard deviation of the LoG kernel 
T=0.0001;       %the threshold for detect zero-crossing (ZCs)

for i=1:K
    for j=1:K
        d=(i-C)*(i-C)+(j-C)*(j-C);
        t1=exp(-0.5*d/V/V);
        t2=(d/V/V-2);
        t3=1/V/V/V/V/2;
        LoG_kernel(i,j)=t1*t2*t3;
    end
end

I0=imread('retina1.jpg');

I=double(I0(:,:,2))/255;        %Normalize image pixel values from 0 to 1

J=conv2(I,LoG_kernel,'same');   %2D image filtering

[X, Y]=size(I);  

K=zeros(X,Y);

for i=2:(X-1)
    for j=2:(Y-1)
        m=0;
        a=J(i-1,j-1)*J(i+1,j+1);
        b=J(i-1,j)*J(i+1,j);
        c=J(i,j-1)*J(i,j+1);
        d=J(i+1,j-1)*J(i-1,j+1);
        if (a<0 && abs(a)>T) m=m+1; end;
        if (b<0 && abs(b)>T) m=m+1; end;
        if (c<0 && abs(c)>T) m=m+1; end;
        if (d<0 && abs(d)>T) m=m+1; end;
        if m>0 K(i,j)=1;            end;
    end
end

Z=K;                            % duplicate an edge map

[P,Pn]=bwlabel(K,8);            % length filtering

for i=1:Pn
    [r c]=find(P==i);
    [rn r2]=size(r);
    if rn<100                   % the threshold of length filtering 
        Z(r,c)=0;               % remove edge contours with less 100 pixels
    end
end
subplot(2,2,1),imshow(I,[]);    % the original image
subplot(2,2,2),imshow(J,[]);    % the LoG filtered image 
subplot(2,2,3),imshow(K,[]);    % the initial edge detection results
subplot(2,2,4),imshow(Z,[]);    % the edge detection result after length filtering