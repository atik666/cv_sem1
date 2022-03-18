clc; clear;
%%%%%%%%%%%%%%%%%% Problem 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_2d = load('observe.dat');  % Load 2D points as camera coordinates

data_3d_tmp = load('model.dat');  % Load 3D points as world coordinates
data_3d = zeros(size(data_3d_tmp, 1), 4);
data_3d(:, 1:3) = data_3d_tmp;
data_3d(:, 4) = 1;

img = imread('test_image.bmp');  % Import test image

[lx, ly] = size(img);
figure(1), imshow(img); 

[On, Ot] = size(data_2d);

for i=1:On
    mx=data_2d(i,1); 
    my=data_2d(i,2);
    for j=mx-2:mx+2
        for k=my-2:my+2
            img(k,j)=0; % Getting all the 2D coordinates from image
        end
    end
end
figure(2), imshow(img);

dots = size(data_3d, 1);
P = zeros(2*dots, 12);

for i = 1:dots
    P((i-1)*2+1, 1:4) = data_3d(i, :);
    P(i*2, 5:8) = data_3d(i, :);
    P((i-1)*2+1, 9:12) = -data_2d(i, 1) * data_3d(i, :);
    P(i*2, 9:12) = -data_2d(i, 2) * data_3d(i, :);
end

Q = P; % Q matrix of dimension of 2n x 12

[V,D] = eig(Q'*Q);

for i = 1:size(D,1)
    D_all(i) = D(i,i);
end

[~, min_index] = min(D_all);

A = V(:, min_index);
M = [A(1:4)';A(5:8)';A(9:12)'];

a1 = M(1, 1:3)';
a2 = M(2, 1:3)';
a3 = M(3, 1:3)';
b = M(:,4);

rho = 1 / norm(a3, 1);
r3 = rho*a3;
r3 = r3';
u0 = (rho^2)*dot(a1, a3);
v0 = (rho^2)*dot(a2, a3);

cosTheta = -dot(cross(a1,a3),cross(a2,a3))/(norm(cross(a1,a3),2)*norm(cross(a2,a3),2));
sinTheta = sqrt(1-cosTheta^2);

alpha = (rho^2)*norm(cross(a1, a3),1)*sinTheta;
beta = (rho^2)*norm(cross(a2, a3), 1)*sinTheta;

tetha = acosd(cosTheta);

r1 = cross(a2,a3)/norm(cross(a2,a3),2);
r2 = cross(r3,r1);
K = [alpha -alpha*cot(tetha) u0;0 beta/sinTheta v0;0 0 1]; % Intrinsic Matrix
     
t = rho * (b\inv(K)); % Translation Matrix
R = [r1' ;r2 ;r3];    % Rotation Matrix

% Optimization error
m_hat = Q * V(:, min_index);

d2d_output = zeros(size(data_3d, 1), 3);
for i = 1:size(data_3d, 1)
    d2d_output(i, :) = M*data_3d(i, :)'/(M(3, :) * data_3d(i, :)');
end
d2d = d2d_output(:, 1:2);

d2d_output = int16(d2d_output); % Simulated coordinates
I = imread('test_image.bmp');
figure(2);
imshow(I);
hold on;

for i = 1:length(d2d_output)
   plot(d2d_output(i,1),d2d_output(i,2),'+r', 'Linewidth', 2); 
end
xlabel('Y'); ylabel('X');
title('\bf Calculated and Plotted points on the same image');

%%%%%%%%%%%%%%%%%% Problem 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cor = length(data_3d_tmp);

% Pixel cordiantesfrom the projection matrix
for loc=1:1:cor
    pixel(:,loc) = transpose([data_3d_tmp(loc,:) 1]);
    Z(loc) = dot(M(3,:),pixel(:,loc));
end

for loc=1:1:cor
    u(loc,1) = dot(M(1,:),pixel(:,loc))/Z(loc);
    v(loc,1) = dot(M(2,:),pixel(:,loc))/Z(loc);
end

P = horzcat(u, v);

% World cordiantes
col = 1;
for i=0:1:10
    for j=0:1:10
        z0(col,:) = [i j 0];
        col = col+1;
    end
end

col = 1;
for i=0:1:10
    for j=0:1:10 
        y10(col,:) = [i 10 j];
        col = col+1;
    end
end

col = 1;
for i=0:1:10
    for j=0:1:10   
        x10(col,:) = [10 i j];
        col = col+1;
    end
end

% For projection matrix
for i=1:1:length(z0)
    z00(:,i) = transpose([z0(i,:) 1]);
    Z(i) = dot((M(3, :)),z00(:,i));
end

for i=1:1:length(z0)
    u1(i,1) = dot((M(1, :)),z00(:,i))/Z(i);
    v1(i,1) = dot((M(2, :)),z00(:,i))/Z(i);
end

P1 = horzcat(u1, v1);

for i=1:1:length(y10)
    y00(:,i) = transpose([y10(i,:) 1]);
    Z(i) = dot((M(3, :)),y00(:,i));
end

for i=1:1:length(y10)
    u2(i,1) = dot((M(1, :)),y00(:,i))/Z(i);
    v2(i,1) = dot((M(2, :)),y00(:,i))/Z(i);
end

P2 = horzcat(u2, v2);

for i=1:1:length(x10)
    x00(:,i) = transpose([x10(i,:) 1]);
    Z(i) = dot((M(3, :)),x00(:,i));
end

for i=1:1:length(x10)
    u3(i,1) = dot((M(1, :)),x00(:,i))/Z(i);
    v3(i,1) = dot((M(2, :)),x00(:,i))/Z(i);
end

P3 = horzcat(u3, v3);

I=imread('test_image.bmp');   % Load image
[Ix, Iy]=size(I); % dimension size
figure(1),
imshow(I);

for i=1:length(x10)
    mx=floor(P3(i,1));
    my=floor(P3(i,2));
    for m1=mx-2:mx+2 % Marking the coordinate
        for n1=my-2:my+2
            I(n1,m1)=0;
        end
    end
end

for i=1:length(y10)
    mx=floor(P2(i,1));
    my=floor(P2(i,2));
    for m1=mx-2:mx+2 
        for n1=my-2:my+2
            I(n1,m1)=0;
        end
    end
end

for i=1:length(z00)
    mx=floor(P1(i,1));
    my=floor(P1(i,2));
    for m1=mx-2:mx+2 
        for n1=my-2:my+2
            I(n1,m1)=0;
        end
    end
end

figure(2),
hold on
imshow(I); % show the simulated figure