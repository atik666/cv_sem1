clc;
clear all;
close all;
load observe.dat % pixel cordinates u,v
[i,j] = size(observe); 
load model.dat % world cordiantes x,y,z

for w=1:1:i
    for l=1:1:j
        pixel(w,l) = observe(w,l); %pixel matrix
    end
end
[I,J] = size(model);
for k=1:1:I
    for m=1:1:J
        world(k,m) = model(k,m); % world matrix 
    end
end
i1 = input("Enter how many 3d-2d point pairs should the algorithm read");
% creating Q matrix which has dimension of 2i1 x 12
%i=2;
%Q1 = [world(i) world(i+(i1)) world(i+2*i1) 1 zeros(1,4) -pixel(i)*world(i) -pixel(i)*world(i+i1) -pixel(i)*world(i+2*i1) -1*pixel(i);zeros(1,4) world(i) world(i+(i1)) world(i+2*i1) 1  -pixel(i+i1)*world(i) -pixel(i+i1)*world(i+i1) -pixel(i+i1)*world(i+2*i1) -1*pixel(i+i1)]

sum =1 ;
for a=1:2:2*i1  %2*i1
    
   Q(a,: ) =  [world(sum) world(sum+i1) world(sum+(2*i1)) 1 zeros(1,4) -pixel(sum)*world(sum) -pixel(sum)*world(sum+i1) -pixel(sum)*world(sum+(2*i1)) -1*pixel(sum)];

   Q(a+1,: ) =  [zeros(1,4) world(sum) world(sum+(i1)) world(sum+(2*i1)) 1  -pixel(sum+i1)*world(sum) -pixel(sum+i1)*world(sum+i1) -pixel(sum+i1)*world(sum+(2*i1)) -1*pixel(sum+i1)];
   
   sum = sum+1;
end

%M = zeros(12,1);
E=transpose(Q)*Q;
%min(eig(E));
[V,D] = eig(E);
m = V(:,1);
M=transpose(reshape(m,4,3));

for b=1:1:3
    for c=1:1:3
        A(b,c) = M(b,c);
    end
end
    
A1 = A(1,:);
A2 = A(2,:);
A3 = A(3,:);
B  = M(:,4);


R = -1/(norm(A3))

u0 = (R^2)*(dot(A1,A3))

v0 = (R^2)*(dot(A2,A3))

CR1 = cross(A1,A3);
CR2 = cross(A2,A3);

tetha = acosd(-(dot(CR1,CR2))/(norm(CR1)*norm(CR2)))

alpha = R^2*norm(CR1)*sin(tetha)
beta = R^2*norm(CR2)*sin(tetha)

r1 = CR2/norm(CR2)
r3 = R*A3
r2 = cross(r3,r1)
K  = [alpha -alpha*cot(tetha) u0;0 beta/sin(tetha) v0;0 0 1]
t  = R* inv(K)*B

% z= m3p;
m1 = M(1,:);
m2 = M(2,:);
m3 = M(3,:);


% Displaying Q * m  to check the optimization
Q*m

% Displaying  pixel cordiantes obatined using projection matrix
for i5=1:1:i1
world1(:,i5) = transpose([world(i5,:) 1]);
Z(i5) = dot(m3,world1(:,i5));
end

for i5=1:1:i1
px1(i5) = dot(m1,world1(:,i5))/Z(i5);
px2(i5) = dot(m2,world1(:,i5))/Z(i5);
end

for i5=1:1:i1
   px(i5,:) = [px1(i5) px2(i5)];
end


% loading new_world cordiantes
sum1 = 1;
for i3=0:1:10
    for j3=0:1:10
            new_world1(sum1,:) = [i3 j3 0];
            sum1 = sum1+1;
        end
end


sum2 = 1;
for i3=0:1:10
    for j3=0:1:10 
            new_world2(sum2,:) = [i3 10 j3];
            sum2 = sum2+1;
        end
end
new_world2


sum3 = 1;
for i3=0:1:10
    for j3=0:1:10   
            new_world3(sum3,:) = [10 i3 j3];
            sum3 = sum3+1;
        end
end
new_world3


% Displaying  pixel cordiantes obatined using projection matrix
for i6=1:1:121

    New_world1(:,i6) = transpose([new_world1(i6,:) 1]);
    Z(i6) = dot(m3,New_world1(:,i6));
end
for i6=1:1:121
New1_px1(i6) = dot(m1,New_world1(:,i6))/Z(i6);
New1_px2(i6) = dot(m2,New_world1(:,i6))/Z(i6);
end

for i6=1:1:121
   New1_px(i6,:) = [New1_px1(i6) New1_px2(i6)];
  
end
New1_px






for i6=1:1:121

New_world2(:,i6) = transpose([new_world2(i6,:) 1]);
Z(i6) = dot(m3,New_world2(:,i6));
end
for i6=1:1:121
New2_px1(i6) = dot(m1,New_world2(:,i6))/Z(i6);
New2_px2(i6) = dot(m2,New_world2(:,i6))/Z(i6);
end

for i6=1:1:121
   New2_px(i6,:) = [New2_px1(i6) New2_px2(i6)];
  
end
New2_px






for i6=1:1:121

New_world3(:,i6) = transpose([new_world3(i6,:) 1]);
Z(i6) = dot(m3,New_world3(:,i6));
end
for i6=1:1:121
New3_px1(i6) = dot(m1,New_world3(:,i6))/Z(i6);
New3_px2(i6) = dot(m2,New_world3(:,i6))/Z(i6);
end

for i6=1:1:121
   New3_px(i6,:) = [New3_px1(i6) New3_px2(i6)];
  
end
New3_px




I=imread('test_image.bmp');   % read an image into I
[Ix Iy]=size(I); % the dimension of image I (#row, #column)
figure(1),
imshow(I); % show image I in figure 1
%load observe.dat % read the 2D observation data
%[On Ot]=size(observe) % the dimension of observe data
for i7=1:121
%mx=ceil(New1_px(i7,1)); % read the y coordinates of each point
%my=ceil(New1_px(i7,2)); % read the y coordinates of each point

%mx=ceil(New2_px(i7,1)); % read the y coordinates of each point
%my=ceil(New2_px(i7,2)); % read the y coordinates of each point

mx=ceil(New3_px(i7,1)); % read the y coordinates of each point
my=ceil(New3_px(i7,2)); % read the y coordinates of each point
for J1=mx-2:mx+2 % Mark each point in the image
for k1=my-2:my+2
I(k1,J1)=0;
end
end
end

for i7=1:121
mx=ceil(New1_px(i7,1)); % read the y coordinates of each point
my=ceil(New1_px(i7,2)); % read the y coordinates of each point

%mx=ceil(New2_px(i7,1)); % read the y coordinates of each point
%my=ceil(New2_px(i7,2)); % read the y coordinates of each point

%mx=ceil(New3_px(i7,1)); % read the y coordinates of each point
%my=ceil(New3_px(i7,2)); % read the y coordinates of each point
for J1=mx-2:mx+2 % Mark each point in the image
for k1=my-2:my+2
I(k1,J1)=0;
end
end
end

for i7=1:121
%mx=ceil(New1_px(i7,1)); % read the y coordinates of each point
%my=ceil(New1_px(i7,2)); % read the y coordinates of each point

mx=ceil(New2_px(i7,1)); % read the y coordinates of each point
my=ceil(New2_px(i7,2)); % read the y coordinates of each point

%mx=ceil(New3_px(i7,1)); % read the y coordinates of each point
%my=ceil(New3_px(i7,2)); % read the y coordinates of each point
for J1=mx-2:mx+2 % Mark each point in the image
for k1=my-2:my+2
I(k1,J1)=0;
end
end
end
figure(2),
hold on
imshow(I); % show the marked image in figure 