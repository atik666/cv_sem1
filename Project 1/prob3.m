clear; clc;

load observe.dat % pixel cordinates u,v
[i,j] = size(observe); 
load model.dat % world cordiantes x,y,z

world = model;

i1 = 27;

% Displaying  pixel cordiantes obatined using projection matrix
for i5=1:1:i1
world1(:,i5) = transpose([world(i5,:) 1]);
Z(i5) = dot(m3,world1(:,i5));
end