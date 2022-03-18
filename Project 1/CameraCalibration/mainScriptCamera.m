clear;
%% Extracting coordinates in terms of image pixels
imagename = 'checkerboard1.jpg';

 % Finds the corner in image as pixel coordinate and save to the given
 % excel files
findCoordinates(imagename,'myExcelW.xlsx','myExcelC.xlsx'); 

data3d = './myExcelW.xlsx';
data2d = './myExcelC.xlsx';

[d3d_input, d2d_input] = data_loading(data3d, data2d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, rho, r3, u0, v0, alpha, beta] = param_estimation(d3d_input, d2d_input);
r3 = r3';
A = M(:,1:3);
b = M(:,4);
a1 = A(1,:)';
a2 = A(2,:)';
a3 = A(3,:)';
cosTheta = -dot(cross(a1,a3),cross(a2,a3))/(norm(cross(a1,a3),2)*norm(cross(a2,a3),2));
sinTheta = sqrt(1-cosTheta^2);
r1 = cross(a2,a3)/norm(cross(a2,a3),2);
r2 = cross(r3,r1);
K = [alpha    0     u0; % Intrinsic Matrix
     0       beta   v0;
     0        0     1];
t = rho * (b\inv(K)); % Translation Matrix
R = [r1' ;r2' ;r3'];    % Rotation Matrix

d2d_output = cor2d_compute(d3d_input, M);
d2d_output = int16(d2d_output);
I = imread(imagename);
figure(2);
imshow(I);
hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting data using calculated pixel coordinates from Marix M
for i = 1:length(d2d_output)
   plot(d2d_output(i,1),d2d_output(i,2),'+r', 'Linewidth', 2); 
end
xlabel('Y'); ylabel('X');
title('\bf Calculated and Plotted points on the same image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(d2d_output)
   m =  d2d_input(i,:)-double(d2d_output(i,:));
   error(i) = sqrt(m*m');
end
%% Error in each computaion
error = error';
figure(3);
stem(error/max(error));
title('Normalized difference between Observed and Modeled');
xlabel('Corner Number');
ylabel('Normalized Pixel distance');
title('\bf Normalised Error at each computed point');
K
R
t