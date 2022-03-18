clear; clc;

I=imread('retina1.jpg');
img(:,:)=I(:,:,2);
imshow(img);

s = 1.5; %sigma
t = -3*s:3*s;
theta=0:15:165; %different rotations
%one dimensional kernel
x = 1/sqrt(2*pi*s)*exp(-t.^2/(2*s.^2));

L=7;
%two dimensional gaussian kernel
x2 = repmat(x,L,1);

m = sum(x2(:))/(size(x2,1)*size(x2,2));
x2 = x2-m; 

%apply rotated matched filter on image
r = {};
for k = 1:12
    x3=imrotate(x2,theta(k),'crop');%figure;imagesc(x3);colormap gray;   
    r{k}=conv2(img,x3);
end

s = 1.5; %sigma
L = 7;
theta = 0:15:165; %different rotations

out = zeros(size(img));

m = max(ceil(3*s),(L-1)/2);
[x,y] = meshgrid(-m:m,-m:m); % non-rotated coordinate system, contains (0,0)
t = pi/6;                    % angle in radian
u = cos(t)*x - sin(t)*y;     % rotated coordinate system
v = sin(t)*x + cos(t)*y;     % rotated coordinate system
N = (abs(u) <= 3*s) & (abs(v) <= L/2);   % domain
k = exp(-u.^2/(2*s.^2));     % kernel
k = k - mean(k(N));
k(~N) = 0;                   % set kernel outside of domain to 0

theta = 0:15:165; %different rotations

out = zeros(size(img));

m = max(ceil(3*s),(L-1)/2);
[x,y] = meshgrid(-m:m,-m:m); % non-rotated coordinate system, contains (0,0)
for t = theta
   t = t / 180 * pi;        % angle in radian
   u = cos(t)*x - sin(t)*y; % rotated coordinate system
   v = sin(t)*x + cos(t)*y; % rotated coordinate system
   N = (abs(u) <= 3*s) & (abs(v) <= L/2); % domain
   k = exp(-u.^2/(2*s.^2)); % kernel
   k = k - mean(k(N));
   k(~N) = 0;               % set kernel outside of domain to 0

   res = conv2(img,k,'same');
   out = max(out,res);
end

T = graythresh(out);
BW = im2bw(out, T);
L = bwlabel(BW, 4);

[L,n]=bwlabel(BW,8);            % length filtering
for i=1:n
    [r, c]=find(L==i);
     rc = [r c];
    [rn, r2]=size(r);
%     if rn<20                      % the threshold of length filtering 
%         L(r,c)=0;               % remove edge contours with less 100 pixels
%     end
end

imshow(L);

