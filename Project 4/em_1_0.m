clear;close all;clc;

% ----------------initilization---------------%
img_org=imread('mosaicB.bmp');
Img=double(img_org);

K = 4;    % the number of cluster

% initialize centers with k-means

[height,width]=size(Img);

Img1 = imnoise(Img,'gaussian',0.001);

% gauss_f = [ 1/256 4/256 6/256 4/256 1/256;
%             4/256 16/256 24/256 16/256 4/256;
%             6/256 24/256 36/256 24/256 6/256;
%             4/256 16/256 24/256 16/256 4/256;
%             1/256 4/256 6/256 4/256 1/256]      ;%gaussian filter
%         
Img = normalize(Img, 'range');
% seg = conv2(Img,gauss_f, 'same');
% 
% y = seg(:);
y = Img(:);
n=size(y,1);

Img_out=zeros(height,width);
iterNum=50;

[IDX,C] = kmeans(y,K,'Maxiter',100,'Display','off');

%%
c_sum = zeros(K,1);
cluster = zeros(n,K);
V = zeros(K,1);  % V: the mean vector for each Gaussian component
dim = 1;
Sigma = zeros(dim,dim,K);    % the standard devision
for i = 1:n
    c_sum(IDX(i)) = c_sum(IDX(i)) + 1;
    cluster(c_sum(IDX(i)),IDX(i),:) = y(i,:);
end

for i = 1:K
    V(i,:) = sum(cluster(1:c_sum(i),i,:))./c_sum(i);
    clu=cluster(1:c_sum(i),i);
    Sigma(i) = cov(clu);
end

beita = 0.5;
laimda = [0 0.0185 0.0414 0.0185 0;
    0.0185 0.0852 0.0865 0.0852 0.0185;
    0.0414 0.0865 0 0.0865 0.0414;
    0.0185 0.0852 0.0865 0.0852 0.0185;
    0 0.0185 0.0414 0.0185 0];

PL = ones(height,width,K)/K;  % Pixel label priors
PI = zeros(height,width,K);   % the posterior
S = zeros(height,width,K);    % the auxiliary set of distributions s_i
Q = zeros(height,width,K);    % an arbitrary class distribution for a pixel i

%------------------Iteration-----------------%

for iter = 1:iterNum
    
    % E-step：Compute posterior probabilities
    temp=reshape(PL,n,K);
    for k = 1:K
        
        temp_(:,k) = temp(:,k).*((2*pi)^(-0.5*dim)*det(Sigma(:,:,k)+eps)^(-0.5)* ...
            exp(-0.5*sum((y - repmat(V(k,:),n,1))*inv(Sigma(:,:,k)).*(y - repmat(V(k,:),n,1)),2)));
      
    end
    PI_sum = sum(temp_,2);
    for k = 1:K
        temp_(:,k) = temp_(:,k)./(PI_sum + eps);
    end
    PI=reshape(temp_,height,width,K);
    
    % Compute {s_i} and {q_i}, then normalize
    for k = 1:K
        S(:,:,k) = PL(:,:,k).*conv2(PL(:,:,k),laimda,'same');
        Q(:,:,k) = PI(:,:,k).*conv2(PI(:,:,k),laimda,'same');
    end
    S_sum = sum(S,3); Q_sum = sum(Q,3);
    for k = 1:K
        S(:,:,k) = S(:,:,k)./(S_sum + eps);
        S_N(:,:,k) = conv2(S(:,:,k),laimda,'same');
        Q(:,:,k) = Q(:,:,k)./(Q_sum + eps);
        Q_N(:,:,k) = conv2(Q(:,:,k),laimda,'same');
    end
    
    % M-step: Update the parameter vector (V & Sigma) and pixel label priors
    temp2=reshape(Q,n,K)+reshape(Q_N,n,K);
    
    for k = 1:K
        temp3=repmat(temp2(:,k),1,dim);
        V(k,:)=sum(temp3.*y,1)/(sum(temp2(:,k))+eps);
        Sigma(:,:,k)= y'*(temp3.*y)/(sum(temp2(:,k)+eps))- V(k,:)'* V(k,:);
        PL(:,:,k) = (0.5*(S(:,:,k)+S_N(:,:,k))+beita*(Q(:,:,k) + Q_N(:,:,k)))/(1+2*beita);
    end
    
    [e_max,N_max] = max(PI,[], 3);
    Img_out = reshape(V(N_max,:),height,width,dim);

     % record gif
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if iter == 1
        imwrite(I,map,'test_gray.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test_gray.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    iter = iter + 1;
    
    iterNums=['segmentation: ',num2str(iter), ' iterations'];
    subplot(1,2,1),imshow(uint8(Img),[]),title('original')
    subplot(1,2,2),imshow(uint8(Img_out),[]),title(iterNums); colormap(gray);
    pause(0.1)
end

plot(cell2mat(Per))
xlabel('iteration')
ylabel('accuracy')

