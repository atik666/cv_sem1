itr=30;
rate = .2;
filter = [0 0.0185 0.0414 0.0185 0;
    0.0185 0.0852 0.0865 0.0852 0.0185;
    0.0414 0.0865 0 0.0865 0.0414;
    0.0185 0.0852 0.0865 0.0852 0.0185;
    0 0.0185 0.0414 0.0185 0];

pixel_label = ones(h,w,K)/K;  % priors
posterior = zeros(h,w,K);   % posterior
aug_s = zeros(h,w,K);   
pixel = zeros(h,w,K);  
Truth = imread('mapA.bmp');
Truth = double(Truth);

acc = {};
for iter = 1:itr
    % E-step：Compute posterior probabilities
    temp=reshape(pixel_label,n_size,K);
    for k = 1:K  
       temp_(:,k) = temp(:,k).*((2*pi)^(-0.5*d)*det(sigma(:,:,k)+eps)^(-0.5)* ...
            exp(-0.5*sum((array - repmat(v_mean(k,:),n_size,1))*inv(sigma(:,:,k)).*(array - repmat(v_mean(k,:),n_size,1)),2)));   
    end
    PI_sum = sum(temp_,2);
    for k = 1:K
        temp_(:,k) = temp_(:,k)./(PI_sum + eps);
    end
    posterior=reshape(temp_,h,w,K);
    for k = 1:K
        aug_s(:,:,k) = pixel_label(:,:,k).*conv2(pixel_label(:,:,k),filter,'same');
        pixel(:,:,k) = posterior(:,:,k).*conv2(posterior(:,:,k),filter,'same');
    end
    S_sum = sum(aug_s,3); Q_sum = sum(pixel,3);
    for k = 1:K
        aug_s(:,:,k) = aug_s(:,:,k)./(S_sum + eps);
        S_N(:,:,k) = conv2(aug_s(:,:,k),filter,'same');
        pixel(:,:,k) = pixel(:,:,k)./(Q_sum + eps);
        Q_N(:,:,k) = conv2(pixel(:,:,k),filter,'same');
    end
    % M-step: Update the parameter vector (V & Sigma) and pixel label priors
    ch2=reshape(pixel,n_size,K)+reshape(Q_N,n_size,K);
    for k = 1:K
        ch3=repmat(ch2(:,k),1,d);
        v_mean(k,:)=sum(ch3.*array,1)/(sum(ch2(:,k))+eps);
        sigma(:,:,k)= array'*(ch3.*array)/(sum(ch2(:,k)+eps))- v_mean(k,:)'* v_mean(k,:);
        pixel_label(:,:,k) = (0.5*(aug_s(:,:,k)+S_N(:,:,k))+rate*(pixel(:,:,k) + Q_N(:,:,k)))/(1+2*rate);
    end   
    [e_max,N_max] = max(posterior,[], 3);
    out_img = reshape(v_mean(N_max,:),h,w,d);
    % record gif
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    if iter == 1
        imwrite(I,map,'test_gray.gif','gif','Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(I,map,'test_gray.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    iterNums=['Output: ','number of iter:',num2str(iter)];
    subplot(1,2,1),imshow(uint8(img),[]),title('Actual image')
    subplot(1,2,2),imshow(uint8(out_img),[]),title(iterNums); colormap(gray);
    pause(0.1)
    acc{1,iter} = accuracy(Truth,out_img);
end

figure()
plot(cell2mat(acc))
xlabel('iteration')
ylabel('accuracy')
