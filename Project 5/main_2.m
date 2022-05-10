clear;
I=double(imread('discs8.bmp'))/255;     % read the test image
T=double(imread('target.bmp'))/255;     % read the target image
[X, Y]=size(I); Z=zeros(X,Y);
Kmax=100;                               % the number of random walks

Oxy=round(rand(1,2)*X)+1;               % an initial position

int_pos = {};
for w = 1:8
L1=likelihood(I,T,Oxy,1);               % the initial likelihood    
Io=drawcircle(I,Oxy,1);                 % locate the object
figure(1),imshow(Io);
Imframe(1:X,1:Y,1)=Io; Imframe(1:X,1:Y,2)=Io; Imframe(1:X,1:Y,3)=Io;
videoseg(1)=im2frame(Imframe);          % make the first frame

for i=1:Kmax
    Dxy=Oxy+round(randn(1,2)*20);       % random walk
    Dxy=clip(Dxy,1,X);                  % make sure the position in the image
    L2=likelihood(I,T,Dxy,1);           % evaluate the likelihood
    v=min(1,L2/L1);                     % compute the acceptance ratio
    u=rand;                             % draw a sample uniformly in [0 1]
    if v>u
        Oxy=Dxy;     L1=L2;             % accept the move
        Io=drawcircle(I,Oxy,1);         % draw the new position
    end
    figure(1),imshow(Io);
    fprintf("w: %d,  i: %d \n",w, i);
%     Imframe(1:X,1:Y,1)=Io; Imframe(1:X,1:Y,2)=Io; Imframe(1:X,1:Y,3)=Io;
%     videoseg(i+1)=im2frame(Imframe); 
end
%movie2avi(videoseg(1:(Kmax+1)),'MCMC1.avi','FPS',10,'COMPRESSION','None');
int_pos{w,1} = Dxy;
I = Io;
Oxy = Dxy;

figure();
imshow(Io);

end

k_max = 40;
N = 20;

lamda = 8;
k = lamda+2;
Oxy1 = round(rand(k,2)*X)+1; 

Io = drawcircle(I, Oxy1, k); % drawing the circles
figure();
imshow(Io);

pos_final = {};
K_final = zeros(1,N);

for t =1:N
    lik = likelihood(I, T, int_pos, k);
    post = lik * poisspdf(k, lamda);
    
    % Jumping
    a = rand(1);
    if (a < 0.33) && (k > 1)
        dk = -1;
    elseif (a < 0.66) && (k < k_max)
        dk = 1;
    else
        dk = 0;
    end

    pos_temp = jump(t, path_im, I, T, int_pos, k, dk, lik);
    like_temp = likelihood(I, T, pos_temp, k + dk);
    post_p = like_temp * poisspdf(k + dk, lamda);

    v = min(1, post_p / post);
    u = rand(1);
    if v > u
        int_pos = pos_temp;
        k = k+dk;
    end

    pos_final{t,1} = int_pos;
    K_final(t,1) = k;
end

M = 5;
pos_final = pos_final(1:M);
K_final = K_final(1:M);    
K_star = int(mean(K_final));

function pos_p = jump(I, T, theta, k, dk)
    k_p = k + dk;
    pos_init ={};
    pos_init = theta(1:k_p);
    
    if dk == 1
        init_new_sample = round(rand(1,2)*X)+1;
        pos_init = init_new_sample;
    end
    L1=likelihood(I,T,Oxy,1);               % the initial likelihood    
    pos_p = diffusion(I, T, theta, k, L1);
end



    
    
