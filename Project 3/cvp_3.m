clc;
clear;
close all;


% part 2
%I = imread("D1.bmp");
X = 640; Y = 640;		% Size of the texture image
Ns=4; No=6;			% Numbers of scale and orientation in Gabor	


% using loop read multiple images
% using gabor filter
NOI = 59;
NOF = 3;
texture = cell(NOI,1);
E = cell(NOI,1);
Mi = cell(NOI,1);
for i=1:NOI
    texture{i} = imread(sprintf('Samples/D%d.bmp',i));
    E0 = gaborconvolve(texture{i},Ns,No,3,2,0.65,1.5);
    E{i} = E0;
end

%extracting feautre vectors
feautre = cell(NOI,1);

for k =1:NOI
    for i =1:Ns
        for j =1:No

            Mi{k}{i,j} = abs(E{k}{i,j});
             I(1,1) = mean(Mi{k}{i,j},'all');
             I(2,1) = var(Mi{k}{i,j},0,'all');
             I(3,1) = skewness(Mi{k}{i,j},0,'all');
             %I(4,1) = kurtosis(Mi{k}{i,j},0,'all');
             feautre{k}{i,j} = I;

        end
    end
end


p1={};close all;
max_mean = [];
min_mean = [];
max_var = [];
min_var = [];
max_skew =[];
min_skew =[];
max_kurt =[];
min_kurt =[];

for k = 1:NOI 
      p1{k} = reshape(cell2mat(feautre{k}(:)),NOF,[]); % 2 
      max_mean(k) = max(p1{k}(1,:));
      min_mean(k) = min(p1{k}(1,:));
      max_var(k) = max(p1{k}(2,:));
      min_var(k) = min(p1{k}(2,:));
     max_skew(k) = max(p1{k}(3,:));
     min_skew(k) = min(p1{k}(3,:));
     %max_kurt(k) = max(p1{k}(4,:));
     %min_kurt(k) = min(p1{k}(4,:));
end

fmax_mean = max(max_mean);
fmin_mean = min(min_mean);
fmax_var = max(max_var);
fmin_var = min(min_var);
fmax_skew = max(max_skew);
fmin_skew = min(min_skew);
%fmax_kurt = max(max_kurt);
%fmin_kurt = min(min_kurt);

%normalizing feautre vector
NF = cell(NOI,1);


for k =1:NOI
    for i =1:Ns
        for j =1:No

             N(1,1) = (feautre{k}{i,j}(1,1)-fmin_mean)/(fmax_mean - fmin_mean);
             N(2,1) = (feautre{k}{i,j}(2,1)-fmin_var)/(fmax_var - fmin_var);
             N(3,1) = (feautre{k}{i,j}(3,1)-fmin_skew)/(fmax_skew - fmin_skew);
             %N(4,1) = (feautre{k}{i,j}(4,1)-fmin_kurt)/(fmax_kurt - fmin_kurt);
             NF{k}{i,j} = N;

        end
    end
end
% creating filter bank
feautre_lib = cell(NOI,1);
for k =1:NOI
    feautre_lib{k} = reshape(cell2mat(NF{k}'),[],1);
end

%NF is normalized feautre vector and it is also called as signatures  of textures 

%dividing given texture image into 100 blocks

nrows = 10; 
ncols = 10;

image_blocks = cell(NOI,1);
for i = 1:NOI
    subimages = mat2cell(texture{i,1}, ones(1, nrows) * size(texture{i,1}, 1)/nrows, ones(1, ncols) * size(texture{i,1}, 2)/ncols, 1);
    image_blocks{i} = reshape(subimages, 100, 1);
end



% we should extract feautre vectors for each image block and should
% normalize the feautre vector of each image block

% applying gabor filter
new_E = cell(NOI,1);
for j = 1: 100
for i=1:NOI 
    E1 = gaborconvolve(image_blocks{i}{j},Ns,No,3,2,0.65,1.5);
    new_E{i}{j} = E1;
   
end
end
% getting absolute value of each element
% extract feautre vector for each image block
Mi_new = cell(NOI,1);
feautre1 = cell(NOI,1);

for k =1:NOI
   for l = 1:100
    for i =1:Ns
        for j =1:No

           Mi_new{k}{l}{i,j}=abs(new_E{k}{l}{i,j});  % Create the magnitude for each Gabor channel
           I1(1,1) = mean(Mi_new{k}{l}{i,j},'all');
           I1(2,1) = var(Mi_new{k}{l}{i,j},0,'all');
           I1(3,1) = skewness(Mi_new{k}{l}{i,j},0,'all');
           %I1(4,1) = kurtosis(Mi_new{k}{l}{i,j},0,'all');
           feautre1{k}{l}{i,j} = I1;

        end
    end
  end
end


%normalizing feautre vector and this should range betweem 0 to 1 so do some
%clipping here

U = cell(NOI,1);

for k =1:NOI
    for l = 1:100
      for i =1:Ns
        for j =1:No

              N1(1,1) = (feautre1{k}{l}{i,j}(1,1)-fmin_mean)/(fmax_mean - fmin_mean);
              N1(2,1) = (feautre1{k}{l}{i,j}(2,1)-fmin_var)/(fmax_var - fmin_var);
              N1(3,1) = (feautre1{k}{l}{i,j}(3,1)-fmin_skew)/(fmax_skew - fmin_skew);
             %N1(4,1) = (feautre1{k}{l}{i,j}(4,1)-fmin_kurt)/(fmax_kurt - fmin_kurt);
             U{k}{l}{i,j} = N1;
             
        end
       end
    end
end




for k=1:NOI
    for l=1:100
        for i = 1:Ns
            for j =1:No
               for a=1:NOF

                    if (U{k}{l}{i,j}(a,1) < 0)
                        U{k}{l}{i,j}(a,1) = 0;
                   elseif (U{k}{l}{i,j}(a,1) > 1)
                          U{k}{l}{i,j}(a,1) = 1;
                    end
              end
           end
        end
    end
end
       
 % create new feautre1_bank
 feautre1_lib  = cell(NOI,1);

for k = 1:NOI
    for l = 1:100

          feautre1_lib{k}{l} = reshape(cell2mat(U{k}{l}'),[],1);

    end
end

%U is normalized feautre vector for 100 image_blocks and it is also called as signatures  of textures 
% now calculate least eculidean distance between the feautre vectors
% compare between NF and U


ecu_distance = cell(NOI,1);
 min_distance = cell(NOI,1);
index = cell(NOI,1);


for k =1:NOI
    for l = 1:100
        for b= 1:NOI

        ecu_distance{k}{l}{b} = norm(feautre_lib{b} - feautre1_lib{k}{l});
        end

    end
end


 for  k = 1:NOI
     for l =1:100
         [min_distance{k}{l},index{k}{l}] = min(cell2mat(ecu_distance{k}{l}));
     end

 end


% texture classification
% find shortest distance and that particular index


correct_classify = 0;
final_classify = zeros(NOI,1);
for k =1:NOI
    for l =1:100
       
                if (index{k}{l} == (k))

                    correct_classify = correct_classify + 1;
               end
              
     end
    final_classify(k) = correct_classify;
   correct_classify = 0;
end

% now calculate pcc percentage of correct classification 
pcc = zeros(NOI,1);
for k =1:NOI
    pcc(k) = (final_classify(k))/(100); 
end

avg_pcc = (sum(pcc)/NOI)*100

       
   