clear; clc;
img = imread('./mosaicA.bmp');

nscale          = 4;     
norient         = 6;     
minWaveLength   = 3;      
mult            = 2;     
sigmaOnf        = 0.65; 
dThetaOnSigma   = 1.5;

EO = gaborconvolve(img, nscale, norient, minWaveLength, mult, ...
			    sigmaOnf, dThetaOnSigma);
            
EO_ = reshape(EO,[],1);

EO_abs = {};
gauss_f = [ 1/256 4/256 6/256 4/256 1/256;
            4/256 16/256 24/256 16/256 4/256;
            6/256 24/256 36/256 24/256 6/256;
            4/256 16/256 24/256 16/256 4/256;
            1/256 4/256 6/256 4/256 1/256]      ;%gaussian filter
        
for i = 1:length(EO_)
    %EO_abs{i,1} = abs(EO_{i,1});
    img_ = abs(EO_{i,1});
    seg = conv2(img_,gauss_f, 'same');
    EO_abs{i,1} = seg;
end

for i = 1:length(EO_abs)
    shape = reshape(EO_abs{i,1},[],1);
    varinece{i,1} = var(shape);
    avr{i,1} = mean(shape);
    skew{i,1} = skewness(shape);
    kurt{i,1} = kurtosis(shape);
end

feat = cell2mat([varinece, avr, skew, kurt]);

X = normalize(feat,'range');

[idx,C,sumd,D] = kmeans(X, 4);

%%
X_ = {};
for i = 1:size(X,2)
    a = X(:,i) - C(i,:);
    
    aa = min(a,[], 2);
    X_{1,i} = aa;
end
X_ = cell2mat(X_);

%% The Elbow Method

WCSS=[];
for k=1:10
    sumd=0;
    [idx,C,sumd]=kmeans(X,k);
    WCSS(k)=sum(sumd);
end

plot(1:10,WCSS);

%% K-means Clustering

[idx,c]=kmeans(X,3); % idx tells us which particular clustar the value belongs to
                        % C gives the center point of the cluster

figure,

gscatter(X(:,1),X(:,2),idx);
hold on
for i=1:3
    scatter(C(i,1),C(i,2),96,'black','filled');
end
legend({'Cluster 1','Cluster 2','Cluster 3'});
xlabel('PC1');
ylabel('PC2');
hold off