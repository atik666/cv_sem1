clear;clc;
% The code to read 12*5 images (90x60) from 12 directories 
for i=1:12
    for j=0:4
        file=sprintf('%s',int2str(i),'/',int2str(i),'_',int2str(j),'.bmp');
        face=imread(file);
        for k=1:90  
            x=(k-1)*60+1;	
            y=k*60;
            A((i-1)*5+1+j,x:y)=double(face(k,:));	% reshape the image into a vector
        end
    end
end

training_tensor = A;

meanFace = sum(training_tensor)/60;

normalized_training_tensor = training_tensor - meanFace;

cov_matrix = cov(normalized_training_tensor');
cov_matrix = cov_matrix/60;

[eigenvectors,D] = eig(cov_matrix);

D = diag(D);
eigenvalues = sort(D, 'descend');

var_comp_sum = cumsum(eigenvalues/sum(eigenvalues));

reduced_data = eigenvectors(:, 1:50);

proj_data = training_tensor'*reduced_data;
proj_data = proj_data';

for j=1:60
    for i = 1:size(proj_data,1)
        w(j,i) = dot(proj_data(i,:), normalized_training_tensor(j,:));
    end
end

% The code to read 12*5 images (90x60) from 12 directories 
for i=1:12
    for j=0:4
        file=sprintf('%s',int2str(i),'/',int2str(i),'_',int2str(j),'.bmp');
        face=imread(file);
        for k=1:90  
            x=(k-1)*60+1;	
            y=k*60;
            B((i-1)*5+1+j,x:y)=double(face(k,:));	% reshape the image into a vector
        end
    end
end

out = {};
for j = 1:60
    b = B(j,:);
    norm_vec = b - mean(b);

    for i = 1:size(proj_data,1)
        w_un(i,1) = dot(proj_data(i,:), norm_vec(:,:));
    end

    diff = w - w_un';
    diffn =sum(diff);

    [M,I] = min(diffn);
    out{j,1} = I;
end

