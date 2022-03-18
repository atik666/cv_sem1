function [norm_feat, img_blocks] = feat_pyramid(img, level)

    feat = {};
    norm_feat = {};
    img_blocks = {};
    num_sample = 59;
    for i = 1:num_sample
        l_pyramid = L_pyramid(img{i,1},level);

        a = l_pyramid{1,5};
        avr =  mean2(a);
        varinece = var(a,[],[1 2]);
        skew = skewness(a,[],[1 2]);
        kurt = kurtosis(a,[],[1 2]);

        feat{i,1} = [avr, varinece, skew, kurt];

        norm_feat{i,1} = normalize(feat{i,1},2);
        
        nrows = 10; 
        ncols = 10;
        subimages = mat2cell(img{i,1}, ones(1, nrows) * size(img{i,1}, 1)/nrows, ones(1, ncols) * size(img{i,1}, 2)/ncols, 1);       

        newCa = reshape(subimages, 100, 1);
        img_blocks{i,1} = newCa;
    end
    
end