function mrp = multiresolutionPyramid(A,num_levels)
%multiresolutionPyramid(A,numlevels)
%   mrp = multiresolutionPyramid(A,numlevels) returns a multiresolution
%   pyramd from the input image, A. The output, mrp, is a 1-by-numlevels
%   cell array. The first element of mrp, mrp{1}, is the input image.
%
%   If numlevels is not specified, then it is automatically computed to
%   keep the smallest level in the pyramid at least 32-by-32.

%   Steve Eddins
%   MathWorks

A = im2double(A);

M = size(A,1);
N = size(A,2);

if nargin < 2
    lower_limit = 32;
    num_levels = min(floor(log2([M N]) - log2(lower_limit))) + 1;
else
    num_levels = min(num_levels, min(floor(log2([M N]))) + 2);
end

mrp = cell(1,num_levels);

smallest_size = [M N] / 2^(num_levels - 1);
smallest_size = ceil(smallest_size);
padded_size = smallest_size * 2^(num_levels - 1);

Ap = padarray(A,padded_size - [M N],'replicate','post');

mrp{1} = Ap;
for k = 2:num_levels
    mrp{k} = imresize(mrp{k-1},0.5,'lanczos3');
end

mrp{1} = A;
end