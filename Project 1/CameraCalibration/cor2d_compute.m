function  d2d = cor2d_compute(d3d, M)
d2d_output = zeros(size(d3d, 1), 3);
for i = 1:size(d3d, 1)
    d2d_output(i, :) = M*d3d(i, :)'/(M(3, :) * d3d(i, :)');
end
d2d = d2d_output(:, 1:2);
end