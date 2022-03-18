function [d3d, d2d] = data_loading(data_3d, data_2d)

d3d_tmp = importdata(data_3d);
d3d = zeros(size(d3d_tmp, 1), 4);
d3d(:, 1:3) = d3d_tmp;
d3d(:, 4) = 1;

d2d = importdata(data_2d);
end