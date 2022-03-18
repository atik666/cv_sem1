function [M, l, r3, u0, v0, alpha, beta] = param_estimation(d3d, d2d)
num_points = size(d3d, 1);
P = zeros(2*num_points, 12);

for i = 1:num_points
    P((i-1)*2+1, 1:4) = d3d(i, :);
    P(i*2, 5:8) = d3d(i, :);
    P((i-1)*2+1, 9:12) = -d2d(i, 1) * d3d(i, :);
    P(i*2, 9:12) = -d2d(i, 2) * d3d(i, :);
end
% Using Vector Decomposition
[V,D] = eig(P'*P);
e = zeros(1, size(D,1));
for i = 1:size(D,1)
    e(i) = D(i,i);
end

[~, i_min] = min(e);
m = V(:, i_min);
M = [m(1:4)';m(5:8)';m(9:12)'];

a1 = M(1, 1:3);
a2 = M(2, 1:3);
a3 = M(3, 1:3);

l = 1 / norm(a3, 1);
r3 = l*a3;
u0 = l*l*dot(a1, a3);
v0 = l*l*dot(a2, a3);
alpha = l*l*norm(cross(a1, a3),1);
beta = l*l*norm(cross(a2, a3), 1);
end