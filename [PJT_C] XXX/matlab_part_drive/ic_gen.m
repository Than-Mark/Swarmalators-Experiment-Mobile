function [x0, y0, theta0,omega_n] = ic_gen(N)
% This function allows to generate uniform initial conditions for the
% swarmalators model between a specified range.
% Inputs:
%         N : Number of particles;
%
% Outputs:
%         x0:
%         y0:
%         theta0:
%

pmax = 1;                       pmin = -1;
thmax = pi;                     thmin = -pi;


x0 = (pmax-pmin).*rand(N,1) + pmin;
y0 = (pmax-pmin).*rand(N,1) + pmin;
theta0 = (thmax-thmin).*rand(N,1) + thmin;
% 生成均匀分布的自然频率
omega_n = linspace(thmin, thmax, N);
% 随机排列自然频率顺序
omega_n = omega_n(randperm(N));%该矩阵为N个振子的自然频率矩阵
end