clc, clearvars, close all
tic


%p=parpool('local',6);
%% 定义初始值：
% Model
N=200; 
%初始
[x0, y0, theta0,omega_n] = ic_gen(N);

Fc=4;

J=0.1;
k=1;
omega=0.5*pi;
RS=zeros();
ome=(0:0.1*omega:250*pi)';%振荡源相位
% Computation time
t0= 0;
dt= 0.1;
tf= 200;
% 定义驱动比例
partly_start = 0.1;
partly_end =0.99;
partly_step = 0.01;

% 循环遍历不同的 partly 值
for partly = partly_start:partly_step:partly_end
% 外驱动对百分50的粒子实施外力
A = zeros(1, N); % 创建全0矩阵
B=  zeros(1, N);
% 随机选择索引位置
indices = randperm(N);
half_size = N*partly;
% 将前一半位置设置为1
A(indices(1:half_size)) = 1;
B(indices(1:end))=1;
%设置Fij
F_p=Fc.*A;%部分驱动
F=Fc.*B;%完全驱动

%部分驱动

[tv, X_p, Y_p, Theta_p]= swarma2D(x0, y0, theta0,omega_n, N, J, k, F_p, omega, t0, dt, tf);

%完全驱动
[tv, X, Y, Theta]= swarma2D(x0, y0, theta0,omega_n, N, J, k, F, omega, t0, dt, tf);
Theta_no=Theta_p(1:10*tf,indices(half_size+1:end));
Theta_pa=Theta_p(1:10*tf,indices(1:half_size));
X_pa=X_p(:,indices(1:half_size));X_no=X_p(:,indices(half_size+1:end));
Y_pa=Y_p(:,indices(1:half_size));Y_no=Y_p(:,indices(half_size+1:end));
foldername = '1';
if ~isfolder(foldername)
    mkdir(foldername);
end % 文件夹名称
filename = sprintf('节点比例%.2f J=%.2f K=%.2f.avi', partly, J, k);
filePath = fullfile(foldername, filename);
v = VideoWriter(filePath);
    open(v);
    an = [];
    for n = dt:0.1:tf
        subplot(1,2,1);
       snap(X, Y, Theta, dt, n)
      subplot(1,2,2);
        snap(X_pa, Y_pa, Theta_pa, dt, n)
        hold on;
        snap1(X_no, Y_no, Theta_no, dt, n)
        hold off;
       dim = [0.1 0.1 0.1 0.1];
       %str=sprintf('$F = $%.2f,$t= $%.2f',F,n); 
       str=sprintf('$node=$%.2f, $k=$%.2f, $F = $%.2f,$J = $%.2f,$t= $%.2f',partly,k,Fc,J,n);%if No floting varibale number, use %d
       delete(an)
       an = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex');
       frame = getframe(gcf);
       drawnow;
       %frame = getframe(gcf);
       writeVideo(v, frame);
       delete(an)
    end
    close(v)
end




function [V]=orderV(X,Y,dt)
 
    V=mean(mean(sqrt(diff(X).^2+diff(Y).^2)/dt),2);  
end    
function [T]=orderT(X,Y,Theta,tf)
    phi=atan2(Y,X);
    T_plus=abs(mean(exp(1j*mean(phi(5*tf:10*tf,:)+Theta(5*tf:10*tf,:),1))));
    T_minus=abs(mean(exp(1j*mean(phi(5*tf:10*tf,:)-Theta(5*tf:10*tf,:),1))));
    T =max(T_plus,T_minus);
end    
function [Va]=orderVa(X,Y,Theta,dt)
    delta=sqrt((X-X0)^2)+(Y-Y0)^2;
    p=find(delta>5);
    delta(p)=2*pi-delta(p);
    Va=mean(mean(sqrt(diff(X).^2+diff(Y).^2+delta.^2)/dt),2);  
end
function [Vp]= orderVp(vx,vy,tf)

    phi=atan2(vy,vx);
    
    Vp=mean(abs(mean((exp(1j*phi(9*tf:10*tf,:))),2)),1);


end

function [Q]=orderphi(X,Y,tf)
  phi=atan2(Y,X);
  Q=mean(abs(mean(exp(1j*phi(9*tf:10*tf,:)),2)),1);
   

    
end

