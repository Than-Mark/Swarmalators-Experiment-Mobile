function y= eq_def(t, y, N, k, omega, J, F,omega_n)
f= zeros(1, 3*N);

for n= 1: N     % Index "i"
    A= 0;
    B= 0;
    C= 0;
    
    for m= 1: N     % Index "j"
        
        if n~=m
            
            a1= y(m)- y(n);             % x
            a2= y(N+ m)- y(N+ n);       % y
            a3= y(2*N+ m)- y(2*N+ n);   % theta
            
            b1= a1^2+ a2^2;             % Pre-norm
            b2= 1+ J*cos(a3);
            
            A= A+ 1/N*(a1*(b2/sqrt(b1)- 1/b1)); % X
            B= B+ 1/N*(a2*(b2/sqrt(b1)- 1/b1)); % Y
            C= C+ 1/N*(sin(a3)/sqrt(b1)); % THETA with no force
            
        end
    end
phase_adjust = sign(omega_n(n)) * pi/2;
    

    f(n)= A;
    f(N+ n)= B;
    %f(2*N+ n)= k*C+ F(1,n)*cos(omega*t- y(2*N+ n))/sqrt((y(n)-R*cos(W*t))^2+ (y(N+ n)- R*sin(W*t))^2);
%    f(2*N + n) = k*C +omega_n(1,n)+ F(1, n) * cos(omega*t - y(2*N + n) + phase_adjust) / sqrt((y(n) - 0)^2 + (y(N + n) - 0)^2);%一个驱动
   f(2*N + n) = k*C +omega_n(1,n)+ F(1, n) * cos(omega*t - y(2*N + n) + phase_adjust) / sqrt((y(n) - 0.5)^2 + (y(N + n) - 0)^2)+F(1, n) * cos(omega*t - y(2*N + n) - phase_adjust) / sqrt((y(n) + 0.5)^2 + (y(N + n) - 0)^2);%两个个驱动
end
y= f';
end