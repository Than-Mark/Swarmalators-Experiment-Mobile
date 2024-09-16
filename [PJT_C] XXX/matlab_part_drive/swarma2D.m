function [tv, X, Y, Theta]= swarma2D(x0, y0, theta0,omega_n, N, J, k, F, omega, t0, dt, tf)


tspan= t0: dt: tf;

y0= [x0; y0; theta0]';

[tv, yv]= ode45(@(t, y) eq_def(t, y, N, k, omega, J, F,omega_n), tspan, y0);

X= yv(:, 1: N); Y= yv(:, N+ 1:2*N); 
Theta= wrapTo2Pi(yv(:, 2*N+ 1: 3*N));
%Theta= yv(:, 2*N+ 1: 3*N);   
    
end