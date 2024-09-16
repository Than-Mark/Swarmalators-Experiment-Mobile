function snap(X, Y, Theta, dt, n)




sz = 20; % Size of the dots
T = round(n/dt); % Scaling

col = Theta(T,:); % Vector of colors










scatter(X(T, :), Y(T, :), sz, col, 'filled')
 
ax_min = min(min(X(:)), min(Y(:)));
ax_max = max(max(X(:)), max(Y(:)));

%axis([-2 2 -2 2]);
axis([ax_min-0.1 ax_max+0.1 ax_min-0.1 ax_max+0.1])
axis square

%% Uncomment for exporting animations

colormap hsv
h = colorbar;
h.Ticks = [0 pi 2*pi];
h.TickLabelInterpreter = 'latex';
h.TickLabels = ["0", "$\pi$", "$2\pi$"];
caxis([0 2*pi])

end