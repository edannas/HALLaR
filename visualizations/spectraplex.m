clear all;
close all;
hold on;
axis equal;

xlim = 100;
ylim = 100;
zlim = 100;

x = 0:.02:xlim;  % Adjusted x-axis limits
y = 0:.02:ylim; % Adjusted y-axis limits

set(gcf,'color','w');
[X,Y] = meshgrid(x,y);
Z = sqrt(X.*Y);

axis([0,xlim,0,ylim,-zlim,zlim]);  % Adjusted axis limits

% Plot the upper half of the cone
surf(X,Y,Z,'edgecolor','none');
shading interp

% Plot the lower half of the cone
surf(X,Y,-Z,'edgecolor','none');
shading interp

% Define grid points
[X, Y] = meshgrid(linspace(0, xlim, 50), linspace(0, ylim, 50));  % Adjusted grid points

% Define the plane x + z = 1
Z_plane = 1 - X;

% Plot the plane
surf(X, Y, Z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'red');

% Plot the feasible region below the plane
% Create a mask for the feasible region
mask = (X + Z_plane <= 1);

% Set values outside the feasible region to NaN
Z_plane(~mask) = NaN;

% Plot the feasible region
surf(X, Y, Z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'blue');

xlabel('x');
ylabel('y');
zlabel('z');
title('Feasible Region for x + z <= 1');
axis equal;

view([62,24]);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
