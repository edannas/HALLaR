clear all;
close all;
hold on;
axis equal;

x = 0:1:40;
y = 0:1:50;

set(gcf,'color','w');
[X,Y] = meshgrid(x,y);
Z = sqrt(X.*Y);

% Plot the lower half of the cone
surf(X,Y,-Z,'edgecolor','none');
surf(X,Y,Z,'edgecolor','none');
shading interp

% Define grid points
[X, Y] = meshgrid(linspace(0, 40, 50), linspace(0, 50, 50));

% Define the plane x + z = 1
Z_plane = 1 - X;

% Plot the plane
surf(X, Y, Z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'red');

% Plot the feasible region below the plane
% Create a mask for the feasible region
% mask = (X + Z_plane >= 0);  % Only plot below the plane

% Set values outside the feasible region to NaN
% Z_plane(~mask) = NaN;

% Plot the feasible region (intersection between cone and plane)
surf(X, Y, Z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'blue');

xlabel('x');
ylabel('y');
zlabel('z');
title('Trace constrained cone of PSD matrices');
axis equal;

view([62,24]);
xlim([0 50])
zlim([-30 30])

