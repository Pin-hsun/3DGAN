source = '/Users/ghc/Dropbox/TheSource/scripts/lightning_pix2pix/outputs/results/womac3/mcfix/descar2/Gdescarsmc_index2'

dx = 384;
m0 = x(:, 0*dx+1:1*dx);
m1 = x(:, 1*dx+1:2*dx);
d0 = x(:, 2*dx+1:3*dx);
d1 = x(:, 3*dx+1:4*dx);
u0 = x(:, 4*dx+1:5*dx);
u1 = x(:, 5*dx+1:6*dx);


imagesc(m0);axis equal;colormap('gray')
hold on;[c,h]=contour(d0.*((u0>0.2)/1), [0.1:0.1:0.2]);set(gca,'Ydir','reverse');axis equal;colorbar;set(h,'LineColor','r')
hold on;[c,h]=contour(d1.*((u1>0.2)/1), [0.1:0.1:0.2]);set(gca,'Ydir','reverse');axis equal;colorbar;set(h,'LineColor','g')