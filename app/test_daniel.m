b = im2double(imread(fullfile('input_images','manWithHat.tiff')));

if size(b,3) == 3
    b = rgb2gray(b);
end

psf = fspecial('gaussian', 9, 2);

params.gamma = 0.01;
params.maxiter = 500;
params.tol = 1e-3;

x = deblur('l1','CP',[],psf,b,params);

imshow(x,[])