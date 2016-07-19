function [patches,mean_p] = test_patch_create(img,patchsize)
%%given an image returns overlapping patches of size patchsize with center
%%pixel as the pixels of the original image
[m,n] = size(img);
q = (patchsize-1)/2;

%padding the image
image_padded = padarray(img,[q,q],'replicate');

%initialising the patch vector
patches = zeros(m*n,patchsize^2);
for i = 1:m
    for j = 1:n
     patches((i-1)*m+j,:) = reshape(image_padded(i:i+patchsize-1,j:j+patchsize-1),[patchsize^2 1]);
     mean_p((i-1)*m+j) = mean(reshape(image_padded(i:i+patchsize-1,j:j+patchsize-1),[patchsize^2 1]));
    end
end
