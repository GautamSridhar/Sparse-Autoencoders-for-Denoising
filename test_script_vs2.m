close all
clear all

patchsize = 21; %AE input patchsize
 
sim_wind = 3;   % measure of side of patch for similarity
s =(sim_wind-1)/2 ;

img = imread('cameraman.tif');
[m,n,p] = size(img);
if (p == 3)
    img = rgb2gray(img);
end

img = mat2gray(img);
img = im2double(img);

img_padded = padarray(img,[s,s],'replicate');
imshow(img_padded);

for i = 1:1
r = randi([s+1,m-s],1);
c = randi([s+1,n-s],1);

temp_patch = img_padded(r-s:r+s,c-s:c+s);
 
figure;subplot(2,1,1) 
imshow(temp_patch)

patch_mod = modify_patch_vs2(temp_patch,img_padded,sim_wind,m,n,patchsize);
subplot(2,1,2); imshow(patch_mod)
end