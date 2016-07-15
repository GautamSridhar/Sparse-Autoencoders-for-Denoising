patchsize =21;
testData = imread('cameraman.tif');
figure; subplot(3,1,1);
imshow(testData,[])
% testData = imnoise(testData,'gaussian',0,0.01);
% testData =imnoise(testData,'poisson');
% testData =imnoise(testData,'speckle',0.2);
testData = imnoise(testData,'gaussian');
testData =imnoise(testData,'poisson');
testData =imnoise(testData,'speckle');

[m,n] = size(testData);

subplot(3,1,2);
imshow(testData,[])
test_patches = test_patch_create(testData,patchsize);

nn_testing = my_nnff(nn,test_patches);
output = nn_testing.a{3};
%generate output image
out_img = img_recons(output,m,n,patchsize);
subplot(3,1,3);imshow(out_img,[])
savefig('testing example.png','png')

imwrite(testData,'noisy.tif');
imwrite(out_img,'output.tif')