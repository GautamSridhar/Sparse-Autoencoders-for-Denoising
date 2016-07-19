function data_pert = add_noise(input,patchsize)
%%adds noise to each patch when an given the input dataset

data_pert = zeros(size(input));

for i = 1:size(input,2)
data_sample = reshape(input(:,i),[patchsize, patchsize]);
G = fspecial('gaussian',[3,3],1);
data_sample = imfilter(data_sample,G,'same');
%testData = imnoise(testData,'gaussian',0,0.001);
data_pert(:,i) =reshape(imnoise(data_sample,'salt & pepper'),[patchsize^2,1]); 
end