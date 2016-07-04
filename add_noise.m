function data_pert = add_noise(input,patchsize)
%%adds noise to each patch when an given the input dataset

data_pert = zeros(size(input));

for i = 1:size(input,2)
data_sample = reshape(input(:,i),[patchsize, patchsize]);
data_pert(:,i) =reshape(imnoise(data_sample,'gaussian'),[patchsize^2,1]); 
end