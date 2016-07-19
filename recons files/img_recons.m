function img = img_recons(patches,m,n,patchsize)
%%given a patch dictionary reconstructs the test image of size mxn

numel = size(patches,1);

l= (patchsize-1)/2;
assert(m*n == numel,'recheck the image vector');
img = zeros(m,n);
for k = 1:numel
    temp_patch = reshape(patches(k,:),[patchsize,patchsize]);
    if (mod(k,n) ==0)
        j  = n;
    else
        j= mod(k,n);
    end
    i = ceil(k/n);
    img(i,j) = temp_patch(l+1,l+1);
end

end
