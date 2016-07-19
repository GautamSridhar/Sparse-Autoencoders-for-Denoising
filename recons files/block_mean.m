function m_p = block_mean(patches,sim_wind)
%%finds the blockwise average of each example with size as patchsize taking
%%blocksize as sim_wind

m_p = zeros(1,size(patches,2));
patchsize = sqrt(size(patches,1));

for i = 1:size(patches,2)
patch = patches(:,i);
img_temp =reshape(patch,[patchsize,patchsize]);

ver = patchsize/sim_wind;
hor = ver;

patch_vec = zeros(sim_wind^2,1);

for c = 1:sim_wind:patchsize-sim_wind+1
 for r= 1:sim_wind:patchsize-sim_wind+1
   temp_patch = img_temp(r:r+sim_wind-1,c:c+sim_wind-1);
   patch_vec = patch_vec + reshape(temp_patch,[sim_wind^2,1]);
end
end

patch_vec =patch_vec/(hor*ver);

patch =reshape(patch_vec,[sim_wind,sim_wind]);
m_p(1,i) = patch((sim_wind-1)/2,(sim_wind-1)/2);


end
end
