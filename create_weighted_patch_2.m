function patch_out = create_weighted_patch_2(patch_d,s_dict,sim_wind,patchsize)
%% takes a patch vector as input along with the search dictionary returns the reconstructed patch vector

h=1; %filter parameter

numel = size(s_dict,1);
weight = zeros(numel,1);

for k = 1:numel
    dist_int = sqrt(sum((patch_d -s_dict(k)).^2));
    weight(k) = exp(-dist_int/h);
end

%weight = weight / sum(weight);
s_dict = horzcat(s_dict,weight);
s_dict = sortrows(s_dict,-size(s_dict,2));
patch_out = s_dict(2:(patchsize/sim_wind)^2,1:(size(s_dict,2)-1));
end