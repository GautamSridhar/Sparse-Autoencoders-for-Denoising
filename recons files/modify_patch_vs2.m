function out = modify_patch_vs2(patch, search_wind,sim_wind,m,n,patchsize)

%create dictionary for searching

patch_temp = search_wind(1:sim_wind,1:sim_wind);
search_dictionary = reshape(patch_temp,[1,sim_wind^2]);

for i = 1: m
    for j = 1: n
        
      patch_temp = search_wind(i:i+sim_wind-1,j:j+sim_wind-1);
      patch_flat = reshape(patch_temp,[1,sim_wind^2]);
      search_dictionary = vertcat(search_dictionary,patch_flat);
   
    end
end
search_dictionary = (search_dictionary(2:end,:));

patch_dictionary_mod = create_weighted_patch_2(reshape(patch,[1,sim_wind^2]),search_dictionary,sim_wind,patchsize);

patch_dictionary_mod = vertcat(reshape(patch,[1 sim_wind^2]),patch_dictionary_mod);
out = patch_reconst(patch_dictionary_mod,patchsize,sim_wind);

end