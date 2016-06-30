function patch_out = create_weighted_patch(patch_d,s_dict,a,hp,hp2)
%% takes a patch vector as input along with the search window dictionary and position of current element in the iteration and returns the weighted and reconstructed patch

h=0.1; %filter parameter
numel = size(s_dict,1);

if (rem(a,hp) == 0)
     i_1 = hp;
 else
     i_1 =rem(a,hp);
end
 j_1 = ceil(a/hp);
 
 i_1 = abs(i_1 -ceil(hp/2));
 j_1 = abs(j_1 -ceil(hp/2));
 
 weight = zeros(numel,1);

for k = 1:numel
    
    if (rem(k,hp2) == 0)
     i_2 = hp2;
 else
     i_2 =rem(k,hp2);
    end
 j_2 = ceil(k/hp2);
 
 i_2 = abs(i_2 -ceil(hp2/2));
 j_2 = abs(j_2 -ceil(hp2/2));
 
 
    %dist_euc =sqrt((i_1 -i_2)^2+(j_1 -j_2)^2);
    
    dist_int = sqrt(sum((patch_d -s_dict(k)).^2));
    weight(k) = exp(-dist_int/h);
end

weight = weight / sum(weight);
patch_out = (weight')*double(s_dict);
end