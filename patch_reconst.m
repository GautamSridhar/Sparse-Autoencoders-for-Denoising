function out = patch_reconst(patch_dictionary_mod,m,n, sim_wind)
%%reconstructs a large patch out of the patch vector patch_dictionary of
%%size mxn
for k = 1: 1: m/sim_wind
    for l =1: n/sim_wind
    
        if (l== 1)
        horiz = reshape(patch_dictionary_mod((k-1)*m/sim_wind+l,:),[sim_wind,sim_wind]);
        else
        horiz = horzcat(horiz,reshape(patch_dictionary_mod((k-1)*m/sim_wind+l,:),[sim_wind,sim_wind]));
        end
      
    end
    if(k == 1)
        out = horiz;
    else  
    out = vertcat(out,horiz);
    clear horiz;    
    end
end
end