function s_d = create_search_dict()

sim_wind =3;

srcFiles = dir('/nfs4/gautam/Documents/MATLAB/images/ImageNet_train_100/*.JPEG'); 

for k = 1 : 50
    %Reading the image files

    filename = strcat('/nfs4/gautam/Documents/MATLAB/images/ImageNet_train_100/',srcFiles(k).name);
    I = imread(filename);
    if (size(I,3)==3)
    I = rgb2gray(I);
    end
    
    I = mat2gray(I);
    I = im2double(I);

    %first patch
    if(k == 1)
    s_d = test_patch_create(I,sim_wind);
    else
    s_d = vertcat(s_d,test_patch_create(I,sim_wind));
    end

end
end