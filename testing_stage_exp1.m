    clear all
    close all
    
    load sae_file_structure.mat
    sim_wind = 7;   % measure of side of patch for similarity 
    patchsize = 21; %AE input patchsize
    swind_hsize = (sim_wind-1)/2;% half size of search window
    s =swind_hsize; 

    filename = 'cameraman.tif';
    img = imread(filename);
    if (size(img,3)==3)
    img = rgb2gray(img);
    end
    figure; subplot(2,2,1);
    imshow(img,[])
    title('original')
    
    img = mat2gray(img);
    img = im2double(img);
    G = fspecial('gaussian',[3,3],1);
    img = imfilter(img,G,'same');
    img =imnoise(img,'speckle');
    
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    
    [m,n] =size(img);
    
    [s_d_testing,~] = test_patch_create(img,sim_wind);
    
    img_padded = padarray(double(img),[sim_wind-rem(m,sim_wind),sim_wind-rem(n,sim_wind)],'replicate','post');
    %out = zeros(patchsize^2,m*n);
    mean_vec = zeros(1,m*n);
    [u,v] = size(img_padded);
    
    for r = s+1:sim_wind:u-s-1
        for c= s+1:sim_wind:v-s-1
        
            temp_patch = img_padded(r-s:r+s,c-s:c+s);
            patch = create_weighted_patch_2(reshape(temp_patch,[1,sim_wind^2]),s_d_testing,sim_wind,patchsize);
            patch = vertcat(reshape(temp_patch,[1 sim_wind^2]),patch);
            out(:,ceil(r/sim_wind)*(ceil(u/sim_wind)-1)+ceil(c/sim_wind)) = reshape(patch_reconst(patch,patchsize,sim_wind),[patchsize^2,1]);        
        end
    end
    
    [out,mean_p] = normalizeData_t(out);
    
    subplot(2,2,3);
    imshow(reshape(out,[u/sim_wind,v/sim_wind]),[])
    title('mean')
    
    
    load sae_file_structure.mat
    nn = nnsetup([441 600 441]);
    nn.output                           = 'sigm';
    opts.numepochs = 1000;
    opts.batchsize = 50;
    nn.dropoutFraction                  = 0;
    nn.W{1} = sae.ae{1}.W{1}; % stores the weights got by auto encoding in nn.w{1}
    nn.W{2} = sae.ae{1}.W{2};
    
    nn_testing = my_nnff(nn,out);
    for i = 1: size(nn_testing.a{3},1)
    output(i,:) = (mean_p(i)*ones(patchsize^2,1))'+ nn_testing.a{3}(i,:);
    end
    
    out_img = img_recons(output,m,n,patchsize);
    subplot(2,2,4);
    imshow(out_img,[])
    title('result')
 
    
