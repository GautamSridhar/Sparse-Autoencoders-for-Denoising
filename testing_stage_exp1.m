    clear all
    close all
   
    sim_wind = 7;   % measure of side of patch for similarity 
    patchsize = 21; %AE input patchsize
    swind_hsize = (sim_wind-1)/2;% half size of search window
    s =swind_hsize; 
    load sae_file_structure
    
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
    
    img_padded = padarray(img,[sim_wind-rem(m,sim_wind),sim_wind-rem(n,sim_wind)],'replicate','post');
    
    mean_vec = zeros(1,m*n);
    
    [u,v] = size(img_padded);
    ver = u/sim_wind;
    hor = v/sim_wind;
    out = zeros(patchsize^2,hor*ver);
    
    for r = 1:ver
        for c= 1:hor
        
            temp_patch = img_padded(r:r+sim_wind-1,c:c+sim_wind-1);
            patch = create_weighted_patch_2(reshape(temp_patch,[1,sim_wind^2]),s_d_testing,sim_wind,patchsize);
            patch = vertcat(reshape(temp_patch,[1 sim_wind^2]),patch);
            out(:,(r-1)*ver+c) = reshape(patch_reconst(patch,patchsize,patchsize,sim_wind),[patchsize^2,1]);        
        end
    end
    
    [out,mean_p(:,:)] = normalizeData_t(out);
    
    subplot(2,2,3);
    imshow(reshape(mean_p,[ver,hor]),[])
    title('mean')
     
    
    nn = nnsetup([441 600 441]);
    nn.output                           = 'sigm';
    nn.activation_function              = 'sigm';
    opts.numepochs = 1000;
    opts.batchsize = 50;
    nn.dropoutFraction                  = 0;
    nn.W{1} = sae.ae{1}.W{1}; % stores the weights got by auto encoding in nn.w{1}
    nn.W{2} = sae.ae{1}.W{2};
    nn_testing = my_nnff(nn,out');
    for i = 1: size(nn_testing.a{3},1)
    output(i,:) = (mean_p(i)'*ones(patchsize^2,1))'+ nn_testing.a{3}(i,:);
    end
    
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
    end
    out_img = patch_reconst(output_s,u,v,sim_wind);
    subplot(2,2,4);
    imshow(out_img,[])
    title('result')
    
    