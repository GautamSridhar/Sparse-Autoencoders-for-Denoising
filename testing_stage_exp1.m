    clear all
    close all
    
    load sae_file_structure.mat
    %parameters
    sim_wind = 5;   % measure of side of patch for similarity 
    patchsize = 15; %AE input patchsize
    swind_hsize = (sim_wind-1)/2;% half size of search window
    s =swind_hsize; 
    
    %read image
    filename = 'cameraman.tif';
    img = imread(filename);
    if (size(img,3)==3)
    img = rgb2gray(img);
    end
    figure; subplot(2,2,1);
    imshow(imread(filename),[])
    title('original')
    
    %corrupting image
    img = mat2gray(img);
    img = im2double(img);
    G = fspecial('gaussian',[3,3],1);
    img = imfilter(img,G,'same');
    img =imnoise(img,'speckle');
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    
    %creating search dictionary
    [m,n] =size(img);
    [s_d_testing,~] = test_patch_create(img,sim_wind);
    
    img_padded = padarray(img,[sim_wind-rem(m,sim_wind),sim_wind-rem(n,sim_wind)],'replicate','post');
    
    [u,v] = size(img_padded);
    ver = u/sim_wind;
    hor = v/sim_wind;
    count = 1;
    %creating testing output patches
    for c = 1:sim_wind:v-sim_wind+1
        for r= 1:sim_wind:u-sim_wind+1
        
            temp_patch = img_padded(r:r+sim_wind-1,c:c+sim_wind-1);
            patch = create_weighted_patch_2(reshape(temp_patch,[1,sim_wind^2]),s_d_testing,sim_wind,patchsize);
            patch = vertcat(reshape(temp_patch,[1 sim_wind^2]),patch);
            
            patch_rep = reshape(repmat(temp_patch,[patchsize/sim_wind,patchsize/sim_wind]),[patchsize^2 1]);
            out(:,count) = reshape(patch_reconst(patch,patchsize,patchsize,sim_wind),[patchsize^2,1]);        
            
            out_rep(:,count) = patch_rep;
            mean_img_sblock(:,count )= mean(mean(temp_patch));
            count =count + 1;
            
        end
    end
    assert(count-1==ver*hor,'test patch extraction incorrect');
    %normalising patches
    [out_norm,mean_p(:,:)] = normalizeData_t(out);
    
    %visualise mean patch
    subplot(2,2,3);
    imshow(reshape(mean_p,[ver,hor]),[])
    title('mean')
     
    
    %% Case 1 Taking average of input patches
    
    %feedforward testing patches
    nn = nnsetup([225 400 225]);
    nn.output                           = 'sigm';
    nn.activation_function              = 'sigm';
    opts.numepochs = 1000;
    opts.batchsize = 50;
    nn.dropoutFraction                  = 0;
    nn.W{1} = sae.ae{1}.W{1}; 
    nn.W{2} = sae.ae{1}.W{2};
    nn_testing = my_nnff(nn,out_norm');
    
    output = nn_testing.a{3};
    
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = patch_reconst(output_s,u,v,sim_wind);
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes.tif');
    imwrite(img_padded,'corrupted.tif');
    
 %% Case 2 Taking mean of individual blocks and adding zero mean AE output
   
    nn_testing = my_nnff(nn,out');
    output1 = nn_testing.a{3};
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output_1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = patch_reconst(output_s1,u,v,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes.tif');
    imwrite(img_padded,'corrupted.tif');
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])
    title('result')
    
    
 %% Case 3 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img_padded;
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_2,[])
    title('result')
    
    
    out_img_2 = mat2gray(out_img_2);
    out_img_2 = im2double(out_img_2);
    
    imwrite(out_img_2,'Addorig_caseRes.tif');
    
    
    %% using examples with repeated patch
    %following code makes use of test patches with 
    %repeated small blocks as input
   %% Case 4 Taking average of input patches
    
    out_norm_rep = normalizeData_t(out_rep);
    nn_testing = my_nnff(nn,out_norm_rep');
    
    output = nn_testing.a{3};
    
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = patch_reconst(output_s,u,v,sim_wind);
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_p,[ver,hor]),[])
    title(' block mean')
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes_rep.tif');
    
 %% Case 5 Taking mean of individual blocks and adding zero mean AE output
   
    nn_testing = my_nnff(nn,out_rep');
    output1 = nn_testing.a{3};
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = patch_reconst(output_s1,u,v,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes_rep.tif');
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])clear all
    close all
    
    load sae_file_structure.mat
    %parameters
    sim_wind = 5;   % measure of side of patch for similarity 
    patchsize = 15; %AE input patchsize
    swind_hsize = (sim_wind-1)/2;% half size of search window
    s =swind_hsize; 
    
    %read image
    filename = 'cameraman.tif';
    img = imread(filename);
    if (size(img,3)==3)
    img = rgb2gray(img);
    end
    figure; subplot(2,2,1);
    imshow(imread(filename),[])
    title('original')
    
    %corrupting image
    img = mat2gray(img);
    img = im2double(img);
    G = fspecial('gaussian',[3,3],1);
    img = imfilter(img,G,'same');
    img =imnoise(img,'speckle');
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    
    %creating search dictionary
    [m,n] =size(img);
    [s_d_testing,~] = test_patch_create(img,sim_wind);
    
    img_padded = padarray(img,[sim_wind-rem(m,sim_wind),sim_wind-rem(n,sim_wind)],'replicate','post');
    
    [u,v] = size(img_padded);
    ver = u/sim_wind;
    hor = v/sim_wind;
    count = 1;
    %creating testing output patches
    for c = 1:sim_wind:v-sim_wind+1
        for r= 1:sim_wind:u-sim_wind+1
        
            temp_patch = img_padded(r:r+sim_wind-1,c:c+sim_wind-1);
            patch = create_weighted_patch_2(reshape(temp_patch,[1,sim_wind^2]),s_d_testing,sim_wind,patchsize);
            patch = vertcat(reshape(temp_patch,[1 sim_wind^2]),patch);
            
            patch_rep = reshape(repmat(temp_patch,[patchsize/sim_wind,patchsize/sim_wind]),[patchsize^2 1]);
            out(:,count) = reshape(patch_reconst(patch,patchsize,patchsize,sim_wind),[patchsize^2,1]);        
            
            out_rep(:,count) = patch_rep;
            mean_img_sblock(:,count )= mean(mean(temp_patch));
            count =count + 1;
            
        end
    end
    assert(count-1==ver*hor,'test patch extraction incorrect');
    %normalising patches
    [out_norm,mean_p(:,:)] = normalizeData_t(out);
    
    %visualise mean patch
    subplot(2,2,3);
    imshow(reshape(mean_p,[ver,hor]),[])
    title('mean')
     
    
    %% Case 1 Taking average of input patches
    
    %feedforward testing patches
    nn = nnsetup([225 400 225]);
    nn.output                           = 'sigm';
    nn.activation_function              = 'sigm';
    opts.numepochs = 1000;
    opts.batchsize = 50;
    nn.dropoutFraction                  = 0;
    nn.W{1} = sae.ae{1}.W{1}; 
    nn.W{2} = sae.ae{1}.W{2};
    nn_testing = my_nnff(nn,out_norm');
    
    output = nn_testing.a{3};
    
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = patch_reconst(output_s,u,v,sim_wind);
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes.tif');
    imwrite(img_padded,'corrupted.tif');
    
 %% Case 2 Taking mean of individual blocks and adding zero mean AE output
   
    nn_testing = my_nnff(nn,out');
    output1 = nn_testing.a{3};
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output_1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = patch_reconst(output_s1,u,v,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes.tif');
    imwrite(img_padded,'corrupted.tif');
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])
    title('result')
    
    
 %% Case 3 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img_padded;
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_2,[])
    title('result')
    
    
    out_img_2 = mat2gray(out_img_2);
    out_img_2 = im2double(out_img_2);
    
    imwrite(out_img_2,'Addorig_caseRes.tif');
    
    
    %% using examples with repeated patch
    %following code makes use of test patches with 
    %repeated small blocks as input
   %% Case 4 Taking average of input patches
    
    out_norm_rep = normalizeData_t(out_rep);
    nn_testing = my_nnff(nn,out_norm_rep');
    
    output = nn_testing.a{3};
    
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = patch_reconst(output_s,u,v,sim_wind);
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_p,[ver,hor]),[])
    title(' block mean')
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes_rep.tif');
    
 %% Case 5 Taking mean of individual blocks and adding zero mean AE output
   
    nn_testing = my_nnff(nn,out_rep');
    output1 = nn_testing.a{3};
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = patch_reconst(output_s1,u,v,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes_rep.tif');
    
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])
    title('result')
    
    
 %% Case 6 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img_padded;
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_2,[])
    title('result')
    
    
    out_img_2 = mat2gray(out_img_2);
    out_img_2 = im2double(out_img_2);
    
    imwrite(out_img_2,'Addorig_caseRes_rep.tif');
    title('result')
    
    
 %% Case 6 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img_padded;
    figure; subplot(2,2,1);
    imshow(imread('cameraman.tif'),[])
    title('original')
    subplot(2,2,2);
    imshow(img_padded,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,[ver,hor]),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_2,[])
    title('result')
    
    
    out_img_2 = mat2gray(out_img_2);
    out_img_2 = im2double(out_img_2);
    
    imwrite(out_img_2,'Addorig_caseRes_rep.tif');
    