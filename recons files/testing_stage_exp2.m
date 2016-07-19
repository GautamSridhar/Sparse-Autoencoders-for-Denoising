    clear all
    close all
    
    load theta.mat
    load params.mat
    %parameters
    sim_wind = 7;   % measure of side of patch for similarity 
    patchsize = params.patchsize; %AE input patchsize
    swind_hsize = (sim_wind-1)/2;% half size of search window
    s =swind_hsize; 
    
    %read image
    filename = 'cameraman.tif';
    img = imread(filename);
    if (size(img,3)==3)
    img = rgb2gray(img);
    end
    
    %img = imresize(img,0.25);
    img =img(1:64,1:64);
    img_copy =img;
    figure; subplot(2,2,1);
    imshow(img,[])
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
    
    for i = 1:size(s_d_testing,1)
        
        temp_patch = reshape(s_d_testing(i,:),[sim_wind,sim_wind]);
        patch = create_weighted_patch_2(s_d_testing(i,:),s_d_testing,sim_wind,patchsize);
        patch = vertcat(reshape(temp_patch,[1 sim_wind^2]),patch);
        
        patch_rep = reshape(repmat(temp_patch,[patchsize/sim_wind,patchsize/sim_wind]),[patchsize^2 1]);
        out(:,i) = reshape(patch_reconst(patch,patchsize,patchsize,sim_wind),[patchsize^2,1]);
        
        out_rep(:,i) = patch_rep;
        mean_img_sblock(:,i)= mean(mean(temp_patch));
    end
    mean_img_sblock = reshape(mean_img_sblock,size(img))' ;
    mean_img_sblock = mean_img_sblock(:)'; 
    %normalising patches
    [out_norm,mean_p(:,:)] = normalizeData_t(out);
    
    %visualise mean patch
    subplot(2,2,3);
    imshow(reshape(mean_p,size(img))',[])
    title('mean')
    mean_p = reshape(mean_p,size(img))' ;
    mean_p = mean_p(:)';
    
    %% Case 1 Taking average of input patches
    
    activation = feedForwardAutoencoder(opttheta, params.hiddenSize, params.visibleSize, out_norm) ;

    output = activation';
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = img_recons(output_s,m,n,sim_wind);
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes.tif');
    imwrite(img,'corrupted.tif');
    
 %% Case 2 Taking mean of individual blocks and adding zero mean AE output
   
    activation = feedForwardAutoencoder(opttheta, params.hiddenSize, params.visibleSize, out) ;

    output1 = activation';
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output_1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = img_recons(output_s1,m,n,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes.tif');
    imwrite(img,'corrupted.tif');
    
    figure; subplot(2,2,1);
    imshow(img_copy,[])
    title('original')
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,size(img)),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])
    title('result')
    
    
 %% Case 3 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img;
    figure; subplot(2,2,1);
    imshow(img_copy,[])
    title('original')
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,size(img)),[])
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

   activation = feedForwardAutoencoder(opttheta, params.hiddenSize, params.visibleSize, out_norm_rep) ;

   output = activation';
    
    %create output image blocks
    output_s = zeros(size(output,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output,1)
        patch_major = reshape(output(i,:),[patchsize,patchsize]);
        output_s(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s(i,:) = mean_p(i) + output_s(i,:);  
    end
    out_img = img_recons(output_s,m,n,sim_wind);
    
    figure; subplot(2,2,1);
    imshow(img_copy,[])
    title('original')
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_p,size(img)),[])
    title(' block mean')
    
    subplot(2,2,4);
    imshow(out_img',[])
    title('result')
    
    out_img = mat2gray(out_img);
    out_img = im2double(out_img);
    
    imwrite(out_img','avg_patch_caseRes_rep.tif');
    
 %% Case 5 Taking mean of individual blocks and adding zero mean AE output

    activation = feedForwardAutoencoder(opttheta, params.hiddenSize, params.visibleSize, out_rep) ;

    output = activation';
   
    [output_1,~] = normalizeData_t(output1);
    
    output_s1 = zeros(size(output1,1),sim_wind^2);
    %extracting only the corner block
    for i =1:size(output1,1)
        patch_major = reshape(output_1(i,:),[patchsize,patchsize]);
        output_s1(i,:) = reshape(patch_major(1:sim_wind,1:sim_wind),[sim_wind^2,1]);   
        output_s1(i,:) = mean_img_sblock(i) + output_s(i,:);  
    end
    out_img_1 = img_recons(output_s1,m,n,sim_wind);
    
    out_img_1 = mat2gray(out_img_1);
    out_img_1 = im2double(out_img_1);
    
    imwrite(out_img_1','blockavg_caseRes_rep.tif');
    
    figure; subplot(2,2,1);
    imshow(img_copy,[])
    title('original')
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,size(img)),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_1',[])
    title('result')
    
    
 %% Case 6 Adding original image to zero mean AE output
    out_img_2 = out_img_1' + img;
    figure; subplot(2,2,1);
    imshow(img_copy,[])
    title('original')
    subplot(2,2,2);
    imshow(img,[])
    title('corrupted')
    subplot(2,2,3);
    imshow(reshape(mean_img_sblock,size(img)),[])
    title(' block mean')
    subplot(2,2,4);
    imshow(out_img_2,[])
    title('result')
    
    
    out_img_2 = mat2gray(out_img_2);
    out_img_2 = im2double(out_img_2);
    
    imwrite(out_img_2,'Addorig_caseRes_rep.tif');