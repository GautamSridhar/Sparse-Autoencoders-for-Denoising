function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 

patchsize = 21; %AE input patchsize
swind_hsize = 21;% half size of search window
s =(patchsize-1)/2 ; 
sim_wind = 3;   % measure of side of patch for similarity 
numpatches = 10;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches,2);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

for  i=1:numpatches
        d = randi(10,1);
    
    %extract size of selected random image
    img = IMAGES(:,:,d);
    [m,n,p] = size(img);
    if (p == 3)
    img = rgb2gray(img);
    end
    
    img = mat2gray(img);
    img = im2double(img);

    img_padded = padarray(img,[swind_hsize,swind_hsize],'replicate');

    %center of patch
    r = swind_hsize+randi([s+1,m-s],1);
    c = swind_hsize+randi([s+1,n-s],1);
 
    search_window = img_padded(r-swind_hsize-s:r+swind_hsize+s,c-swind_hsize-s:c+swind_hsize+s);
    temp_patch = img_padded(r-s:r+s,c-s:c+s);
    temp_patch_copy = temp_patch;
    patch_mod = modify_patch(temp_patch,search_window,sim_wind);
    
    patch_mod(s:s+2,s:s+2) = temp_patch_copy(s:s+2,s:s+2);    
    patches(:,i,1) =reshape(patch_mod,[patchsize^2 1]);
    n = 1;
    switch n
        case 0
    patches(:,i,2) =reshape(patch_mod,[patchsize^2 1]);
        case 1
    patches(:,i,2) =reshape(repmat(temp_patch_copy(s:s+2,s:s+2),[patchsize/sim_wind,patchsize/sim_wind]),[patchsize^2 1]);        
        otherwise
            disp('not a valid option')
    end
end



% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches(:,:,1) = normalizeData(patches(:,:,1));
patches(:,:,2) = normalizeData(patches(:,:,2));

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
