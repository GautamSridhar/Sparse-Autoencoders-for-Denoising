%% comment put the part before load patches to avoid fresh creation of testing dataset
clear all
close all

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
params.patchsize = 21;
params.visibleSize = params.patchsize* params.patchsize;   % number of input units 
params.hiddenSize = 500;     % number of hidden units 
params.sparsityParam = 0.05;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
params.lambda = 0.00001;     % weight decay parameter       
params.beta =0.001;            % weight of sparsity penalty term       
params.train_type = 0; % 0 or 1 depending on repeated and non repeated case
params.batchsize = 250;
params.alpha = 0.1;
%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset
%[patches,patches_norm] = sampleIMAGES(params.train_type,params.patchsize);
% save('patches_n_7.mat','patches')
% save('patches_n_7_norm.mat','patches_norm')
load patches_norm_7
load patches_7

str1 = randi(size(patches,2),200,1);

display_network(patches(:,str1,1));
figure;display_network(patches_norm(:,str1,1));

[p,q,~] = size(patches);

Ytrain = patches(:,1:ceil(0.9*q),:);
Xtrain = patches_norm(:,1:ceil(0.9*q),:);
Yval = patches(:,ceil(0.9*q)+1:end,:);
Xval = patches_norm(:,ceil(0.9*q)+1:end,:); 

%  Obtain random parameters theta
%theta = initializeParameters(params.hiddenSize, params.visibleSize);

%%======================================================================
% %% STEP 2: Implement sparseAutoencoderCost
% %  Feel free to change the training settings when debugging your
% %  code.  (For example, reducing the training set size or 
% %  number of hidden units may make your code run faster; and setting beta 
% %  and/or lambda to zero may be helpful for debugging.)  However, in your 
% %  final submission of the visualized weights, please use parameters we 
% %  gave in Step 0 above.
% 
% [cost, grad] = sparseAutoencoderCost(theta, params.visibleSize, params.hiddenSize, params.lambda, ...
%                                      params.sparsityParam, params.beta, patches,train_type);
% 
% %%======================================================================
% STEP 3: Gradient Checking

% Hint: If you are debugging your code, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
% units) may speed things up.
% 
% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following: 
% checkNumericalGradient();
% % 
% % Now we can use it to check your cost function and derivative calculations
% % for the sparse autoencoder.  
% numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, params.visibleSize, ...
%                                                  params.hiddenSize, params.lambda, ...
%                                                   params.sparsityParam, params.beta, ...
%                                                  Xtrain,params.train_type), theta);
% 
% % Use this to visually compare the gradients side by side
% disp([numgrad grad]); 
% 
% %Compare numerically computed gradients with the ones obtained from backpropagation
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % Should be small. In our implementation, these values are
%             %usually less than 1e-9.
% 
%             %When you got this working, Congratulations!!! 

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(params.hiddenSize, params.visibleSize);

%  Use minFunc to minimize the function
%addpath minFunc/

options = optimset('MaxIter', 600);	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
options.Method = 'LBFGS'; % Here, we use conjugate gradient to optimize our cost
                          % function. L-BFGS can also be used as in the original exercise 
                          % code
%[opttheta, cost,exitflag,output]= minFunc(@(p)sparseAutoencoderCost(p,params.visibleSize,params.hiddenSize,params.lambda,params.sparsityParam,params.beta,Xtrain,params.train_type),theta,options);                         
[opttheta, cost] = fmincg(@(p)sparseAutoencoderCost(p,params.visibleSize,params.hiddenSize,params.lambda,params.sparsityParam,params.beta,Xtrain,Ytrain,params.patchsize, ...
    params.train_type),theta,options,Xval,Yval,params); 
%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:params.hiddenSize*params.visibleSize), params.hiddenSize, params.visibleSize);
figure;display_network(W1', 12); 
savefig('kernels.png','png')
  % save the visualization to a file 
%======================================================================
%% STEP 6: Prediction
testData = imread('cameraman.tif');
figure; subplot(3,1,1);
imshow(testData,[])
G = fspecial('gaussian',[3,3],1);
%testData = imnoise(testData,'gaussian',0,0.001);
testData = imfilter(testData,G,'same');
%testData =imnoise(testData,'salt & pepper');
testData =imnoise(testData,'speckle');
[m,n] = size(testData);

subplot(3,1,2);
imshow(testData,[])
test_patches = test_patch_create(testData,params.patchsize);

%output = feedForwardAutoencoder(opttheta, hiddenSize, visibleSize, testData);
%output_image = reshape(output, [21 21]);

%feedforward all testing examples
output = feedForwardAutoencoder(opttheta, params.hiddenSize, params.visibleSize, test_patches);

%generate output image
out_img = img_recons(output,m,n,params.patchsize);
subplot(3,1,3);imshow(out_img,[])
savefig('testing example.png','png')
imwrite(out_img,'output.tif');
%%======================================================================
%% OPTIONAL: Cross Validation
% [error_train,error_val] = crossValidate(X_train,X_val,lambda, sparsityParam, beta,...
%                                                    hiddenSize,visibleSize,type_train);
                                                   

