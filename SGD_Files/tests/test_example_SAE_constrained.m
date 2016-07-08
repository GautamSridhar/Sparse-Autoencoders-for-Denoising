% function test_example_SAE
clear all;
close all;

load patches
train_x = patches(:,:,1)';
train_x_gaussian = add_noise(patches(:,:,1),21);
train_x_gaussian =train_x_gaussian' ;
n=size(train_x);
n=n(1);
train_x_1 = train_x(1:floor(n*0.6),:);
test_x_1 = train_x(ceil(n*0.6):n,:);
train_x_gaussian_1 = train_x_gaussian(1:floor(n*0.6),:);
test_x_gaussian_1 = train_x_gaussian(ceil(n*0.6):n,:);
%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([441 600]);
%1st layer
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.1;
sae.ae{1}.inputZeroMaskedFraction   = 0.1;
sae.ae{1}.output                    = 'sigm';
sae.ae{1}.nonSparsityPenalty        = 0.1;
sae.ae{1}.sparsityTarget            = 0.02;
opts.numepochs = 1000;
opts.batchsize = 50;
sae = constrained_saetrain(sae, train_x_gaussian_1, opts, train_x_gaussian_1,test_x_gaussian_1,test_x_1);
save('sae_file_structure','sae');
figure;
visualize(sae.ae{1}.W{1}(:,2:end)')

savefig('kernels.png','png')
figure;
subplot(1,2,1);
plot(sae.ae{1}.Loss,'-b');
% legend('only first derivative','with second derivative');
title('Plot of Loss function');
xlabel('number of runs on training data');
ylabel('Loss')
hold off
subplot(1,2,2)
plot(sae.ae{1}.epochloss, '-b');
% legend('only first derivative','with second derivative');
title('Plot of loss in each epoch');
xlabel('number of epochs');
ylabel('epoch loss')
hold off
savefig('SAE_loss.png','png')

%% Use the SDAE to initialize a FFNN
nn = nnsetup([441 600 441]);
nn.output                           = 'sigm';
nn.activation_function              = 'sigm';
opts.numepochs = 1000;
opts.batchsize = 50;
nn.dropoutFraction                  = 0;
nn.W{1} = sae.ae{1}.W{1}; % stores the weights got by auto encoding in nn.w{1}
nn.W{2} = sae.ae{1}.W{2};
[nn, L]  = nntrain(nn, train_x_gaussian, train_x, opts);
save('nn_trained','nn');
savefig('nn_loss.png','png')
test_sgd.m
