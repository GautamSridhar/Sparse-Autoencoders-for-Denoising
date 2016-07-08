% function test_example_SAE
clear all;
close all;
[train_x,train_x_gaussian] = my_patches();

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([441 600]);
%1st layer
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.2;
sae.ae{1}.inputZeroMaskedFraction   = 0.0;
sae.ae{1}.output                    = 'sigm';
sae.ae{1}.nonSparsityPenalty        = 0.1;
sae.ae{1}.sparsityTarget            = 0.2;
opts.numepochs = 1000;
opts.batchsize = 200;
sae = saetrain(sae, train_x_gaussian, opts, train_x_gaussian);
save('sae_file_structure','sae');
figure;
visualize(sae.ae{1}.W{1}(:,2:end)')

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

% for i=1:5    
% figure;
% subplot(1,2,1);
% plot(sae.ae{1}.psnr(:,i),'-b');
% title('Plot of PSNR in each epoch');
% xlabel('number of runs on training data');
% ylabel('PSNR')
% hold off
% 
% subplot(1,2,2)
% plot(sae.ae{1}.ssim(:,i), '-b');
% title('Plot of SSIM in each epoch');
% xlabel('number of epochs');
% ylabel('SSIM')
% hold off
% end

% train_x;
% train_x_gaussian;
%% Use the SDAE to initialize a FFNN
nn = nnsetup([441 600 441]);
nn.output                           = 'sigm';
nn.activation_function              = 'sigm';
opts.numepochs = 1000;
opts.batchsize = 200;
nn.dropoutFraction                  = 0;
nn.W{1} = sae.ae{1}.W{1}; % stores the weights got by auto encoding in nn.w{1}
nn.W{2} = sae.ae{1}.W{2};
[nn, L]  = nntrain(nn, train_x_gaussian, train_x, opts);
save('nn_trained','nn');
test_reconst;