function [optTheta,cost] = sgd(batchsize,alpha,theta, visibleSize, hiddenSize, ...
                                lambda, sparsityParam, beta, data,train_type)
theta_int = theta;
m = size(data,2);
v = zeros(size(theta));
figure; hold on;
axis([0 m 0 50]);
for i=1:batchsize:m
data_int = data(:,i:i+batchsize,:);
[cost,grad] = sparseAutoencoderCost(theta_int,visibleSize,hiddenSize,lambda,sparsityParam,beta,data_int,train_type);
thet = alpha.*grad;
theta_out = theta_int - thet;
theta_int = theta_out;
% v = gamma.*v + thet; % Uncomment for momentum
% theta_out = theta_int - v; % based update
% theta_int = theta_out;
plot(i,cost,'c*');
drawnow;
end
optTheta = theta_out;
end
