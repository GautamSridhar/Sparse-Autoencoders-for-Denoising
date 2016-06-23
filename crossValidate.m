% Function for calculating the training and cross validation errors
% over different values of the training set to choose the correct
% amount of data for training and choose appropriate hyperparameters
function [error_train,error_val] = crossValidate(X_train,X_val,lambda, sparsityParam, beta,...
                                                   hiddenSize,visibleSize,train_type)
m = size(X_train,2);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i=1:m
    theta = initializeParameters(hiddenSize, visibleSize);
    costFunc = @(p) sparseAutoencoderCost(p,visibleSize,hiddenSize,lambda, sparsityParam, beta,X_train(:,1:i),train_type);
    options = optimset('MaxIter', 600);
    [opttheta,~] = fmincg(costFunc,theta,options);
    [error_train(i),~] = sparseAutoencoderCost(opttheta,visibleSize,hiddenSize,0,sparsityParam,0,X_train(:,1:i),train_type);
    [error_val(i),~] = sparseAutoencoderCost(opttheta,visibleSize,hiddenSize,0,sparsityParam,0,X_val,train_type);
end
plot(1:m, error_train, 1:m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])            %can be modified accordingly
    
    


