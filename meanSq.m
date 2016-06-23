function error = meanSq(theta, visibleSize, hiddenSize, data,train_type)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
error = 0;
[nFeatures, nSamples] = size(data(:,:,1));
switch train_type
    
    case 0
    data_pert = imnoise(data(:,:,1),'gaussian');
    [a1, a2, a3] = getActivation(W1, W2, b1, b2, data_pert);
    error = ((a3 - data(:,:,1)) .^ 2) ./ 2;
    
    case 1    
    
    data_pert = imnoise(data(:,:,1),'gaussian');    
    [a1, a2, a3] = getActivation(W1, W2, b1, b2, data_pert);
    error = ((a3 - data(:,:,2)) .^ 2) ./ 2;
end
function sigm = sigmoid(x)
 
sigm = 1 ./ (1 + exp(-x));
end
 
%-------------------------------------------------------------------
% This function calculate dSigmoid
%
function dsigm = dsigmoid(a)
dsigm = a .* (1.0 - a);
 
end
 
%-------------------------------------------------------------------
% This function return the activation of each layer
%
function [ainput, ahidden, aoutput] = getActivation(W1, W2, b1, b2, input)
 
ainput = input;
ahidden = bsxfun(@plus, W1 * ainput, b1);
ahidden = sigmoid(ahidden);
aoutput = bsxfun(@plus, W2 * ahidden, b2);
aoutput = sigmoid(aoutput);
end
end
