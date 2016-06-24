function aoutput = feedForwardAutoencoder(opttheta, hiddenSize, visibleSize, input)


W1 = opttheta(1:hiddenSize*visibleSize);
W1_prime = opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize);

W1 = reshape(W1,[hiddenSize,visibleSize]);
W1_prime = reshape(W1_prime,[visibleSize,hiddenSize]);

b1 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b1_prime = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);


[ainput, ahidden, aoutput] = getActivation(W1, W1_prime, b1, b1_prime, input');
aoutput = aoutput';
end

function [ainput, ahidden, aoutput] = getActivation(W1, W2, b1, b2, input)
 
ainput = input;
ahidden = bsxfun(@plus, W1 * ainput, b1);
ahidden = sigmoid(ahidden);
aoutput = bsxfun(@plus, W2 * ahidden, b2);
aoutput = sigmoid(aoutput);
end