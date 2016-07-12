function nn = my_nnff(nn, x)

%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
    % Calculate the unit's outputs (including the bias term)
    nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
           
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
    end

