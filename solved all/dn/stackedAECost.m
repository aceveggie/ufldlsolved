function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end
% You might find these variables useful
N = size(data, 2);
groundTruth = full(sparse(labels, 1:N, 1));
%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
hidden1 = sigmoid(bsxfun(@plus, stack{1}.w * data, stack{1}.b));
hidden2 = sigmoid(bsxfun(@plus, stack{2}.w * hidden1, stack{2}.b));
output = softmaxTheta * hidden2;
output = exp(bsxfun(@minus, output, max(output)));
output = bsxfun(@rdivide, output, sum(output));
% Total cost of stacked autoencoder
% cost = -sum(vec(groundTruth .* log(output))) + 0.5*lambda*norm(softmaxTheta, 'fro')^2;
cost = -sum(vec(groundTruth .* log(output))) + 0.5*lambda*norm(softmaxTheta, 'fro')^2;
% Back-propagation for delta 
softmax_delta = -(groundTruth - output);
hidden2_delta = (softmaxTheta'*softmax_delta) .* hidden2 .* (1-hidden2);
hidden1_delta = (stack{2}.w'*hidden2_delta) .* hidden1 .* (1-hidden1);
% Back-propagation for gradient
softmaxThetaGrad = softmax_delta * hidden2' + lambda*softmaxTheta;
stackgrad{2}.w = hidden2_delta * hidden1'; 
stackgrad{2}.b = sum(hidden2_delta, 2);
stackgrad{1}.w = hidden1_delta * data';
stackgrad{1}.b = sum(hidden1_delta, 2);
% -------------------------------------------------------------------------
%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
