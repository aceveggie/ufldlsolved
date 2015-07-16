clc;
clear all;
close all;

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       


%% load data
trainData = loadMNISTImages('./mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1


testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10



visibleSize = inputSize;

%%  Randomly initialize the parameters for first autoencoder
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% train the first autoencoder using sparse autoencoder cost function
maxIter = 400;
options = struct('MaxIter', maxIter);
% tic;
% disp('training 1st auto encoder now...');
% myCostFunc = @(sae1Theta) sparseAutoencoderCost(sae1Theta, ...
%                                    inputSize, hiddenSizeL1, ...
%                                    lambda, sparsityParam, ...
%                                    beta, trainData);

% [sae1OptTheta, cost] = fmincg(myCostFunc, sae1Theta(:), options);
% fprintf('training 1st auto encoder took %f seconds.\n', toc);
% save sae1OptTheta.mat sae1OptTheta;


disp('--loading first autoencoder from previous pre-training session--');
load sae1OptTheta.mat;

W1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
W2 = reshape(sae1OptTheta(hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize), inputSize, hiddenSizeL1);
b1 = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
b2 = sae1OptTheta(2*hiddenSizeL1*inputSize+hiddenSizeL1+1:end);

%% replace input data with transformed data using feed forwarding

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);



%%  Randomly initialize the parameters for second autoencoder
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% train the second autoencoder using sparse autoencoder cost function with 

% tic;
% disp('training 2nd auto encoder now...');
% myCostFunc = @(sae2Theta) sparseAutoencoderCost(sae2Theta, ...
%                                    hiddenSizeL1, hiddenSizeL2, ...
%                                    lambda, sparsityParam, ...
%                                    beta, sae1Features);

% [sae2OptTheta, cost] = fmincg(myCostFunc, sae2Theta(:), options);
% fprintf('training 2nd auto encoder took %f seconds.\n', toc);
% save sae2OptTheta.mat sae2OptTheta;

disp('--loading second autoencoder from previous pre-training session--');
load sae2OptTheta.mat;


%% replace input data with transformed data using feed forwarding

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%%  Randomly initialize the parameters for final softmax layer
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% train the final layer (SOFTMAX REGRESSION) using the training data transformed by
%% two sparse autoencoder feed forwarding network
%% the cost function to optimize is the cost function of a softmax regression


% softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, sae2Features, trainLabels, options);
% saeSoftmaxOptTheta = softmaxModel.optimizedTheta(:);
% save saeSoftmaxOptTheta.mat saeSoftmaxOptTheta;

disp('--loading saeSoftmaxOptTheta from previous training session--');
load saeSoftmaxOptTheta.mat


% stack first, second auto encoder along with their weights, biases into a cell to
% obtain the final theta for the entire stacked auto encoder

stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% fine tuning

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

% disp('--training stackedAE, optimizing stackedAECost function--');
% myCostFunc = @(stackedAETheta) stackedAECost(stackedAETheta(:), inputSize, hiddenSizeL2, numClasses, netconfig, lambda, trainData, trainLabels);
% [stackedAETheta(:), cost] = fmincg(myCostFunc, stackedAETheta(:), options);
% save stackedAETheta.mat stackedAETheta;
disp('--loading stackedAETheta from previous training session--');
load stackedAETheta;



% disp('--fine tuning stacked Auto encoder to increase  accuracy --');
% stackedAEOptTheta = fineTuning(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, ...
%                                lambda, trainData, trainLabels);
% save stackedAEOptTheta.mat stackedAEOptTheta;

disp('--loading stackedAEOptTheta from previous training session --');
load stackedAEOptTheta;



[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 98.40%
% After Finetuning Test Accuracy:  98.43%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
