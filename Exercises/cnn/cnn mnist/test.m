% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
images = loadMNISTImages('train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

disp('initializing parameters');
% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output
hiddenSize = outputDim^2*numFilters; % dimension of input to softmax classifier

% for filterNum = 1:numFilters
%     % convolution of image with feature matrix
%     % Obtain the feature (filterDim x filterDim) needed during the convolution
%     filter = Wc(:, :, filterNum);
%     % Flip the feature matrix because of the definition of convolution, as explained later
%     filter = rot90(squeeze(filter),2);
%     % Vectorized code to do convolution to all the images at the same time
%     convolvedImages = convn(images, filter, 'valid');
%     convolvedImages = convolvedImages + bc(filterNum);
%     convolvedImages = 1 ./ (1+exp(-convolvedImages));
%     convolvedFeatures(:, :, filterNum, :) = convolvedImages;
% end

