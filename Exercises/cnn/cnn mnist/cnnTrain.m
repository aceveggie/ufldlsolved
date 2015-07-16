%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

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


% compute gradient
DEBUG=false;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db_numFilters = 2;
    db_filterDim = 9;
    db_poolDim = 5;
    db_images = images(:,:,1:10);
    db_labels = labels(1:10);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
                db_poolDim,numClasses);
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                                db_filterDim,db_numFilters,db_poolDim);
    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim), db_theta);
    % Use this to visually compare the gradients side by side
    [WcGrad, WdGrad, bcGrad, bdGrad] = cnnParamsToStack(grad, imageDim, db_filterDim, ...
        db_numFilters, db_poolDim, numClasses);
    [numWcGrad, numWdGrad, numbcGrad, numbdGrad] = cnnParamsToStack(numGrad, imageDim, ...
        db_filterDim, db_numFilters, db_poolDim, numClasses);
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp('[Wc numWc]');
    disp([WcGrad(:), numWcGrad(:)]);
    disp('[Wd numWd]');
    disp([WdGrad(:), numWdGrad(:)]);
    disp('[bc numbc]');
    disp([bcGrad(:), numbcGrad(:)]);
    disp('[bd numbd]');
    disp([bdGrad(:), numbdGrad(:)]);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 100;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

% opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
%                       numFilters,poolDim),theta,images,labels,options);
% save('opttheta.mat', 'opttheta', 'filterDim', 'numFilters', 'poolDim', 'numClasses', 'imageDim');

disp('loading classifier from prev session');
load opttheta.mat;