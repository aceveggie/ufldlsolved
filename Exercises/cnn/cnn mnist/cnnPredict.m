load opttheta.mat

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10
tic;
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);
acc = sum(preds==testLabels)/length(preds);
% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f%% \n',acc*100);
fprintf('Time taken to load and classify: %f seconds\n', toc);