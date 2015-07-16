%%======================================================================
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

numPatches = 20000;
numFeatures = 121;
imageChannels = 3;
patchDim = 8;
visibleSize = patchDim * patchDim * imageChannels;

outputDir = '.';
epsilon = 1e-6; % L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)

%%======================================================================
%% STEP 1: Sample patches

patches = load('stlSampledPatches.mat');
patches = patches.patches(:, 1:numPatches);
displayColorNetwork(patches(:, 1:100));

%%======================================================================
%% STEP 2: ZCA whiten patches
%  In this step, we ZCA whiten the sampled patches. This is necessary for
%  orthonormal ICA to work.
numPatches = size(patches, 2);
patches = patches / 255;
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s))) * u';
patches = ZCAWhite * patches;

%%======================================================================
%% STEP 4: Optimization for orthonormal ICA
%  Optimize for the orthonormal ICA objective, enforcing the orthonormality
%  constraint. Code has been provided to do the gradient descent with a
%  backtracking line search using the orthonormalICACost function 
%  (for more information about backtracking line search, you can read the 
%  appendix of the exercise).
%
%  However, you will need to write code to enforce the orthonormality 
%  constraint by projecting weightMatrix back into the space of matrices 
%  satisfying WW^T  = I.
%
%  Once you are done, you can run the code. 10000 iterations of gradient
%  descent will take around 2 hours, and only a few bases will be
%  completely learned within 10000 iterations. This highlights one of the
%  weaknesses of orthonormal ICA - it is difficult to optimize for the
%  objective function while enforcing the orthonormality constraint - 
%  convergence using gradient descent and projection is very slow.

weightMatrix = rand(numFeatures, visibleSize);

[cost, grad] = orthonormalICACost(weightMatrix(:), visibleSize, numFeatures, patches, epsilon);

fprintf('%11s%16s%10s\n','Iteration','Cost','t');

startTime = tic();

% Initialize some parameters for the backtracking line search
t = 0.02;
% Do 10000 iterations of gradient descent
for iteration = 1:10000                   
    fprintf('  %9d  %14.6f  %8.7g\n', iteration, cost, t);
    grad = reshape(grad, size(weightMatrix));
    weightMatrix = weightMatrix - t*grad;
    weightMatrix = (weightMatrix*weightMatrix')^(-0.5) * weightMatrix;
    [cost, grad] = orthonormalICACost(weightMatrix(:), visibleSize, numFeatures, patches, epsilon);
           
    % Visualize the learned bases as we go along    
    if mod(iteration, 1000) == 0
        duration = toc(startTime);
        % Visualize the learned bases over time in different figures so 
        % we can get a feel for the slow rate of convergence
        figure(floor(iteration /  1000));
        save('ICAImages.mat', 'weightMatrix');
        displayColorNetwork(weightMatrix'); 
    end    
end

% Visualize the learned bases
save('finalICAImage.mat', 'weightMatrix');
displayColorNetwork(weightMatrix');
