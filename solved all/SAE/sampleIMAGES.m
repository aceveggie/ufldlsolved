function patches = sampleIMAGES(patchsize)
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 
numpatches = 10000;
% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
% [num_row, num_col, num_images] = size(images);
% patch_per_image = numpatches / num_images;
% patch_iter = 0;
% for i = 1:num_images
%     for j = 1:patch_per_image
%         patch_iter = patch_iter+1;
%         corner_x = randi(num_row-patchsize+1);
%         corner_y = randi(num_col-patchsize+1);
%         patches(:, patch_iter) = vec(images(corner_x:corner_x+patchsize-1, ...
%                                         corner_y:corner_y+patchsize-1, ...
%                                         i));
%     end
% end


for imageNum = 1:10%??????????1000?patch??10000?patch
    [rowNum colNum] = size(images(:,:,imageNum));
    for patchNum = 1:2000%????????1000?patch
        xPos = randi([1,rowNum-patchsize+1]);
        yPos = randi([1, colNum-patchsize+1]);
        patches(:,(imageNum-1)*2000+patchNum) = reshape(images(xPos:xPos+patchsize-1,yPos:yPos+patchsize-1,...
                                                        imageNum),patchsize*patchsize,1);
    end
end



%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
% patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
% patches = (patches + 1) * 0.4 + 0.1;

end
