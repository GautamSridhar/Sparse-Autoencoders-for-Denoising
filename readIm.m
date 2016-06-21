function patches = readIm()
srcFiles = dir('C:\Users\Gautam Sridhar\Documents\MATLAB\train_data\*.jpg');  % the folder in which ur images exists
for i = 1 : 20000
    filename = strcat('C:\Users\Gautam Sridhar\Documents\MATLAB\train_data\',srcFiles(i).name);
    I = imread(filename);
    I = im2double(I);
    feat = reshape(I,[441,1]);
    patches(:,i) = feat;
    i
end
patches = normalizeData(patches);
end
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
