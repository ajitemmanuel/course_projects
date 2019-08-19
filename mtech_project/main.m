warning('off','all'), clc, close all
addpath('libs')

load data

facecount = 0;

nmp = zeros(1,5);

for i = 1:5
    QueryImage = rgb2gray(data(:,:,:,i));
%QueryImage = rgb2gray(imread('A1.png'));
%QueryImage = rgb2gray(uigetfile('*.jpg;*.png;*.tiff;*.bmp'));

figure; imshow(QueryImage);
title('input Query Image');

SearchImage = rgb2gray(imread('crowd3.jpg'));
%SearchImage = rgb2gray(uigetfile('*.jpg;*.png;*.tiff;*.bmp'));

figure; imshow(SearchImage);
title('Search Image');

querypoints = detectSURFFeatures(QueryImage);
SearchPoints = detectSURFFeatures(SearchImage);

figure; imshow(QueryImage);
title('100 Strongest Feature Points from Query Image');
hold on;
plot(querypoints.selectStrongest(100));

figure; imshow(SearchImage);
title('300 Strongest Feature Points from Scene Image');
hold on;
plot(SearchPoints.selectStrongest(300));

[queryFeatures, querypoints] = extractFeatures(QueryImage, querypoints);
[SearchImgFeatures, SearchPoints] = extractFeatures(SearchImage, SearchPoints);
QueryPairs = matchFeatures(queryFeatures, SearchImgFeatures);
MatchedQueryPoints = querypoints(QueryPairs(:, 1), :);
matchedSearchImgPoints = SearchPoints(QueryPairs(:, 2), :);

figure;
showMatchedFeatures(QueryImage, SearchImage, MatchedQueryPoints, ...
    matchedSearchImgPoints, 'montage');
title(['Matched Points ' num2str(matchedSearchImgPoints.Count)] );

x = matchedSearchImgPoints.Count;

nmp(1,i) = x;

threshold = 20;
if x >= threshold
    facecount = facecount + 1;
end

end

fprintf('facecount value is %d',facecount)