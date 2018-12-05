function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% =============== Without Grader ======================

% load('ex6data3.mat');

% ====================================================

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

stepsSize = size(steps, 2);
sampleParams = zeros(stepsSize ^ 2, 3);

for row = 1:stepsSize
	for collumn = 1:stepsSize
		sampleParams(stepsSize * (row - 1) + collumn, :) = [steps(1, row), steps(1, collumn), 0];
	end
end

for index = 1:size(sampleParams, 1)

	model= svmTrain(X, y, sampleParams(index, 1), @(x1, x2) gaussianKernel(x1, x2, sampleParams(index, 2)));
	
	predictions = svmPredict(model, Xval);

	predError = mean(double(predictions ~= yval));

	sampleParams(index, 3) = predError;
end

[minError, minRow] = min(sampleParams(:, 3));

C = sampleParams(minRow, 1);
sigma = sampleParams(minRow, 2);

% =========================================================================

end
