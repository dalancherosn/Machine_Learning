function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
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

vector = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

errorMatrix = zeros(length(vector), length(vector));

for i = 1:length(vector),
    C = vector(i);
    for j = 1:length(vector),
        sigma = vector(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        errorMatrix(i, j) = mean(double(predictions ~= yval));
    end
end

error = min(min(errorMatrix));

for i = 1:length(vector),
    for j = 1:length(vector),
        if errorMatrix(i, j) == error,
            C = vector(i);
            sigma = vector(j);
        end
    end
end

% =========================================================================

end
