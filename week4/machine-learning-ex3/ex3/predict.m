function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)


% ============== Without Grader  ==================

% load('ex3data1.mat');
% load('ex3weights.mat');

% =================================================


% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1), X];

layer1 = X * Theta1';

layer1 = sigmoid(layer1);

layer1 = [ones(m, 1), layer1];

layer2 = layer1 * Theta2';

layer2 = sigmoid(layer2);

[maxVal, p] = max(layer2, [], 2);

% =========================================================================
end