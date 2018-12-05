function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% ============ Without Grader ================

% data = load('ex2data1.txt');
% X = data(:, [1, 2]); y = data(:, 3);
% [m, n] = size(X);
% X = [ones(m, 1) X];
% theta = zeros(n + 1, 1);

% ============================================


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X * theta;

hTheta = sigmoid(z);

J = (- 1 / m) * sum((y .* log(hTheta)) + ((1 .- y) .* log((1 .- hTheta))));

grad = ((1 / m) *  (X' * (hTheta - y)));

% =============================================================

end
