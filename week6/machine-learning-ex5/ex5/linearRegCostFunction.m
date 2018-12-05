function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% ======================== Without Grader =======================

%load ('ex5data1.mat');
%mX = size(X, 1);
%X = [ones(mX, 1) X];
%theta = [1 ; 1];
%lambda = 1;

% ===============================================================


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = (1 / (2 * m)) * sum(((X * theta) - y) .^ 2);

regTheta = [zeros(1, 1); theta(2:end, :)];

J =  J  + (lambda / (2 * m)) * sum((regTheta .^ 2));

hTheta = (X * theta);
hTheta = hTheta - y;

grad = X' * hTheta;

grad = ((1 / m) * grad) + ((lambda / m) * regTheta);

% =========================================================================

grad = grad(:);


end
