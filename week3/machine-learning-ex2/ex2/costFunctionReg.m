function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% ============== Without Grader  =========================
% data = load('ex2data2.txt');
% X = data(:, [1, 2]); y = data(:, 3);
% X = mapFeature(X(:,1), X(:,2));
% theta = ones(size(X, 2), 1);
% lambda = 1;

% ========================================================

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

% Calculate the Cost
z = X * theta;

hTheta = sigmoid(z);

J = (- 1 / m) * sum((y .* log(hTheta)) + ((1 .- y) .* log((1 .- hTheta))));

regCost = theta(2:size(theta, 1), :) .^ 2;

regCost = (lambda / (2 * m))* sum(regCost);

J = J + regCost;

% Calculate the Gradient
grad = ((1 / m) *  (X' * (hTheta - y)));

regTheta = theta;

regTheta(1) = 0;

regTheta = (lambda / m) * regTheta;

grad = grad + regTheta;

% =============================================================

end
