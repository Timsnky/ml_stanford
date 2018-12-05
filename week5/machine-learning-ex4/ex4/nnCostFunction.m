function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


% nnCostFunction============== Without Grader ====================

%load('ex4data1.mat');
%input_layer_size  = 400;  % 20x20 Input Images of Digits
%hidden_layer_size = 25;   % 25 hidden units
%num_labels = 10;  
%lambda = 0;

%load('ex4weights.mat');
%nn_params = [Theta1(:) ; Theta2(:)];


% ==================================================

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%X = X(1:1, :);
%m = size(X, 1);
%y = y(1:1, :);
%disp(size(X));
%disp(size(Theta1));
%disp(size(Theta2));

% ======== Feedforward =======================
% ========= Computing Cost Function ==========
activation1 = [ones(m, 1), X];

z2 = sigmoid(activation1 * Theta1');
activation2 = [ones(m, 1), z2];

activation3 = sigmoid(activation2 * Theta2');
hTheta = activation3;

costY = ones(m, num_labels) .* [1:num_labels];
costY = costY == y;

layerSum = (-costY .* log(hTheta)) - ((1 .- costY) .* log(1 .- hTheta));
layerSum = sum(layerSum, 2);

J = (1 / m) * sum(layerSum);

% ============== Computing regularization parameter =================

regTheta1 = Theta1( :, 2:size(Theta1, 2)) .^ 2;
regTheta1 = sum(regTheta1, 1);
regTheta1 = sum(regTheta1);

regTheta2 = Theta2( :, 2:size(Theta2, 2)) .^ 2;
regTheta2 = sum(regTheta2, 1);
regTheta2 = sum(regTheta2);

regSum = (lambda / (2 * m)) * (regTheta1 + regTheta2);

J = J + regSum;

% ========== Back Propagation ====================================

costY = ones(m, num_labels) .* [1:num_labels];
costY = costY == y;

% =============== Vectorized Method =====================================

delta3 = activation3 - costY;

sigmoidZ2 = z2 .* (1 -z2);
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidZ2;

acDelta2 = (delta3' * activation2);
Theta2_grad = Theta2_grad + acDelta2;
Theta2_grad = (1 / m) .* Theta2_grad;
Theta2Reg = Theta2(:, 2:end);
Theta2Reg = [zeros(size(Theta2, 1), 1), Theta2Reg];
Theta2_grad = Theta2_grad + ((lambda / m) .* Theta2Reg);

acDelta1 = (delta2' * activation1);
Theta1_grad = Theta1_grad + acDelta1;
Theta1_grad = (1 / m) .* Theta1_grad;
Theta1Reg = Theta1(:, 2:end);
Theta1Reg = [zeros(size(Theta1, 1), 1), Theta1Reg];
Theta1_grad = Theta1_grad + ((lambda / m) .* Theta1Reg);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
