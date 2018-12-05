function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% ============= Without Grader =================
% z = ones(5)
% ==============================================

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

epislon = exp(-(z));

denom = (1 .+ epislon);

g = 1 ./ denom ;

% =============================================================

end
