function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
% 	stat = zeros(1, length(theta));
% 	for j = 1:length(theta)
% 		for nsample = 1:m
% 			stat(j) = stat(j) + (theta(1) * X(nsample, 1) + theta(2) * X(nsample, 2) - y(nsample))*X(nsample, j);
% 		end
% 		theta(j) = theta(j) - alpha * 1/m * stat(j)
    theta = theta - alpha/m*X'*(X*theta - y);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
%disp(['theta(', num2str(j), ') is ', num2str(theta(j))]);

end
