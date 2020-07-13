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


% We make first forward propagation.

X = [ones(m, 1), X];
A = sigmoid(X*Theta1');
A = [ones(m, 1), A];
% H represents the output of the Neural Network. It is a matrix of size m times the number of labels
H = A*Theta2';

Y = zeros(m, num_labels);
for i = 1:m,
    Y(i, y(i)) = 1;
end

Uno = ones(size(Y, 2), 1);
B = zeros(m, 1);
H = sigmoid(H);
for i = 1:m,
    B(i) = sum(-Y(i,:)'.*log(H(i,:)').-(Uno-Y(i,:)').*log(Uno-H(i,:)'));
end

J = (1/m).*sum(B);

Theta1J = Theta1;
Theta2J = Theta2;
Theta1J(:, 1) = [];
Theta2J(:, 1) = [];
Theta1J = Theta1J(:);
Theta2J = Theta2J(:);
J = J + (lambda/(2*m))*(sum(Theta1J.*Theta1J)+sum(Theta2J.*Theta2J));

% -------------------------------------------------------------

% =========================================================================

Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end


Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);

% Theta1 has dimensions hidden_layer_size times (input_layer_size + 1)
% Theta2 has dimensions num_labels times (hidden_layer_size + 1)

% ======================== Forward Propagation ========================= %
for t = 1:m
    a_1 = X(t, :);
    a_1 = a_1(:);                    % a_1 is a vector of dimensions (input_layer_size + 1) times 1
    z_2 = Theta1*a_1;                % z_2 is a matrix with dimensions hidden_layer_size times 1
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];                  % a_2 has dimensions (hidden_layer_size + 1) times 1
    z_3 = Theta2*a_2;                % z_3 has dimensions num_labels times 1
    a_3 = sigmoid(z_3);              % a_3 has dimensions num_labels times 1

    
    delta_3 = zeros(num_labels, 1);  % delta_3 has dimesnions num_labels times 1
    delta_3 = a_3 - Y(t, :)';
    
   
    delta_2 = (Theta2'*delta_3);                % delta_2 matrix has dimesnions (hidden_layer_size + 1) times 1
    delta_2 = delta_2(2:end).*sigmoidGradient(z_2);            % delta_2 matrix has dimesnions (hidden_layer_size) times 1
    
    Delta_1 = Delta_1 + delta_2*a_1';
    Delta_2 = Delta_2 + delta_3*a_2';
end

  

Theta1_grad = (1/m).*Delta_1;
Theta2_grad = (1/m).*Delta_2;

for i = 1:hidden_layer_size,
    for j = 2:input_layer_size + 1,
            Theta1_grad(i, j) = Theta1_grad(i, j) + (lambda/m)*Theta1(i,j);
    end
end 

for i = 1:num_labels,
    for j = 2:hidden_layer_size + 1,
            Theta2_grad(i, j) = Theta2_grad(i, j) + (lambda/m)*Theta2(i,j);
    end
end 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
