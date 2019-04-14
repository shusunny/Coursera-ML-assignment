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
% CostFunc and Gradient with for loop, slow but easy to understand

a1 = [ones(m,1), X];
z2 = a1*Theta1';
a2 = [ones(size(z2,1),1), sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);


y_vec = zeros(m, num_labels);

 for i = 1:m,
   y_vec(i,y(i)) = 1;
   J += (-y_vec(i,:) * log(a3(i,:))' - (1 - y_vec(i, :)) * log(1 - a3(i,:))');
endfor
 
J = J/m;

% Implement Regularization

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);

reg = lambda/2/m*(sum(sum(t1.^2)) + sum(sum(t2.^2)));
J = J + reg;


% Implement the backpropagation algorithm and regularization to gradients.

G1 = zeros(size(Theta1))
G2 = zeros(size(Theta2))

for i = 1:m,
  ra1 = X(i,:)';
  rz2 = Theta1*ra1;
  ra2 = [1;sigmoid(rz2)];
  rz3 = Theta2*ra2;
  ra3 = sigmoid(rz3);
  
  err3 = ra3 - y_vec(i,:)';
  err2 = (Theta2'*err3)(2:end,1) .* sigmoidGradient(rz2);
  
  G1 = G1 + err2* ra1';
  G2 = G2 + err3* ra2';
endfor

Theta1_grad = G1/m + lambda/m * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = G2/m + lambda/m * [zeros(size(Theta2,1),1) Theta2(:,2:end)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
