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
X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;

y_tmp = zeros(m, num_labels);
total = 0;

for i = 1: m
    row_label = y(i);
    y_tmp(i, row_label) = 1;
    
    output = sigmoid([ones(1,1) sigmoid(X(i,:) * Theta1')]...
        * Theta2');
    cost = -y_tmp(i, :) * log(output') - (ones(1, num_labels)...
        - y_tmp(i, :)) * log(ones(num_labels, 1) - output');
    
    total = total + cost;
end

sums_1 = 0;
sums_2 = 0;


for i = 1 : hidden_layer_size
    for j = 2: (input_layer_size + 1)
        tmp_theta1 = (Theta1(i,j)) ^ 2;
        sums_1 = sums_1 + tmp_theta1;
    end
end

for i = 1 : num_labels
    for j = 2: (hidden_layer_size + 1)
        tmp_theta2 = (Theta2(i,j)) ^ 2;
        sums_2 = sums_2 + tmp_theta2;
    end
end

penal_sum = lambda / 2 * (sums_1 + sums_2);
            
J = 1 / m * (total + penal_sum);

    
%compute the BP algrithm

delta_total1 = zeros(size(Theta1));
%delta_total2 = zeros(num_labels, hidden_layer_size);
delta_total2 = zeros(size(Theta2));

for i = 1:m     
    
    %compute the layer-wise units
    a1 = X(i,:);
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [1 a2];
    
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    z2 = [1 z2];
    
    %compute the delta within layers
    delta3 = a3 - y_tmp(i);
    delta_tmp =  delta3 * Theta2; 
    delta2 =  delta_tmp .* sigmoidGradient(z2);
        
    delta2 = delta2(2 : end);
    %sum all the delta by formula
    delta_total1 = delta_total1 + delta2' * a1;
    delta_total2 = delta_total2 + delta3' * a2;
    
end



Theta1_grad = (1 / m) .* delta_total1;
Theta2_grad = (1 / m) .* delta_total2;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
