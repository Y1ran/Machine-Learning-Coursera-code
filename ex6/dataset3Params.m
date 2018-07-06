function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================

%predictions = svmPredict(model, Xval);

%svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma))

Para = zeros(62,2);

sigma = zeros(8,1);
C = zeros(8,1);

sigma = [0.01; 0.03; 0.1; 0.3; 1; 3;10 ;30];
C = [0.01; 0.03; 0.1; 0.3; 1; 3;10 ;30];


tmp = 0.01;


for i = 1:8
    for j = 1:8
    Para(8 * (i -1) + j,1) = sigma(i,:);
    end
end

for i = 1:8
    for j = 1:8
    Para(8 * (i -1) + j,2) = C(j,:);
    end
end
       
error = zeros(64,1);
for i = 1:64
    
    model = svmTrain(X, y, Para(i,1), @(x1, x2) gaussianKernel...
        (x1, x2, Para(i,2)));
    
    %  Note: You can compute the prediction error using 
    predictions = svmPredict(model, Xval);
    err_tmp = mean(double(predictions ~= yval));
    error(i,:) = err_tmp;
end

pos = 0;
for j = 1:64
    mins = min(error);
    if(error(j,:) == mins)
        pos = j;
        break;
    end
end

C = Para(pos,1);
sigma = Para(pos, 2);
    
      


% =========================================================================

end
