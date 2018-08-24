function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

m = length(pval);

bestEpsilon = 0;
bestF1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
%tmp = (max(pval) - min(pval))/ stepsize;
%F1 = zeros(tmp,1);
cvPred = zeros(m, 1);
%count = 1;

F1 = 0;

for epsilon = min(pval):stepsize:max(pval)
    for i = 1:m
        if(pval(i) < epsilon)
            cvPred(i) = 1;
        end
    end
    
    fp = sum((cvPred == 1) & (yval == 0));
    tp = sum((cvPred == 1) & (yval == 1));
    fn = sum((cvPred == 0) & (yval == 1));
    
    prec = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = 2 * prec * recall / (prec + recall);
      
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
    
    % ====================== YOUR CODE HERE ======================



end
