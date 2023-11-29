function obj = trainOutputWeightsOffline(obj, x, d)

[~, num_sample] = size(x);

X = [x; ones(1, num_sample)];

if obj.Regularization == 0
    obj.OutputWeights = d * pinv(X);
elseif obj.Regularization > 0
    obj.OutputWeights = d * (X' / (X*X' + obj.Regularization * eye(obj.NeuronsNumber+1)));
end

end

