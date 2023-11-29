function obj = trainOutputNetworkOffline(obj, z, d)

num_samples = size(z, 2);

Z = [z; ones(1, num_samples)];

if obj.Regularization == 0
    obj.OutputWeights = d * pinv(Z);
    obj.OutputNetwork = @(z) readout(z, obj.OutputWeights);
elseif obj.Regularization > 0
    obj.OutputWeights = d * (Z' / (Z*Z' + obj.Regularization * eye(obj.HiddenSize+1)));
    obj.OutputNetwork = @(z) readout(z, obj.OutputWeights);
end

end

