function  new_hidden = rungeKutta(bias, hidden_weights, hidden, f, eps, varargin)
numvarargs = length(varargin);
if numvarargs > 2
    error('myfuns:FrankWolfe:TooManyInputs', ...
        'requires at most 2 optional inputs');
end
% set defaults for optional inputs
optargs = {[], []};
optargs(1:numvarargs) = varargin;
[input_weights, input] = optargs{:};
% RK4 method
A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
b = [1/6 1/3 1/3 1/6];
m = length(b);
k = zeros(length(hidden), m);
for j = 1 : m
    if isempty(input_weights) && isempty(input)
        k(:,j) = f(bias, hidden_weights, hidden + eps * sum(k * diag(A(j,:)), 2));
    else
        k(:,j) = f(bias, hidden_weights, hidden + eps * sum(k * diag(A(j,:)), 2), input_weights, input);
    end
end
new_hidden = hidden + eps * sum(k * diag(b), 2);
end