% function new_hidden = eulerForward(bias, input_weights, input, hidden_weights, hidden, f, eps)
% new_hidden = hidden + eps*f(bias, input_weights, input, hidden_weights, hidden);
% end
function new_hidden = eulerForward(bias, hidden_weights, hidden, f, eps, varargin)
numvarargs = length(varargin);
if numvarargs > 2
    error('myfuns:FrankWolfe:TooManyInputs', ...
        'requires at most 2 optional inputs');
end
% set defaults for optional inputs
optargs = {[], []};
optargs(1:numvarargs) = varargin;
[input_weights, input] = optargs{:};

if isempty(input_weights) && isempty(input)
    new_hidden = hidden + eps*f(bias, hidden_weights, hidden);
else
    new_hidden = hidden + eps*f(bias, hidden_weights, hidden, input_weights, input);
end

end