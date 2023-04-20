% function new_hidden = eulerForward(bias, input_weights, input, hidden_weights, hidden, f, eps)
% new_hidden = hidden + eps*f(bias, input_weights, input, hidden_weights, hidden);
% end
function new_hidden = eulerForward(bias, hidden_weights, hidden, f, eps)
new_hidden = hidden + eps*f(bias, hidden_weights, hidden);
end