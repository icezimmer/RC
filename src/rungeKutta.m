function  new_hidden = rungeKutta(bias, input_weights, input, hidden_weights, hidden, f, eps)
% RK4 method
A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
b = [1/6 1/3 1/3 1/6];
m = length(b);
k = zeros(length(hidden), m);
for j = 1 : m
    k(:,j) = f(bias, input_weights, input, hidden_weights, hidden + eps * sum(k * diag(A(j,:)), 2));
end
new_hidden = hidden + eps * sum(k * diag(b), 2);
end