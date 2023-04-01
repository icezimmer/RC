function [x, x_ws, pooler, W_in, W_hat] = eurc(u, seed, omega_in, omega_r, omega_b, Nh, eps, gamma, ws, x0)

[Nu, time_steps] = size(u);

if nargin < 9 % no ws, x0
    ws = 0;
    x = zeros(Nh,1);
elseif nargin < 10 % no x0
    x = zeros(Nh,1);
else
    x = x0;
end

W_in = initInputMatrix(Nu, omega_in, Nh, seed);
W_in = W_in(:,1:end-1); %leave bias
W_hat = euInitStateMatrix(Nh, omega_r, seed);
b = initBias(Nh, omega_b, seed);

% EuSN
for t=1:time_steps
    x = cat(2, x, x(:,end) + eps*tanh(b + W_in*u(:,t) + (W_hat-gamma*eye(Nh))*x(:,end)));
end

% Discard the initial state
x = x(:, 2:end);

% Discard the washout
x_ws = x(:, ws+1:end);

pooler = x(:, end);

end


