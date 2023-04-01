function [x, x_ws, pooler, W_in, W_hat] = rc(u, seed, omega_in, rho, Nh, dns, a, ws, x0)

[Nu, time_steps] = size(u);

if nargin < 6 % no dns, a, ws, x0
    dns = 1;
    a = 1;
    ws = 0;
    x = zeros(Nh,1);
elseif nargin == 6 % no a, ws, x0
    a = 1;
    ws = 0;
    x = zeros(Nh,1);
elseif nargin == 7 % no ws, x0
    ws = 0;
    x = zeros(Nh,1);
elseif nargin == 8 % no x0
    x = zeros(Nh,1);
else
    x = x0;
end

if a < 0 || a > 1
    error('The parameter a must be in [0, 1]')
else
    W_in = initInputMatrix(Nu, omega_in, Nh, seed, a);
    W_hat = initStateMatrix(Nh, rho, seed, dns, a);
    
    % Add ones for bias
    u = [u; ones(1, time_steps)];
    
    % LI-ESN
    for t=1:time_steps
        x = cat(2, x, (1-a)*x(:,end) + a*tanh(W_in*u(:,t) + W_hat*x(:,end)));
    end
    
    % Discard the initial state
    x = x(:, 2:end);
    
    % Discard the washout
    x_ws = x(:, ws+1:end);
    
    pooler = x(:, end);
end

end

