function W_out = trainOffline(x, d, lambda_r, ws)

if nargin < 3
    lambda_r = 0;
    ws = 0;
elseif nargin < 4
    ws = 0;
end

if size(x,2) < size(d,2) && nargin < 4
    error('Hidden state dimension less than target dimension. Try to insert the washout!')
elseif size(x,2) == size(d,2) && nargin == 4 && ws > 0
    error('Hidden state and target dimension are equals. Try to not insert the washout!')
end

if ws > 0 %480 os the length of the timeseries
    d = washout(d, ws);
end

[Nh, num] = size(x);

X = [x; ones(1, num)];

if lambda_r == 0
    W_out = d * pinv(X);
elseif lambda_r > 0
    W_out = d * X' * inv(X*X' + lambda_r * eye(Nh+1));
end

end

