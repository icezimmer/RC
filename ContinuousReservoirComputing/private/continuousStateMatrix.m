function W_hat = continuousStateMatrix(spectrum, seed)
rng(seed)

LAMBDA_blocks = {};
N_blocks = {};
for eigenvalue = spectrum'
    lambda = eigenvalue{1};
    eigenvalue_blocks_sizes = eigenvalue{2};
    re = lambda(1);
    im = lambda(2);
    if im == 0
        LAMBDA_blocks= cat(2, LAMBDA_blocks, repmat({re}, 1, sum(eigenvalue_blocks_sizes)));
        for size = eigenvalue_blocks_sizes
            N_block = double((1:size)==((1:size)-1).');
            N_blocks = cat(2,N_blocks,N_block);
        end
    else
        LAMBDA_blocks = cat(2, LAMBDA_blocks, repmat({[re, -im; im, re]}, 1, sum(eigenvalue_blocks_sizes)));
        for size = eigenvalue_blocks_sizes
            mask = double((1:size)==((1:size)-1).');
            N_block = kron(mask,eye(2));
            N_blocks = cat(2,N_blocks,N_block);
        end
    end
end

LAMBDA = blkdiag(LAMBDA_blocks{:});
N = blkdiag(N_blocks{:});

% Jordan matrix = diagonal matrix of eigenvalues + nilpotent matrix
J = LAMBDA + N;
W_hat = J;

%D = diag(eigenvalues);
%V = rand(Nh);
%W_hat = V * (D / V);
%W_hat = V - V';

