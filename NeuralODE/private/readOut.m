function y = readOut(z, W_out)

[~, num] = size(z);
y = W_out * [z; ones(1, num)];
end

