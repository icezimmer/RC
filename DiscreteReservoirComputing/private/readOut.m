function y = readOut(obj, x)

[~, num] = size(x);
y = obj.OutputWeights * [x; ones(1, num)];
end

