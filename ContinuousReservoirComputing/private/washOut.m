function d_washout = washOut(obj, d)
d_washout = {};

for i = 1:numel(d)
    vector = d{i};
    d_washout = cat(2,d_washout, {vector(1+obj.Transient:end)});
end

end