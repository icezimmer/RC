function d_washout = washout(d,ws)
d_washout = {};

for i = 1:numel(d)
    vector = d{i};
    d_washout = cat(2,d_washout, {vector(1+ws:end)});
end

end