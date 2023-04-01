function d = washout(d,ws)

d = reshape(d, [size(d,1), 480, size(d,2)/480]);
d = d(:,1+ws:480,:);
d = reshape(d,[size(d,1),(480-ws)*size(d,3)]);
end