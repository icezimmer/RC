function [input_data, target_data] = dataLoader()

input_data = zeros(6, 480, 0);
target_data = zeros(7, 0); % one hot encoding for target

one_hot = eye(7);

opts = detectImportOptions(fullfile('AReM','bending1','dataset1.csv'));
var_names = {'avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'};
opts.SelectedVariableNames = var_names;

for num = 1:7
    T = readtable(fullfile('AReM','bending1', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,1));
end

for num = 1:6
    if num == 4 % dataset4.csv in Bending2 doesn't have commas
        opts_new = detectImportOptions(fullfile('AReM','bending2','dataset4.csv'));
        opts_new.VariableNames = [{'time'}, var_names];
        opts_new.SelectedVariableNames = var_names;
        T = readtable(fullfile('AReM','bending2', strcat('dataset', num2str(num), '.csv')),opts_new);
    else
        T = readtable(fullfile('AReM','bending2', strcat('dataset', num2str(num), '.csv')),opts);
    end
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,2));
end

for num = 1:15
    T = readtable(fullfile('AReM','cycling', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,3));
end

for num = 1:15
    T = readtable(fullfile('AReM','lying', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,4));
end

for num = 1:15
    T = readtable(fullfile('AReM','sitting', strcat('dataset', num2str(num), '.csv')),opts);
    if num == 8 % time step 13500 doesn't exist (line 60 / row 55)
        % insert the mean of the two adiacent rows
        T = [T(1:54,:);array2table((T{54,:}+T{55,:})/2, "VariableNames",var_names);T(55:end,:)];
    end
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,5));
end

for num = 1:15
    T = readtable(fullfile('AReM','standing', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,6));
end

for num = 1:15
    T = readtable(fullfile('AReM','walking', strcat('dataset', num2str(num), '.csv')),opts);
    ts = T{:,:}';
    input_data = cat(3,input_data,ts);
    target_data = cat(2,target_data,one_hot(:,7));
end

end
