function [loss, accuracy_K, accuracy, accuracy_av, F1, F1_macro, support] = evaluation(target,predict)
one_hot = eye(7);
if strcmp(whos('target').class,'categorical')
    target = double(target);
    target = one_hot(:,target);
end
if strcmp(whos('predict').class,'categorical')
    predict = double(predict);
    predict = one_hot(:,predict);
end
if strcmp(whos('target').class,'cell')
    target = [target{:}];
    target = one_hot(:,target);
end
if strcmp(whos('predict').class,'cell')
    predict = [predict{:,:}];
    predict = one_hot(:,predict);
end


loss = immse(target, predict);
accuracy_K = nnz(min(target==predict,[],1)) / size(target,2);

[classes_target, ~] = find(target);
[classes_predict, ~] = find(predict);

if length(unique(classes_target)) < 7
    error('Not all classes are present in the target, Try to change seed_shuffle!')
else
    confusionMatrix = confusionmat(classes_target, classes_predict);
    
    accuracy = zeros(7,1);
    precision = zeros(7,1);
    recall = zeros(7,1);
    support = zeros(7,1);
    for i = 1:7
        index_target_i = find(classes_target == i);
        index_predict_i = find(classes_predict == i);
    
        precision(i) = length(intersect(index_target_i, index_predict_i)) / length(index_predict_i);
        recall(i) = length(intersect(index_target_i, index_predict_i)) / length(index_target_i);
    
        minor = confusionMatrix; minor(i,:) = []; minor(:,i) = [];
        accuracy(i) = (confusionMatrix(i,i) + sum(minor, "all")) / sum(confusionMatrix, "all");
    
        support(i) = length(index_target_i);
    end
    
    accuracy_av = mean(accuracy);
    precision_av = mean(precision);
    recall_av = mean(recall);
    F1 = 2 * ((precision .* recall) ./ (precision + recall));
    F1_macro = 2 * ((precision_av * recall_av) / (precision_av + recall_av));
end

end