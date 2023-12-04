function pooler = finalHiddenState(obj, input_data)
            
            % Flat the input data
            flatten = @(x) reshape(x, [], 1);
            input_data = cellfun(flatten, input_data, 'UniformOutput', false);

            % Cast to double
            input_data = cellfun(@double, input_data, 'UniformOutput', false);

            input_data_mat = cell2mat(input_data');

            % Compute initial hidden_state_0
            hidden_mat = obj.InputNetwork(input_data_mat);
            % Compute new hidden_states
            for t=1:obj.TimeSteps
                %hidden_mat(:,:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t), obj.OdeFunction, obj.StepSize);
                %hidden_mat(:,:,t+1) = hidden_mat(:,:,t) + obj.StepSize * obj.OdeFunction(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t));
                hidden_mat = hidden_mat + obj.StepSize * obj.OdeFunction(hidden_mat);
            end

            % Convert to cell array
            pooler = arrayfun(@(k) squeeze(hidden_mat(:,k)), 1:size(hidden_mat,2), 'UniformOutput', false);
            % Transpose the cell array to get num_samples x 1 shape
            pooler = pooler';
        end