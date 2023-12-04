classdef NeuralODE
    % Data are taken in input and given in output as cell array of size
    % [num_sample * 1], where each cell is an array of size [dim * 1] and
    % dim is the dimension of the space.
    % The trajectories (in hidden space) are tensor of size [dim * num_samples * time_steps],
    % where dim is the dimension of the space
    % The label of target data for classification are {1,...,num_classes}

    properties
        % Input Network attributes
        InputNetwork
        % Hidden Network attributes
        HiddenSize
        BiasScaling
        TimeSteps
        OdeFunction
        NumericalMethod
        StepSize
        Spectrum
        Transient
        %LayersNumber
        Seed
        Bias
        HiddenWeights
        %HiddenHiddenWeights
        % Output Network attributes
        Regularization
        OutputWeights
        OutputNetwork
    end

    methods
        %function obj = NeuralODE(Nz, omega_b, T, f, phi, eps, eigs, ws, lambda_r, seed)
        function obj = NeuralODE(Nz, omega_b, T, eps, eigs, ws, lambda_r, seed)
            obj.InputNetwork = @(x) inputAugmentation(x, Nz);
            obj.HiddenSize = Nz;
            obj.BiasScaling = omega_b;
            obj.TimeSteps = floor(T/eps);
            %obj.OdeFunction = f;
            %obj.NumericalMethod = phi;
            obj.StepSize = eps;
            obj.Spectrum = eigs;
            obj.Transient = ws;
            %obj.LayersNumber = Nl;
            obj.Seed = seed;
            % We set bias and hidden weights at the start (fixed parameters)
            obj.Bias = bias(Nz, omega_b, seed);
            obj.HiddenWeights = continuousStateMatrix(eigs, seed);
            obj.OdeFunction = @(z) obj.HiddenWeights * z + obj.Bias;
            %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
            obj.Regularization = lambda_r;
            obj.OutputWeights = [];
            obj.OutputNetwork = [];
        end


        function [hidden, hidden_washout, pooler] = hiddenState(obj, input_data)

            num_samples = length(input_data);
            
            % Flat the input data and cast to double
            flatten = @(x) reshape(x, [], 1);
            input_data = cellfun(flatten, input_data, 'UniformOutput', false);
            input_data = cellfun(@double, input_data, 'UniformOutput', false);

            input_data_mat = cell2mat(input_data');

            hidden_mat = zeros([obj.HiddenSize, num_samples, obj.TimeSteps]);
            hidden_mat_0 = obj.InputNetwork(input_data_mat);

            hidden_mat(:,:,1) = hidden_mat_0;
            for t=1:obj.TimeSteps
                %hidden_mat(:,:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t), obj.OdeFunction, obj.StepSize);
                %hidden_mat(:,:,t+1) = hidden_mat(:,:,t) + obj.StepSize * obj.OdeFunction(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t));
                hidden_mat(:,:,t+1) = hidden_mat(:,:,t) + obj.StepSize * obj.OdeFunction(hidden_mat(:,:,t));
            end

            hidden_washout_mat = hidden_mat(:,:,obj.Transient+1:end);
            pooler_mat = hidden_mat(:,:,end);
            % Convert to cell array
            hidden = arrayfun(@(k) squeeze(hidden_mat(:,k,:)), 1:size(hidden_mat,2), 'UniformOutput', false);
            hidden_washout = arrayfun(@(k) squeeze(hidden_washout_mat(:,k,:)), 1:size(hidden_washout_mat,2), 'UniformOutput', false);
            pooler = arrayfun(@(k) squeeze(pooler_mat(:,k,:)), 1:size(pooler_mat,2), 'UniformOutput', false);
            % Transpose the cell array to get num_samples x 1 shape
            hidden = hidden';
            hidden_washout = hidden_washout';
            pooler = pooler';


            % hidden = cell(size(input_data,1), 1);
            % hidden_washout = cell(size(input_data,1), 1);
            % pooler = cell(size(input_data,1), 1);
            % % hidden = cell(size(input_data,1), obj.LayersNumber);
            % % hidden_washout = cell(size(input_data,1), obj.LayersNumber);
            % 
            % num_samples = size(input_data,1);
            % for index_sample=1:num_samples
            %     input_sample = input_data{index_sample};
            % 
            %     % if size(input_sample, 2) > 1
            %     %     input_sample = input_sample(:);
            %     % end
            %     input_sample = double(input_sample);
            %     hidden_sample_0 = obj.InputNetwork(input_sample);
            % 
            %     obj.Bias = bias(length(hidden_sample_0), obj.BiasScaling, obj.Seed);
            %     hidden_sample = zeros(length(hidden_sample_0), 1+obj.TimeSteps);
            % 
            %     hidden_sample(:,1) = hidden_sample_0;
            %     for t=1:obj.TimeSteps
            %         hidden_sample(:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_sample(:,t), obj.OdeFunction, obj.StepSize);
            %     end
            % 
            %     pooler{index_sample, 1} = hidden_sample(:,end);
            %     hidden{index_sample,1} = hidden_sample;
            %     % Discard the transient
            %     hidden_sample = hidden_sample(:, obj.Transient+1:end);
            %     hidden_washout{index_sample,1} = hidden_sample;
            % end
        end

        function classification = classify(obj, input_data)
            pooler = finalHiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');

            output_mat = obj.OutputNetwork(pooler_mat);
            [~, classification_mat] = max(output_mat, [], 1);
            classification = num2cell(classification_mat');
            classification = cellfun(@categorical, classification, 'UniformOutput', false);

        end

        function confusionMatrix(obj, input_data, target_data)
            if(isa(target_data,"categorical"))
                % Predict on input data
                predicted_data = obj.classify(input_data);

                % Convert to {1, ..., num_classes} categories
                target_data = grp2idx(target_data);
                num_classes = length(unique(target_data));
                labels = arrayfun(@num2str, 1:num_classes, 'UniformOutput', false);
                target_data = categorical(target_data, 1:num_classes, labels);

                % Plot Confusion Matrix
                figure
                confusionchart(target_data, [predicted_data{:}]);
            end
        end

        function regression = predict(obj, input_data)
            pooler = finalHiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');

            output_mat = obj.OutputNetwork(pooler_mat);
            regression = num2cell(output_mat, 1)';

        end    

        function best_net = fitIC2classify(obj, input_data, target_data)
            % Convert to {1, ..., num_classes} categories
            target_data = grp2idx(target_data);
            num_classes = length(unique(target_data));
            labels = arrayfun(@num2str, 1:num_classes, 'UniformOutput', false);
            target_data = categorical(target_data, 1:num_classes, labels);

            accuracy = @(y,d) 100 * sum(y == d) / length(d);

            % We must fit at leat 1 epoch before to classify because the
            % output weights matrix and the output network initially are [] 
            obj = fitIC1epoch(obj, input_data, target_data);
            best_net = obj;

            output_data = classify(obj, input_data);
            output_data = [output_data{:}];

            accuracy_new = accuracy(output_data(:), target_data(:));
            disp(['Accuracy: ', num2str(accuracy_new), '%'])

            count = 0;
            max_accuracy = accuracy_new;
            % Training using Early Stopping
            while(count < 5)
                accuracy_old = accuracy_new;
                obj = fitIC1epoch(obj, input_data, target_data);
                output_data = classify(obj, input_data);
                output_data = [output_data{:}];
                accuracy_new = accuracy(output_data(:), target_data(:));
                if(accuracy_new - accuracy_old > 0)
                    count = 0;
                    if(accuracy_new > max_accuracy)
                        best_net = obj;
                        max_accuracy = accuracy_new;
                    end
                else
                    count = count + 1;
                end
                disp(['Accuracy: ', num2str(accuracy_new), '%'])
            end
        end


        function obj = fitODE(obj, input_data, target_data)
            [hidden, ~, pooler] = hiddenState(obj, input_data);
            hidden_mat = cell2mat(hidden');
            pooler_mat = cell2mat(pooler');
            target_data_mat = onehotencode(target_data', 1);


            obj = trainOutputNetworkOffline(obj, pooler_mat, target_data_mat);

            % Define the adjoint ODE: da(t)/dt = -a(t)'*df/dh
            adjoint_ODE = @(x) -x'*obj.HiddenWeights;

            % Compute the initial condition a(T):
            % L(y) = (y - d)' * (y - d), where y = OuputWeights * [h(T); 1]
            % a(T) = dL/dh(T) = ((y - d)' * OutputWeights(:,1:end-1))' =
            % = (OutputWeights(:,1:end-1))' * (y - d)
            % We leave the last column to leave the bias weights
            adjoint_mat_T = (obj.OutputWeights(:,1:end-1))' * (obj.OutputNetwork(pooler_mat) - target_data_mat);

            % Discretization of the solution a(t): from a(T) to a(0)
            adjoint_mat = zeros([length(obj.HiddenWeights), length(target_data), obj.TimeSteps]);
            adjoint_mat(:,:,1) = adjoint_mat_T;
            % Using Euler method in reverse
            for t=1:obj.TimeSteps
                adjoint_mat(:,:,t+1) = adjoint_mat(:,:,t) - obj.StepSize * (adjoint_ODE(adjoint_mat(:,:,t)))';
            end

            % Learn ODE
            % Compute dL/dTheta, where Theta are the parameters of ODE f
            % dL/dTheta(t) = dL/dTheta(0) + Integral(a(t)' * df/dTheta, dt, [0,t])
            % = dL/dTheta(T) + Integral(-a(t)' * df/dTheta, dt, [T,t])
            % = Integral(-a(t)' * df/dTheta, dt, [T,t]), being IC = 0
            % So dL/dTheta(0) = Integral(-a(t)' * df/dTheta, dt, [T,0])
            % We approximate it by the trapezoidal rule
            % dL/dTheta \approx a(t)' * df/dTheta = a(t)' * h(t)
            % supposing f(t,h(t),Theta) = HiddenWeights * h(t) + bias.
            %dL_dTheta = pagemtimes(permute(adjoint_mat, [2, 1, 3]), hidden_mat, 'transpose', 'none');
            
        end


        function obj = fitIVP(obj, input_data, target_data)
            [~, ~, pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');
            target_data_mat = onehotencode(target_data', 1);


            obj = trainOutputNetworkOffline(obj, pooler_mat, target_data_mat);

            % Define the adjoint ODE: da(t)/dt = -a(t)'*df/dh
            adjoint_ODE = @(x) -x'*obj.HiddenWeights;

            % Compute the initial condition a(T):
            % L(y) = (y - d)' * (y - d), where y = OuputWeights * [h(T); 1]
            % a(T) = dL/dh(T) = ((y - d)' * OutputWeights(:,1:end-1))' =
            % = (OutputWeights(:,1:end-1))' * (y - d)
            % We leave the last column to leave the bias weights
            adjoint_mat_T = (obj.OutputWeights(:,1:end-1))' * (obj.OutputNetwork(pooler_mat, obj.OutputWeights) - target_data_mat);

            % Discretization of the solution a(t): from a(T) to a(0)
            adjoint_mat = zeros([length(obj.HiddenWeights), length(target_data), obj.TimeSteps]);
            adjoint_mat(:,:,1) = adjoint_mat_T;
            % Using Euler method in reverse
            for t=1:obj.TimeSteps
                adjoint_mat(:,:,t+1) = adjoint_mat(:,:,t) - obj.StepSize * (adjoint_ODE(adjoint_mat(:,:,t)))';
            end

            % Learn ODE
            % Compute dL/dTheta, where Theta are the parameters of ODE f
            % dL/dTheta = Integral(a(t)' * df/dTheta, dt, [0,T])
            % We approximate it by the trapezoidal rule
            % dL/dTheta \approx a(t)' * df/dTheta = a(t)' * HiddenWeights

            % Learn IC
            % Upgrade the initial condition h(0) by SGD:
            % h(0)_new = h(0)_old - dL/dh(0) = h(0)_old - a(0)
            obj.InputNetwork = @(x) obj.InputNetwork(x) - adjoint_mat(:,:,end);
        end    

    end

end