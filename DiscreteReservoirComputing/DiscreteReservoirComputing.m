classdef DiscreteReservoirComputing
    % Data must be cell array of size [num_sample * 1],
    % where each cell is an array of size [dim * time_steps],
    % where dim is the dimension of the space

    properties
        InputDimension
        InputScaling
        BiasScaling
        NeuronsNumber
        InitialCondition
        SpectralRadius
        Density
        LeakyFactor
        Transient
        %LayersNumber
        Seed
        Bias
        InputWeights
        HiddenWeights
        %HiddenHiddenWeights
        Regularization
        OutputWeights
    end

    methods
        function obj = DiscreteReservoirComputing(Nu, omega_in, omega_b, Nh, x0, rho, dns, a, ws, lambda_r, seed)
            if a < 0 || a > 1
                error('The parameter a must be in [0, 1]')
            else
                obj.InputDimension = Nu;
                obj.InputScaling = omega_in;
                obj.BiasScaling = omega_b;
                obj.NeuronsNumber = Nh;
                obj.InitialCondition = x0;
                obj.SpectralRadius = rho;
                obj.Density = dns;
                obj.LeakyFactor = a;
                obj.Transient = ws;
                %obj.LayersNumber = Nl;
                obj.Seed = seed;
                obj.Bias = bias(Nh, omega_b, seed);
                obj.InputWeights = inputMatrix(Nu, omega_in, Nh, seed);
                obj.HiddenWeights = discreteStateMatrix(Nh, rho, seed, dns, a);
                %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
                obj.Regularization = lambda_r;
                obj.OutputWeights = [];
            end
        end


        function [hidden, hidden_washout, pooler] = hiddenState(obj, input_data)
            num_samples = size(input_data,1);
            
            pooler = cell(num_samples, 1);
            hidden = cell(num_samples, 1);
            hidden_washout = cell(num_samples, 1);
            %hidden = cell(num_samples, obj.LayersNumber);
            %hidden_washout = cell(num_samples, obj.LayersNumber);

            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                input_sample = input_data{index_sample};
               
                hidden_sample = zeros(obj.NeuronsNumber,time_steps+1);
                hidden_sample(:,1) = obj.InitialCondition;
                for t=1:time_steps
                    hidden_sample(:,t+1) = (1-obj.LeakyFactor)*hidden_sample(:,t) + obj.LeakyFactor*tanh(obj.InputWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,t));
                end

                pooler{index_sample,1} = hidden_sample(:,end);
                % Discard the initial state
                hidden_sample = hidden_sample(:, 2:end);
                hidden{index_sample,1} = hidden_sample;
                % Discard the transient
                hidden_sample = hidden_sample(:, obj.Transient+1:end);
                hidden_washout{index_sample,1} = hidden_sample;
            end

%             for layer=1:obj.LayersNumber-1
%                 for index_sample=1:num_samples
%                     time_steps = size(input_data{index_sample},2);
%                     % Add ones to input for bias
%                     input_sample = zeros(obj.NeuronsNumber+1,time_steps);
%                     input_sample(1:end-1,:) = hidden{index_sample,layer};
%                     input_sample(end,:) = ones(1, time_steps);
%                    
%                     hidden_sample = zeros(obj.NeuronsNumber,time_steps);
%                     hidden_sample(:,1) = obj.InitialCondition;
%                     for t=1:time_steps
%                         hidden_sample(:,t+1) = (1-obj.LeakyFactor)*hidden_sample(:,t) + obj.LeakyFactor*tanh(obj.HiddenHiddenWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,t));
%                     end
%                     % Discard the initial state
%                     hidden_sample = hidden_sample(:, 2:end);
%                     hidden{index_sample,layer+1} = hidden_sample;
%                     % Discard the transient
%                     hidden_sample = hidden_sample(:, obj.Transient+1:end);
%                     hidden_washout{index_sample,layer+1} = hidden_sample;
%                 end
%             end
        end

        function obj = fit(obj, input_data, target_data)
            [~, hidden_washout] = hiddenState(obj, input_data);
            hidden_washout_mat = cell2mat(hidden_washout');
            % hidden_mat = cell2mat(hidden');

            target_data_washout = washOut(obj, target_data');
            target_data_washout_mat = onehotencode([target_data_washout{:}], 1);

            obj = trainOutputWeightsOffline(obj, hidden_washout_mat, target_data_washout_mat);
        end

        function classification = classifySeq2Vec(obj, input_data)
            [~, ~, pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');

            output_mat = readOut(obj, pooler_mat);
            [~, classification_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            classification = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                % time_steps is equal to 1
                classification_sample = classification_mat(prec+1);
                classification{index_sample} = categorical(classification_sample);
                prec = prec + 1;
            end
        end

        function regression = predictSeq2Vec(obj, input_data)
            [~, ~, pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');

            output_mat = readOut(obj, pooler_mat);

            num_samples = size(input_data,1);
            regression = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                % time_steps is equal to 1
                regression{index_sample} = output_mat(prec+1);
                prec = prec + 1;
            end
        end        

        function classification = classifySeq2Seq(obj, input_data)
            hidden = hiddenState(obj, input_data);
            hidden_mat = cell2mat(hidden');

            output_mat = readOut(obj, hidden_mat);
            [~, classification_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            classification = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                classification_sample = classification_mat(prec+1:prec+time_steps);
                classification{index_sample} = categorical(classification_sample);
                prec = prec + time_steps;
            end
        end

        function regression = predictSeq2Seq(obj, input_data)
            hidden = hiddenState(obj, input_data);
            hidden_mat = cell2mat(hidden');

            output_mat = readOut(obj, hidden_mat);

            num_samples = size(input_data,1);
            regression = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                regression{index_sample} = output_mat(prec+1:prec+time_steps);
                prec = prec + time_steps;
            end
        end

    end

end