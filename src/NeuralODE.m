classdef NeuralODE
    %UNTITLED Summary of this class goes here
    %   Detailed eobj.Hiddenplanation goes here

    properties
        BiasScaling
        NeuronsNumber
        TimeSteps
        OdeFunction
        NumericalMethod
        StepSize
        EigenValues
        Density
        Transient
        Regularization
        %LayersNumber
        Seed
        Bias
        HiddenWeights
        %HiddenHiddenWeights
        OutputWeights
    end

    methods
        function obj = NeuralODE(omega_b, Nh, ts, f, phi, eps, eigs, dns, ws, lambda_r, seed)
            %if a < 0 || a > 1
            %    error('The parameter a must be in [0, 1]')
            %else
            obj.BiasScaling = omega_b;
            obj.NeuronsNumber = Nh;
            obj.TimeSteps = ts;
            obj.OdeFunction = f;
            obj.NumericalMethod = phi;
            obj.StepSize = eps;
            obj.EigenValues = eigs;
            obj.Density = dns;
            obj.Transient = ws;
            obj.Regularization = lambda_r;
            %obj.LayersNumber = Nl;
            obj.Seed = seed;
            obj.Bias = bias(Nh, omega_b, seed);
            obj.HiddenWeights = continuousStateMatrix(Nh, eigs, seed);
            %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
            obj.OutputWeights = [];
            %end
        end


%         function [hidden, hidden_washout, pooler] = hiddenState(obj, input_data)
%             hidden = cell(size(input_data,1), 1);
%             hidden_washout = cell(size(input_data,1), 1);
%             pooler = cell(size(input_data,1), 1);
%             %hidden = cell(size(input_data,1), obj.LayersNumber);
%             %hidden_washout = cell(size(input_data,1), obj.LayersNumber);
% 
%             num_samples = size(input_data,1);
%             for index_sample=1:num_samples
%                 input_sample = input_data{index_sample};
%                 if length(size(input_sample)) > 1
%                     input_sample = input_sample(:);
%                 end
%                 hidden_sample = zeros(obj.NeuronsNumber, 1+obj.TimeSteps);
%                 hidden_sample(:,1) = double(input_sample);
%                 for t=1:obj.TimeSteps
%                     hidden_sample(:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_sample(:,t), obj.OdeFunction, obj.StepSize);
%                 end
%                 hidden{index_sample,1} = hidden_sample;
%                 pooler{index_sample, 1} = hidden_sample(:,end);
%                 % Discard the transient
%                 hidden_sample = hidden_sample(:, obj.Transient+1:end);
%                 hidden_washout{index_sample,1} = hidden_sample;
%             end
%         end

        function pooler = hiddenState(obj, input_data)
            pooler = cell(size(input_data,1), 1);
            %hidden = cell(size(input_data,1), obj.LayersNumber);
            %hidden_washout = cell(size(input_data,1), obj.LayersNumber);

            num_samples = size(input_data,1);
            for index_sample=1:num_samples
                input_sample = input_data{index_sample};
                if length(size(input_sample)) > 1
                    input_sample = input_sample(:);
                end
                pooler_sample = double(input_sample);
                for t=1:obj.TimeSteps
                    pooler_sample = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, pooler_sample, obj.OdeFunction, obj.StepSize);
                end
                pooler{index_sample, 1} = pooler_sample;
            end
        end

        function [obj, prediction, pooler] = fit(obj, input_data, target_data)
            %[~,hidden_washout,pooler] = hiddenState(obj, input_data);
            pooler = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');
            %hidden_washout_mat = cell2mat(hidden_washout');
            %one_hot = eye(num_classes);
            target_data_t = target_data';
            %categories(target_data)
            target_data_mat = onehotencode(target_data_t, 1);
            %target_data_mat = kron(target_data_mat, ones(1+obj.TimeSteps-obj.Transient));
            %size(target_data_mat)
            %target_data_mat = one_hot(:,[target_data_t{:}]);

            %obj.OutputWeights = trainOffline(hidden_washout_mat, target_data_mat, obj.Regularization, 0);
            obj.OutputWeights = trainOffline(pooler_mat, target_data_mat, obj.Regularization, 0);
            output_mat = readout(pooler_mat, obj.OutputWeights);
            [~, prediction_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            prediction = zeros(num_samples,1);
            for index_sample=1:num_samples
                prediction_sample = prediction_mat(index_sample);
                prediction(index_sample) = prediction_sample;
            end
            prediction = categorical(prediction-1,0:9);
        end


        function [prediction, hidden] = classify(obj, input_data)
            hidden = hiddenState(obj, input_data);
            hidden_mat = cell2mat(hidden');

            output_mat = readout(hidden_mat, obj.OutputWeights);
            [~, prediction_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            prediction = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                prediction_sample = prediction_mat(prec+1:prec+obj.TimeSteps+1);
                prediction{index_sample} = categorical(prediction_sample);
                prec = prec + obj.TimeSteps+1;
            end
        end

    end

end