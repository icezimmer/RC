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
        Spectrum
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
        function obj = NeuralODE(omega_b, Nh, ts, f, phi, eps, eigs, ws, lambda_r, seed)
            obj.BiasScaling = omega_b;
            obj.NeuronsNumber = Nh;
            obj.TimeSteps = ts;
            obj.OdeFunction = f;
            obj.NumericalMethod = phi;
            obj.StepSize = eps;
            obj.Spectrum = eigs;
            obj.Transient = ws;
            obj.Regularization = lambda_r;
            %obj.LayersNumber = Nl;
            obj.Seed = seed;
            obj.Bias = bias(Nh, omega_b, seed);
            obj.HiddenWeights = continuousStateMatrix(eigs, seed);
            %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
            obj.OutputWeights = [];
        end


        function [hidden, hidden_washout, pooler] = hiddenState(obj, input_data)
            hidden = cell(size(input_data,1), 1);
            hidden_washout = cell(size(input_data,1), 1);
            pooler = cell(size(input_data,1), 1);
            %hidden = cell(size(input_data,1), obj.LayersNumber);
            %hidden_washout = cell(size(input_data,1), obj.LayersNumber);
            
%             odefun = @(t,y) tanh(obj.Bias + obj.HiddenWeights*y);

            num_samples = size(input_data,1);
            for index_sample=1:num_samples
                input_sample = input_data{index_sample};

%                 if size(input_sample, 2) > 1
%                     input_sample = input_sample(:);
%                 end
                input_sample = inputAugmentation(input_sample, obj.NeuronsNumber);

                hidden_sample = zeros(obj.NeuronsNumber, 1+obj.TimeSteps);
                hidden_sample(:,1) = double(input_sample);
                for t=1:obj.TimeSteps
                    hidden_sample(:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_sample(:,t), obj.OdeFunction, obj.StepSize);
                end
                
%                 tspan = linspace(0,1,obj.TimeSteps+1);
%                 solution = ode45(odefun,tspan,double(input_sample));
%                 hidden_sample = solution.y;

                pooler{index_sample, 1} = hidden_sample(:,end);
                hidden{index_sample,1} = hidden_sample;
                % Discard the transient
                hidden_sample = hidden_sample(:, obj.Transient+1:end);
                hidden_washout{index_sample,1} = hidden_sample;
            end
        end


        function [obj, prediction, pooler] = fit(obj, input_data, target_data)
            [~,hidden_washout,pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');
            hidden_washout_mat = cell2mat(hidden_washout');
            target_data_t = target_data';

            target_data_mat_start = onehotencode(target_data_t, 1);
            target_data_mat = zeros(size(target_data_mat_start,1),0);
            sizes = cellfun(@size,hidden_washout,'UniformOutput',false);
            for index = 1:numel(hidden_washout)
                target_data_mat = cat(2,target_data_mat,target_data_mat_start(:,index)*ones(1,sizes{index}(2)));
            end

            obj.OutputWeights = trainOffline(hidden_washout_mat, target_data_mat, obj.Regularization);
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