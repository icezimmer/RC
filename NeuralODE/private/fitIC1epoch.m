function obj = fitIC1epoch(obj, input_data, target_data)
    %[~, ~, pooler] = hiddenState(obj, input_data);
    pooler = finalHiddenState(obj, input_data);
    pooler_mat = cell2mat(pooler');
    target_data_mat = onehotencode(target_data', 1);


    obj = trainOutputNetworkOffline(obj, pooler_mat, target_data_mat);

    % Define the adjoint ODE: da(t)/dt = (-a(t)'*df/dh)'
    %adjoint_ODE = @(x) -x'*obj.HiddenWeights;
    adjoint_ODE = @(x) -obj.HiddenWeights'*x;

    % Compute the initial condition a(T):
    % L(y) = (y - d)' * (y - d), where y = f_out(h(T)) = OuputWeights * [h(T); 1]
    % Supposing that L depends only on f_out(h(T)) and not on all h(t)
    % a(T) = dL/dh(T) = ((y - d)' * OutputWeights(:,1:end-1))' =
    % = (OutputWeights(:,1:end-1))' * (y - d)
    % We leave the last column to leave the bias weights
    adjoint_mat_T = (obj.OutputWeights(:,1:end-1))' * (obj.OutputNetwork(pooler_mat) - target_data_mat);

    % Discretization of the solution a(t): from a(T) to a(0)
    %adjoint_mat = zeros([length(obj.HiddenWeights), length(target_data), obj.TimeSteps]);
    %adjoint_mat(:,:,1) = adjoint_mat_T;
    adjoint_mat = adjoint_mat_T;
    % Using Euler method in reverse
    for t=1:obj.TimeSteps
        %adjoint_mat(:,:,t+1) = adjoint_mat(:,:,t) - obj.StepSize * (adjoint_ODE(adjoint_mat(:,:,t)))';
        %adjoint_mat = adjoint_mat - obj.StepSize * (adjoint_ODE(adjoint_mat))';
        %adjoint_mat = obj.NumericalMethod(adjoint_ODE, adjoint_mat); but
        %must be backward
        adjoint_mat = adjoint_mat - obj.StepSize * adjoint_ODE(adjoint_mat);
    end

    % Learn IC
    % Upgrade the initial condition h(0) by SGD:
    % h(0)_new = h(0)_old - dL/dh(0) = h(0)_old - a(0)
    % WARNING!! Inputnetwork is defined at the start and it has size hidden_dim*N
    % L'apprendimento della condizione iniziale è diversa per ciascun input
    % Quindi se apprendiamo le IC del training set
    % non possiamo generalizzare l'apprendimento alle IC del validation set
    %obj.InputNetwork = @(x) obj.InputNetwork(x) - adjoint_mat(:,:,end);
    obj.InputNetwork = @(x) obj.InputNetwork(x) - adjoint_mat;
end
