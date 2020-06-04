BASE_DIR ="DTLZ2_TEST";
for i = 1:20
    [Xpareto,Ypareto,X,Y,XparetoGP,YparetoGP,hypf] = dtlz2_test();
    
    dir = BASE_DIR+ '/experiment_'+i+"/";
    mkdir(dir)
    csvwrite(dir+"Xpareto.csv", Xpareto)
    csvwrite(dir+"Ypareto.csv", Ypareto)
    csvwrite(dir+"X.csv", X)
    csvwrite(dir+"Y.csv", Y)
    csvwrite(dir+"XparetoGP.csv", XparetoGP);
    csvwrite(dir+"YparetoGP.csv", YparetoGP);
    csvwrite(dir+"hypf.csv", hypf);
end

function [Xpareto,Ypareto,X,Y,XparetoGP,YparetoGP,hypf] = dtlz2_test()
    f = @dtlz2;

    no_outputs = 2;               % number of objectives
    no_inputs  = 6;               % number of decision variables
    lb = zeros(1,6);            % define lower bound on decision variables, [lb1,lb2,...]
    ub = ones(1,6);            % define upper bound on decision variables, [ub1,ub2,...]


    dataset_size = 5*no_inputs;             % initial dataset size
    X = lhsdesign(dataset_size,no_inputs); % Latin hypercube design
    Y = zeros(dataset_size,no_outputs);     % corresponding matrix of response data
    for k = 1:size(X,1)
        X(k,:) = X(k,:).*(ub-lb)+lb;        % adjustment of bounds
        Y(k,:) = f(X(k,:));                 % calculation of response data
    end

    opt = TSEMO_options;             % call options for solver, see TSEMO_options file to adjust
    opt.maxeval = 100;               % number of function evaluations before termination
    opt.NoOfBachSequential = 1;      % number of function evaluations per iteration
    % Total number of iterations = opt.maxeval/opt.NoOfBachSequential
    
    [Xpareto,Ypareto,X,Y,XparetoGP,YparetoGP,hypf] = TSEMO_V3(f,X,Y,lb,ub,opt);
end