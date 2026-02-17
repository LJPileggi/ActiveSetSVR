classdef SVR < handle
    properties
        alpha1    %alpha+
        alpha2    %alpha-
        alpha     %alpha+ and alpha-
        gamma     %sum of alpha+ - alpha-
        X         %our data
        Y         %our labels
        N         %space dimension
        Ker       %kernel matrix
        q
        Q
        A
        A_eq
        b
        b_eq
        C
        beta      %exponent constant for kernel
        eps       %insensitivity tube thickness
        init      %initialisation mode for alpha
    end
    methods
        function obj = SVR(X, Y, C, eps, beta, init, perc_low, perc_upp, idx_low, idx_upp)

            obj.X = X;
            obj.Y = Y;
            obj.C = C;
            obj.N = size(obj.X, 2);
            obj.eps = eps;
            obj.beta = beta;
            obj.set_ker();
            obj.q = [(-obj.Y + obj.eps).' ; (obj.Y + obj.eps).'];
            obj.Q = [obj.Ker -obj.Ker ; -obj.Ker obj.Ker];
	        obj.A = [-eye(obj.N) zeros(obj.N) ; zeros(obj.N) -eye(obj.N) ; eye(obj.N) zeros(obj.N) ; zeros(obj.N) eye(obj.N)];
            obj.A_eq = [ones(1, obj.N) (-ones(1, obj.N))];
            obj.b = [zeros(2*obj.N, 1) ; obj.C*ones(2*obj.N, 1)];
            obj.b_eq = 0;
            obj.init = init;
            if init == "standard"
                choice = randsample([0 obj.N], obj.N, true);
                choice_neg = obj.N - choice;
                idx = [1:obj.N] + choice;
                obj.alpha1 = obj.C/sum(choice_neg==0)*ones(obj.N, 1);
                obj.alpha2 = obj.C/sum(choice_neg==obj.N)*ones(obj.N, 1);
                obj.cat_alpha_gamma();
                obj.alpha(idx) = 0;
            elseif init == "perc"
                choice = randsample([0 obj.N], obj.N, true);
                choice_neg = obj.N - choice;
                idx = [1:obj.N] + choice;
                obj.alpha1 = 0.5*obj.C/sum(choice_neg==0)*ones(obj.N, 1);
                obj.alpha2 = 0.5*obj.C/sum(choice_neg==obj.N)*ones(obj.N, 1);
                obj.cat_alpha_gamma();
                obj.alpha(idx) = 0;
                if perc_low + perc_upp <= 0.5
                    idx_left = [1:2*obj.N];
                    idx_left = idx_left(~ismember(1:end, idx));%(idx_left ~= idx);
                    idx_left_shuffled = randperm(2*obj.N);
                    idx_left_shuffled = idx_left_shuffled(ismember(idx_left_shuffled, idx_left));
                    idx_l = idx_left_shuffled(1:round(2*obj.N*perc_low));
                    idx_u = idx_left_shuffled(round(2*obj.N*perc_low)+1:round(2*obj.N*perc_low)+round(2*obj.N*perc_upp));
                    obj.alpha(idx_l) = 0;
                    obj.alpha(idx_u) = obj.C;
                end
            elseif init == "fix"
                choice = randsample([0 obj.N], obj.N, true);
                choice_neg = obj.N - choice;
                idx = [1:obj.N] + choice;
                obj.alpha1 = obj.C/sum(choice_neg==0)*ones(obj.N, 1);
                obj.alpha2 = obj.C/sum(choice_neg==obj.N)*ones(obj.N, 1);
                obj.cat_alpha_gamma();
                obj.alpha(idx) = 0;
                if length(idx_low) + length(idx_upp) <= obj.N
                    obj.alpha(idx_low + choice_neg) = 0;
                    obj.alpha(idx_upp + choice_neg) = obj.C;
                end
            else
                error("Invalid initialisation method.");
            end
        end
        function obj = cat_alpha_gamma(obj)
            obj.alpha = [obj.alpha1 ; obj.alpha2];
            obj.gamma = obj.alpha1 - obj.alpha2;
        end
        function obj = update_alpha(obj, alpha_new)
            obj.alpha1 = alpha_new(1:end/2);
            obj.alpha2 = alpha_new(end/2+1:end);
            obj.cat_alpha_gamma();
        end
        function obj = set_ker(obj)
            obj.Ker = zeros(obj.N);
            for i = 1:obj.N
                for j = 1:obj.N
                    obj.Ker(i, j) = exp(-obj.beta*(sum((obj.X(:,i) - obj.X(:,j)).^2))/2);
                end
            end
        end
        function objval = objfunc(obj, alpha_new)
            objval = dot(obj.q, alpha_new) + alpha_new.'*obj.Q*alpha_new/2;
        end
        function grad = objgrad(obj, alpha)
            grad = obj.q + obj.Q*alpha;
        end
        function predict = eval_predict(obj, x)
            predict = zeros(1, size(x, 2));
            for i = 1:size(predict, 2)
                predict(i) = dot(obj.gamma, exp(-obj.beta*sum((obj.X - x(:,i)).^2, 1)/2));
            end
            support_idx = find((obj.alpha1 > 0) & (obj.alpha1 < obj.C));
            if ~isempty(support_idx)
                i = support_idx(1);
                bias = dot(obj.gamma, exp(-obj.beta*sum((obj.X - obj.X(:,i)).^2, 1)/2)) - obj.Y(i) - obj.eps;
            else
                support_idx = find((obj.alpha2 > 0) & (obj.alpha2 < obj.C));
                if ~isempty(support_idx)
                    i = support_idx(1);
                    bias = -(dot(obj.gamma, exp(-obj.beta*sum((obj.X - obj.X(:,i)).^2, 1)/2)) - obj.Y(i) - obj.eps);
                else
                    bias = 0;
                end
            end
            predict = predict - bias;
        end
        function loss = HingeLoss(obj, x, y)
            abs_vec = abs(y - obj.eval_predict(x)) - obj.eps;
            loss = sum(max([abs_vec ; zeros(size(abs_vec))], [], 1))/length(abs_vec) - 0.5 * obj.gamma.' * obj.Ker * obj.gamma;
        end
        function loss = MSE(obj, x, y)
            loss = sum((y - obj.eval_predict(x)).^2)/2;
        end
        
        function [f_s, time_t, it_cost, k, func_val_l, n_low, n_upp, it_cost_l] = ActiveSet(obj, maxiter, verbose, return_info)
            start_t = cputime;
            it_cost = 0;
            prec = 1e-14*obj.C;
            warning('off','all')
            lastwarn('', '');
            %initialise indexes of variables touching the constraints
            low = obj.alpha <= prec;
            upp = obj.alpha >= obj.C - prec;
            free = ~(low | upp);

            %initialise variables
            x = obj.alpha;
            f_s = obj.objfunc(x);
            k = 0;
            print_every = 100;
            termination_flag = false;
            if return_info
                func_val_l = zeros(maxiter, 1);
                n_low = zeros(maxiter, 1);
                n_upp = zeros(maxiter, 1);
                it_cost_l = zeros(maxiter, 1);
            end

            print_info = (fix(k/print_every) == 0);
            
            while true
                start_t_it = cputime;
                if verbose & print_info
                    fprintf("Iteration n. %d\n", k)
                end

                x_new = zeros(size(x, 1), 1);
                x_new(upp) = obj.C;
                %constant term in equality constraint;
                %have to set the alphas to C
                C_eq = -obj.C*sum(obj.A_eq(upp));
                KKT = [obj.Q(free, free) obj.A_eq(1, free).' ; obj.A_eq(1, free) 0];
                b_KKT = [-( obj.q(free) + obj.Q(free, upp) * obj.C*ones(size(find(upp), 1), 1) ) ; C_eq];
                sol = KKT \ b_KKT;
                %detects warning of singular matrix
                [warnMsg, ~] = lastwarn;
                if isempty(warnMsg)
                    x_new(free) = sol(1:end-1);
                else
                    Z = null(obj.Q(free, free));
                    Z_orth = Z - (obj.A_eq(free) * Z) .* repmat(obj.A_eq(free).',1, size(Z, 2)) / dot(obj.A_eq(free).', obj.A_eq(free).');
                    d = sum((obj.q(free).' * Z_orth) .* Z_orth ./ diag(Z_orth.' * Z_orth).', 2);
                    g = obj.objgrad(x);
                    g = g - dot(obj.A_eq, g)*obj.A_eq.'/sum(obj.A_eq.^2);
                    d = - d.' * g(free) / abs(d.' * g(free)) * d;
                    x_new(free) = x(free) + d;
                    lastwarn('', '');
                end

                %checking no variable touches constraints not in the active
                %set
                if all (x_new(free) >= -prec & x_new(free) <= obj.C + prec)
                    

                    %evaluate gradient
                    if isnan(sol(end)) | isinf(sol(end))
                        grad = obj.objgrad(x_new);
                        g = obj.objgrad(x_new).*(low - upp);
                        g = g - (rand*(mean(maxk(grad, 2)) - mean(mink(grad, 2))) + mean(mink(grad, 2)))*(obj.A_eq.').*(low - upp);
                    else
                        g = obj.objgrad(x_new).*(low - upp) + sol(end)*(obj.A_eq.').*(low - upp);
                    end

                    %find indexes on lower and upper bound with gradient of
                    %opposite sign (multiplier ~ gradient)
                    low_opp = low & g < -prec;
                    upp_opp = upp & g < -prec;

                    if sum(low_opp) == 0 && sum(upp_opp) == 0
                        fprintf("Minimum found. Terminate\n");
                        termination_flag = true;
                    else
                        if verbose & print_info
                            fprintf("remove constraint\n");
                        end

                        [~, idx_rmv] = max((low_opp | upp_opp) .* abs(g));

                        free(idx_rmv) = true;
                        low(idx_rmv) = false;
                        upp(idx_rmv) = false;
                    end
                %we are beyond the boundaries
                else
                    if verbose & print_info
                        fprintf("add constraint\n");
                    end
                    p = x_new - x;
                    idx_pos = free & p > prec;
                    a_max = min(( obj.C - x(idx_pos))./(p(idx_pos)));
                    idx_neg = free & p < -prec;
                    a_max = min([a_max min((-x(idx_neg))./(p(idx_neg)))]);
                    if isempty(a_max)
                        a_max = 0;
                    end

                    x_new = x + a_max*p;

                    %check which free variables are not such anymore
                    %(blocking constraints) and update active constraints
                    low_new = free & x_new <= prec;
                    upp_new = free & x_new >= obj.C - prec;

                    low(low_new) = true;
                    upp(upp_new) = true;
                    %set blocking constraint to not free anymore
                    free(low_new | upp_new) = false;

                end
                f_e = obj.objfunc(x_new);
                x = x_new;
                n_act = sum(low) + sum(upp);
                end_t_it = cputime - start_t_it;
                it_cost = it_cost + end_t_it;
                if verbose & (print_info | termination_flag)
                    fprintf("Function value: %f\n", f_e);
                    fprintf("n. of active constraints: %d;\n", n_act);
                    fprintf("of which %d lower and %d upper bound.\n", sum(low), sum(upp));
                    fprintf("Iteration cost: %f\n", end_t_it);
                end
                f_s = f_e;
                k = k + 1;
                print_info = (mod(k, print_every) == 0);
                if return_info
                    func_val_l(k) = f_s;
                    n_low(k) = sum(low);
                    n_upp(k) = sum(upp);
                    it_cost_l(k) = end_t_it;
                end
                if k > maxiter
                    break;
                end
                if termination_flag == true
                    break
                end
            end
            %update alphas in object
            obj.update_alpha(x);
            time_t = cputime - start_t;
            it_cost = it_cost/k;
            if verbose & print_info
                fprintf("Elapsed time: %f\n", time_t)
            end
            if return_info
                func_val_l = func_val_l(1:k);
                n_low = n_low(1:k);
                n_upp = n_upp(1:k);
                it_cost_l = it_cost_l(1:k);
                it_cost_l(it_cost_l < prec) = prec;
            end
        end
    end
end