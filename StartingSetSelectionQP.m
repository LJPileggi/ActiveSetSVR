function [combs, f_l, time_l, it_cost_l, n_it_l] = StartingSetSelectionQP(N, x_l, x_u, eps, C, e, beta)
    [x, y] = datagen(N, x_l, x_u, eps);
    perc_l = [0 0.1 0.2 0.3 0.4 0.5];
    perc_u = [0 0.1 0.2 0.3 0.4 0.5];
    n_comb = 21;
    combs = zeros(n_comb, 2);
    f_l = zeros(n_comb, 2);
    time_l = zeros(n_comb, 2);
    it_cost_l = zeros(n_comb, 2);
    n_it_l = zeros(n_comb, 2);
    n_attemps = 10;
    i = 1;
    for j = 1:length(perc_l)
        for k = 1:length(perc_u)
            if (perc_l(j) + perc_u(k) <= 0.5)
                fprintf("Lower bound perc: %f; upper bound: %f;\n", perc_l(j), perc_u(k));
                time_t_mean = 0;
                time_t_std = 0;
                f_e_mean = 0;
                f_e_std = 0;
                it_cost_mean = 0;
                it_cost_std = 0;
                n_it_mean = 0;
                n_it_std = 0;
                for n = 1:n_attemps
                    svr = SVR(x, y, C, e, beta, "perc", perc_l(j), perc_u(k));
                    ops = optimoptions('fmincon');
                    ops.Algorithm = 'active-set';
                    ops.MaxIterations = round(N*10000);
                    start_t = cputime;
                    [~, f_e_k, ~, output] = quadprog(svr.Q, svr.q, svr.A, svr.b, svr.A_eq, svr.b_eq, [], [], svr.alpha, ops);
                    time_t_k = cputime - start_t;
                    n_it_k = output.iterations;
                    it_cost_k = time_t_k/n_it_k;
                    time_t_mean = time_t_mean + time_t_k/n_attemps;
                    time_t_std = time_t_std + time_t_k^2/n_attemps;
                    f_e_mean = f_e_mean + f_e_k/n_attemps;
                    f_e_std = f_e_std + f_e_k^2/n_attemps;
                    it_cost_mean = it_cost_mean + it_cost_k/n_attemps;
                    it_cost_std = it_cost_std + it_cost_k^2/n_attemps;
                    n_it_mean = n_it_mean + n_it_k/n_attemps;
                    n_it_std = n_it_std + n_it_k^2/n_attemps;
                    fprintf("N. of iterations: %d;\n", n_it_k);
                    fprintf("Average cost per iteration: %f\n", it_cost_k);
                    fprintf("Total compuational cost: %f\n", time_t_k);
                    fprintf("Function value: %f\n\n", f_e_k);
                end
                combs(i,:) = [perc_l(j), perc_u(k)];
                time_t_std = (time_t_std - time_t_mean^2)^0.5;
                f_e_std = (f_e_std - f_e_mean^2)^0.5;
                it_cost_std = (it_cost_std - it_cost_mean^2)^0.5;
                n_it_std = (n_it_std - n_it_mean^2)^0.5;
                time_l(i, :) = [time_t_mean ; time_t_std];
                it_cost_l(i, :) = [it_cost_mean ; it_cost_std];
                n_it_l(i, :) = [n_it_mean ; n_it_std];
                f_l(i, :) = [f_e_mean ; f_e_std];
                i = i + 1;
            end
        end
    end
end