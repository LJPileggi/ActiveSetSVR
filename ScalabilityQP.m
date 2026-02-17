function [f_l, time_l, it_cost_l, n_it_l] = ScalabilityQP(x_l, x_u, eps, C, e, beta)
    N_l = [10 50 100 150 200 250 300 350 400 450 500 550 600 650 700];
    %first column for mean, second for std
    time_l = zeros(length(N_l), 2);
    it_cost_l = zeros(length(N_l), 2);
    n_it_l = zeros(length(N_l), 2);
    f_l = zeros(length(N_l), 2);
    n_attemps = 10;
    for i = 1:length(N_l)
        fprintf("Dataset with %d points\n", N_l(i));
        [x, y] = datagen(N_l(i), x_l, x_u, eps);
        time_t_mean = 0;
        time_t_std = 0;
        f_e_mean = 0;
        f_e_std = 0;
        it_cost_mean = 0;
        it_cost_std = 0;
        n_it_mean = 0;
        n_it_std = 0;
        for k = 1:n_attemps
            svr = SVR(x, y, C, e, beta, "standard");
            ops = optimoptions('fmincon');
            ops.Algorithm = 'active-set';
            ops.MaxIterations = round(N_l(i)*10000);
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
            fprintf("Elapsed time: %f\n", time_t_k)
        end
        time_t_std = (time_t_std - time_t_mean^2)^0.5;
        f_e_std = (f_e_std - f_e_mean^2)^0.5;
        it_cost_std = (it_cost_std - it_cost_mean^2)^0.5;
        n_it_std = (n_it_std - n_it_mean^2)^0.5;
        time_l(i, :) = [time_t_mean ; time_t_std];
        it_cost_l(i, :) = [it_cost_mean ; it_cost_std];
        n_it_l(i, :) = [n_it_mean ; n_it_std];
        f_l(i, :) = [f_e_mean ; f_e_std];
    end
end