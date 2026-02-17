function ModelSelection(N, x_l, x_u, eps)
    [x, y] = datagen(N, x_l, x_u, eps);
    idx = randperm(N);
    x_tr = x(:,ismember(1:end, idx(1:round(0.8*N))));
    y_tr = y(ismember(1:end, idx(1:round(0.8*N))));
    x_vl = x(:,ismember(1:end, idx(round(0.8*N)+1:end)));
    y_vl = y(ismember(1:end, idx(round(0.8*N)+1:end)));
    Cs = [0.001 0.01 0.1 1 5 10 50];
    epss = [0.001 0.01 0.1 1 5 10];
    betas = [0.01 0.1 1 5 10];
    f_i = inf;
    best_conf = [0 0 0];
    it = 1;
    for i = 1:length(Cs)
        for j = 1:length(epss)
            for k = 1:length(betas)
                fprintf("Iteration %d\n", it);
                svr = SVR(x_tr, y_tr, Cs(i), epss(j), betas(k), "standard");
                [~, ~, ~, ~] = svr.ActiveSet(round(N*10000), false, false);
                f_p = svr.MSE(x_vl, y_vl);
                if f_p < f_i
                    best_conf = [Cs(i) epss(j) betas(k)];
                    f_i = f_p;
                end
                it = it + 1;
            end
        end
    end
    fprintf("Best config: C %f, eps %f, beta %f with obj func %f", best_conf(1), best_conf(2), best_conf(3), f_i);
end