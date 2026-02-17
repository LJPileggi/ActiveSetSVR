function [x, y] = datagen(N, x_l, x_u, eps)
    rng(1);
    a1 = 5;
    a2 = 3;
    k1 = 0.5;
    k2 = 1.5;
    x = (x_l - x_u)*(1:N)/N + x_u;
    y = a1*sin(k1*x) + a2*sin(k2*x) + rand(1, N)*eps;
end