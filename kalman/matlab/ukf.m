function [Xk, P_k] = ukf( X_k_1, P_k_1, z_k, kx, ky, g, Ts, Qt, Rt, n_X, n_Z) % ekf ÂË²¨º¯Êý
n = n_X;

% sqrt_P_k_1 = sqrt(P_k_1);
sqrt_P_k_1 = chol(P_k_1);

alfa = 200;
k = 0;
beta = 2;

lambda = alfa^2 * (n + k) - n;
n_plus_lambda = lambda + n;
sqrt_n_lambda = sqrt(n_plus_lambda);
%% 2
sigma_k_1 = sample_sigma(n, X_k_1, sqrt_n_lambda, sqrt_P_k_1);

omega_m = zeros(1, 2*n+1);
omega_c = zeros(1, 2*n+1);
omega_m(:) = 1/(2*n_plus_lambda);
omega_c(:) = 1/(2*n_plus_lambda);
omega_m(1) = lambda/n_plus_lambda;
omega_c(1) = lambda/n_plus_lambda + (1 - alfa^2 + beta);
%% 3
sigma_k_pred_star = zeros(n, 2*n+1);
for i = 1:2*n+1
    sigma_k_pred_star(:, i) = fff(sigma_k_1(:, i), kx, ky, g, Ts);
end
%% 4
X_k_pred = sum(omega_m.*sigma_k_pred_star, 2);
%% 5
tmp = sigma_k_pred_star-X_k_pred;
%P_k_pred = (omega_c.*tmp) * tmp' + Rt;
P_k_pred = weight_vec_wise(omega_c, tmp, tmp) + Rt;
% sqrt_P_k_pred = sqrt(P_k_pred);
%% 6
sqrt_P_k_pred = chol(P_k_pred);
sigma_k_pred = sample_sigma(n, X_k_pred, sqrt_n_lambda, sqrt_P_k_pred);
%% 7
Z_k_pred = zeros(n_Z, 2*n+1);
for i = 1:2*n+1
    Z_k_pred(:, i) = hhh(sigma_k_pred(:, i), Ts);
end
%% 8
z_k_pred = sum(omega_m.*Z_k_pred, 2);
%% 9
tmp = Z_k_pred - z_k_pred;
% S_k = (omega_c.*tmp)*tmp' + Qt;
S_k = weight_vec_wise(omega_c, tmp, tmp) + Qt;
%% 10
% P_k_xz_pred = (omega_c.*(sigma_k_pred - X_k_pred))*tmp';
P_k_xz_pred = weight_vec_wise(omega_c, sigma_k_pred - X_k_pred, tmp);
%% 11
K_k = P_k_xz_pred * inv(S_k);
%% 12
Xk = X_k_pred + K_k*(z_k - z_k_pred);
%% 13
P_k = P_k_pred - K_k * S_k * K_k';
end

function [sigma_k_1] = sample_sigma(n, mu, sqrt_n_lambda, sqrt_var)
    sigma_k_1 = zeros(n, 2*n+1);
    for i = 0:2*n
        if i <= n && i > 0
            sigma_k_1(:, i+1) = mu - sqrt_n_lambda*sqrt_var(:, i);
        elseif i > n
            sigma_k_1(:, i+1) = mu + sqrt_n_lambda*sqrt_var(:, i-n);
        elseif i == 0
            sigma_k_1(:, i+1) = mu;
        end
    end
end

function [out] = weight_vec_wise(weight, A, B)
out = zeros(size(A, 1), size(B, 1));
for i = 1:size(weight,2)
   out = out + weight(i)*A(:, i) * B(:, i)';
end
end
