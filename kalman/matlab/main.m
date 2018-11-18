    kx = .01; ky = .02; 	% 阻尼系数
    kx = .0; ky = .0; 	% 阻尼系数
    g = 9.8; 		% 重力
    t = 10; 		% 仿真时间
    Ts = 0.1; 		% 采样周期
    len = fix(t/Ts);    % 仿真步数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %（真实轨迹模拟）
%     dax = 1.5; day = 1.5;  % 系统噪声
    dax = 1.; day = 1.;  % 系统噪声
    X = zeros(len,4); X(1,:) = [0, 50, 500, 0]; % 状态模拟的初值
    for k=2:len
        x = X(k-1,1); vx = X(k-1,2); y = X(k-1,3); vy = X(k-1,4); 
        x = x + vx*Ts;
        vx = vx + (-kx*vx^2+dax*randn(1,1))*Ts;
        y = y + vy*Ts;
        vy = vy + (ky*vy^2-g+day*randn(1))*Ts;
        X(k,:) = [x, vx, y, vy];
    end
    figure(1), hold off, plot(X(:,1),X(:,3),'-b'), grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 构造量测量
    mrad = 0.001;
    dr = 10; dafa = 10*mrad; % 量测噪声
    for k=1:len
        r = sqrt(X(k,1)^2+X(k,3)^2) + dr*randn(1,1);
        a = atan(X(k,1)/X(k,3)) + dafa*randn(1,1);
        Z(k,:) = [r, a];
    end
    figure(1), hold on, plot(Z(:,1).*sin(Z(:,2)), Z(:,1).*cos(Z(:,2)),'*')

    % ekf 滤波
    Qk = diag([0; dax; 0; day])^0.2;
    Rk = diag([dr; dafa])^1;
    Xk = zeros(4,1);
    Pk = 100*eye(4);
    X_est = zeros(size(X,1), size(X, 2));
    X_est(1, :) = [0, 0, 0.1, 0.];
   
    for k=2:len
        Ft = JacobianF(X_est(k-1,:), kx, ky, g);
        Hk = JacobianH(X_est(k-1,:));
        fX = fff(X_est(k-1,:), kx, ky, g, Ts);
        hfX = hhh(fX, Ts);
        [Xk, Pk, Kk] = ekf(eye(4)+Ft*Ts, Qk, fX, Pk, Hk, Rk, Z(k,:)'-hfX);
        X_est(k,:) = Xk';
    end
    figure(1), plot(X_est(:,1),X_est(:,3), '+r')
    xlabel('X'); ylabel('Y'); title('ekf simulation');
%     legend('real', 'measurement', 'ekf estimated');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     X_est = zeros(size(X,1), size(X, 2));
%     X_est(1, :) = [0, 50, 500, 0];
%     P_k = diag([100, 100, 100, 100]);
%     R_t = diag([0, 0.1, 0, 0.1]);
%     Q_t = diag([10, 0.01]);
%     n_X = 4;
%     n_Z = 2;
    % ukf 滤波
    error1 = X_est - X;
   
    X_est = zeros(size(X,1), size(X, 2));
    X_est(1, :) = [0, -10, 500, 0];
    P_k = diag([100, 100, 100, 100]);
    R_t = diag([0, 0.1, 0, 0.1]);
    Q_t = diag([0.01, 0.001]);
    n_X = 4;
    n_Z = 2;
    
    for k=2:len
        X_k_1 = X_est(k-1, :)';
        P_k_1 = P_k;
        z_k = Z(k, :)';
        [Xk, P_k] = ukf(X_k_1, P_k_1, z_k, kx, ky, g, Ts, Q_t, R_t, n_X, n_Z);
%         [Xk, P_k] = ukf( X_k_1, P_k_1, z_k);
        X_est(k,:) = Xk';
    end
    
    error2 = X_est - X;
    
    figure(1), plot(X_est(:,1),X_est(:,3), 'oy')
    xlabel('X'); ylabel('Y'); title('ekf, ukf simulation');
    legend('real', 'measurement', 'ekf estimated', 'ukf estimated');

    figure(2), 
    subplot(121),
    plot(error1(:, 1)), hold on
    plot(error2(:, 1))
    legend('ekf x error', 'ukf x error')
    xlabel('time')
    ylabel('error')
    title('error')
    subplot(122),
    plot(error1(:, 3)), hold on
    plot(error2(:, 3))
    legend('ekf y error', 'ukf y error')
    title('error')
    xlabel('time')
    ylabel('error')
    
    