function H = JacobianH(X) % 量测雅可比函数
    x = X(1); y = X(3);
    if y > 0
        y = max(y, 1e-3);
    else
        y = min(y, -1e-3);
    end
    H = zeros(2,4);
    r = sqrt(x^2+y^2);
    H(1,1) = x/r; H(1,3) = y/r;
    xy2 = 1+(x/y)^2;
    H(2,1) = 1/xy2*1/y; H(2,3) = 1/xy2*x*(-1/y^2);