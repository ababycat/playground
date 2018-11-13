x = (1:1:100)';
len = size(x, 1);
y = x + randn(len, 1);
noise = zeros(len, 1);
noise(1:5:end) = 10;
% noise(98:end) = 20;
y = y + noise;

X = [ones(len, 1), x];

method = 0;
if method == 0
    theta = Least_Squares(X, y);
elseif method == 1
    theta = Weighted_Least_Squares(X, y);
elseif method == 2
    theta = Robust_Least_Squares(X, y);
end

y_p = linear(x, theta);

plot(x, y, '.'), hold on
plot(x, y_p);

function y = linear(x, theta)
len = size(x, 1);
X = [ones(len, 1), x];
y = X*theta;
end

function theta = Least_Squares(X, y)
theta = Normal_Equation(X, y);
end

function theta = Weighted_Least_Squares(X, y)
W = eye(size(y, 1));

theta = Weight_Normal_Equation(X, y, W);
end

function theta = Robust_Least_Squares(X, y)
W = eye(size(y, 1));
theta = Weight_Normal_Equation(X, y, W);

y_p = X*theta;
r = y_p - y;

end

function theta = Normal_Equation(X, y)
theta = pinv(X'*X)*X'*y;
end

function theta = Weight_Normal_Equation(X, y, W)
theta = pinv(X'*W*X)*X'*W*y;
end

