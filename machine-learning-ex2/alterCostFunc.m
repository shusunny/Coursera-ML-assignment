% Alternative costFunc

h = sigmoid(X*theta); %m*1
part1 = y.*(log(h)); %m*1
part2 = (1-y).*(log(1-h)); %m*1

J = sum(-part1 - part2) / m; %1*1

diff = h - y; %m*1
temp = X' * diff; % (n+1)*m ¡Á m*1 -> (n+1)*1
temp = temp / m; % (n+1)*1£»

grad = temp;

% costFunc Regularized

[J_ori, grad_ori] = costFunction(theta, X, y);
sz_theta = size(theta, 1);
theta_temp = theta(2:sz_theta);
punish_J = sum(theta_temp.^2)*lambda/2/m;
J = J_ori + punish_J;
 
%--- grad
punish_theta = theta_temp*lambda/m;
punish_theta = [0; punish_theta];
grad = grad_ori + punish_theta;