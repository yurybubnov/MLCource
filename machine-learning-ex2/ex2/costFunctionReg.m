function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


tl = 0;
for i = 2:size(theta)
  tl = tl+theta(i)^2;
endfor

tl = tl* lambda /(2*m);

for i = 1:m
  %J = J+(theta'*X(i, :)' - y(i))^2
  J = J + ( -1 * y(i) * log(sigmoid(theta'*X(i, :)')) - (1-y(i))* log(1-sigmoid(theta'*X(i, :)')));
endfor

J = J/m + tl;

for j = 1:size(theta)
  for i = 1:m
    grad(j) =  grad(j)+(sigmoid(theta'*X(i, :)') - y(i))*X(i, j);
  endfor
  grad(j) =grad(j)/m ;
  if j>1 
     grad(j) = grad(j)+ lambda* theta(j)/m;
  endif
endfor



% =============================================================

end
