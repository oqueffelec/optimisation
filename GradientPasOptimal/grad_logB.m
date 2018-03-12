function g = grad_logB(a,X,y,c)
r = X*a-y;
df = 2*c*r./(c^2-r.^2);
g = X'*df;
end