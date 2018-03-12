function H = Hess_logB(a,X,y,c)
r = (X*a-y);
d2f = 2*c*(c^2+r.^2)./((c^2-r.^2).^2);
D =diag(d2f);
H = X'*D*X;
end
