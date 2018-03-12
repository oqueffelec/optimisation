close all
clear all
clc
%% 1 Programmation lineaire et regression MAD
% Generation des données suivant la loi uniforme
randn('seed',1);
rand('seed',1);
n = 100;
x =sort(rand(n,1));
nt = 1000;
xt =linspace(0,1,nt)';
y =cos(pi*x);
yt =cos(pi*xt);
sig = 0.25;
ya = y+sig*randn(size(y));

%Ajout de bruit
ya(1) = -1;
ya(n) = 1;

%Plot donnees
figure(1);
tt =plot(x,y);
hold on;
oo =plot(x,ya,'x');

%% 2 Resolution du pb
% Reformulation matricielle du pb 
X = [ones(size(x)) x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9 ];
Xt = [ones(size(xt)) xt xt.^2 xt.^3 xt.^4 xt.^5 xt.^6 xt.^7 xt.^8 xt.^9 ];
[n,p] =size(X);

%Resolution cvx
cvx_begin
    variables beta_cvx(p)
    minimize(sum(abs(X*beta_cvx - ya)) )
cvx_end

% En virant la valeure abs
e = ones(n,1);
cvx_begin
variables beta_cvx2(p) ep(n) em(n)
dual variable d
minimize( e'*ep + e'*em)
subject to
d : X*beta_cvx2 - ya == ep-em;
ep >= 0;
em >= 0;
cvx_end

%Reformulation LP
f = [zeros(2*p,1) ; ones(2*n,1)];
I = eye(n);
A = [X -X -I I];
b = ya;
lb = zeros(2*p+2*n,1);
ub = [];

% Resolution avec cvx again
cvx_begin
    cvx_quiet(true)
    cvx_precision best
    variables xl(2*p+2*n)
    dual variables d mu
    minimize( f'*xl )
    subject to
    d : A*xl == b;
    mu: xl >= lb;
cvx_end

beta_cvx3 = xl(1:p)-xl(p+1 : 2*p);

%verif des kkt 
% Stattionarite
stationnarite=f-A'*d-mu; % ATTENTION !!!! f+ou- A'... en fonction du solveur utilise.. Bien verif que la condition egal a 0
primal=A*xl-b % primalite est bien verif egal a 0
% mu>0 doit etre pos
complementarite=mu.*xl % nul

% DUAL resolution
cvx_begin
cvx_quiet(true)
cvx_precision best
variables yl(n)
dual variable dd
maximize( b'*yl )
subject to
dd : A'*yl <= f;
cvx_end


x_lin = linprog(f,[],[],A,b,0*f);
beta_lin = x_lin(1:p)-x_lin(p+1 : 2*p);

addpath('/nfs/opt/CPLEX/cplex/matlab');
x_cpl = cplexlp(f,[],[],A,b,0*f);

a= beta_cvx3

[beta_cvx beta_cvx2 xl(1:p) - xl(p+1:2*p) x_lin(1:p) - x_lin(p+1:2*p) x_cpl(1:p) - x_cpl(p+1:2*p) dd(1:p) - dd(p+1:2*p)]
ee = plot(xt,Xt*a,'r');
set(ee,'LineWidth',3);
ee2 = plot(xt,Xt*(X\ya),'g');
legend([tt, oo, ee , ee2],'true function','observed data','estimated function','estimation des MC');

%le premier programme CVX est plus lent car il doit réaliser les
%transformations en programme linéaire. Le dernier est plus rapide car bien
%optimisé. Systeme de reecriture prends bcp de temps avant d'appeler
%linprog.

%% 3

sig = .5;
yb = y+sig*abs(randn(size(y)));

cvx_begin
cvx_quiet(true)
cvx_precision best
variables beta_cvx(p)
dual variable d
minimize( sum(abs(X*beta_cvx - yb)) )
subject to
d : X*beta_cvx <= yb;
cvx_end

figure(2)
set(gcf,'Color',[1,1,1])
oo = plot(x,yb,'ob');hold on
tt = plot(x,y,':c');
ee = plot(xt,Xt*beta_cvx,'r');
legend([tt, oo, ee],'true function','observed data','estimated function')
