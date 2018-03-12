close all
clear all
clc
addpath('/opt/cvx')
addpath('/opt/glpkmex')
addpath('/opt/gurobi/linux64/matlab')
cvx_setup
%% TP3 PROG LINEAIRE
%Question 1
%% Avec CVX
tic 
cvx_begin
variables x_1 x_2
dual variables lip glu pro die p1 p2;
minimize( 30*x_1 + 20*x_2 )
subject to
lip : 8 *x_1 + 12* x_2 >= 24;
glu : 12 *x_1 + 12 *x_2 >= 36;
pro : 2 *x_1 + x_2 >= 4;
die : x_1 + x_2 <= 5;
p1 : x_1 >=0;
p2 : x_2 >=0;
cvx_end
temps_cvx=toc;

%% Sous forme matriciel 
f = [30 ; 20];
b = [24;36;4;-5];
A = [8 12 ; 12 12 ; 2 1 ; -1 -1];

%% Avec CVX sous forme standard
tic
cvx_begin
variable x(2)
dual variables a p;
minimize( f'*x )
subject to
    a : A*x >= b
    p : x>=0;
cvx_end  
temps_cvxMatriciel=toc

%% Avec GLPK sous forme standard
ctype ='UUUU'; % contraintes dé'égalites
vartype ='CC'; % variables continues
s = 1;          % 1 minimisation (-1 -> maximisation)
param.msglev = 1;
param.itlim = 100;
tic
[xmin_g,fmin,status,extra]=glpk(f,-A,-b,0*f, [], ctype, vartype, s, param);
temps_GLPK=toc
%% Avec linprog sous forme standard
tic
[xl c e o p] = linprog(f,-A,-b,[],[],0*f);
temps_linprog=toc;

%% Comparaison des differents solveurs
[[x_1;x_2] x xl xmin_g]

% Les solveurs trouvent les memes solutions. 
% Par contre, GLPK est le solveur le plus rapide avec un temps d'execution
% de 0.01s, par rapport à linprog (0.18s) et cvx avec forme standard(0.28s)


%% Question 2 
%% lineaire avec CVX
close all
clear all
clc

%%
a = [400; 1500; 900];
b = [700; 600; 1000; 500];
C = [20 40 70 50
100 60 90 80
10 110 30 200];
[n,p] =size(C);
tic
cvx_begin
    variable X(n,p)
    dual variables D A pos;
    minimize(sum(sum(C.*X)))
    subject to
        D : sum(X) == b';
        A : sum(X') == a';
        pos : X >= 0;
cvx_end
temps2_cvx=toc


%% Lineaire et standard avec CVX

% ici on choisis de vectoriser x mais on peux faire autrement

e = ones(1,n);
z =zeros(1,n);
I =eye(n);
M = [e z z z; z e z z; z z e z; z z z e ;repmat(I,1,p)];
Cl =reshape(C,1,n*p);
bb = [b ; a];

tic
cvx_begin
variable x(n*p)
dual variables D pos;
minimize( Cl*x )
subject to
D : M*x == bb;
pos : x >= 0;
cvx_end
temps2_cvxFS=toc


%% Avec linprog
tic
[xl c e o p] = linprog(Cl',[],[],M,bb,0*Cl);
temps2_linprog=toc

%% Avec GUROBI
clear model;
model.obj = Cl;
model.A =sparse(M);
model.sense = ['=';'=';'=';'=';'=';'=';'='];
model.rhs = bb;
model.lb =zeros(size(x));
clear params;
params.Presolve = 2;
params.TimeLimit = 100;
tic
result = gurobi(model, params);
temps2_gurobi=toc
disp(result.objval);

%% Avec Cplex
tic
[xcl c e o p] = cplexlp(Cl',[],[],M,bb,0*Cl);
temps2_cplex=toc


%% Avec GLPK 

ctype ='SSSSSSS';        % contraintes d'éégalites
vartype ='CCCCCCCCCCCC'; % variables continues
s = 1;                    % 1 minimisation (-1 -> maximisation)
param.msglev = 1;
param.itlim = 100;
tic
[xmin_g,fmin,status,extra]=glpk(Cl',M,bb,0*Cl, [], ctype, vartype, s, param);
temps2_glpk=toc

%% Comparaison des solveurs

[xl result.x xcl xmin_g]

% Pour ce probleme, c'est glpk qui fourni la solution en un temps minimal
% de 0.01s, cplex et gurobi sont equivalent avec 0.05s et 0.06s .. linprog
% est loin derriere avec 0.16s et cvx avec la forme standard 0.24s








