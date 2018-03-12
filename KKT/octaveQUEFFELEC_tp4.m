close all
clear all
clc
%% TP4 KKT dualite 
%Question 1
%resolution avec cvx
f= [30 ; 20];
b = [24;36;4;-5];
A = [8 12 ; 12 12 ; 2 1 ; -1 -1];
cvx_begin
variable x(2)
dual variables a p;
minimize( f'*x )
subject to
a : A*x >= b
p : x >= 0;
cvx_end

%% avec le langrangien 

% a et p sont des multiplicateurs de lagrange = dual variables 
f - A'*a - p
sign(A*x-b)
sign(x)
sign(a)
sign(p)
a.*(A*x-b)
p.*x

ind =find(abs(a)>0.00001)
A(ind,:)\b(ind)

% le pb dual 

cvx_begin
variable y(4)
dual variables d e;
maximize( b'*y )
subject to
d : A'*y <= f;
e : y >=0 ;
cvx_end

% pb primal avec linprog
[xl c e o du] = linprog(f,-A,-b,[],[],0*f);

%verif de l'optimabilité 
a = du.ineqlin;
p = du.lower;
f - A'*a - p
sign(A*xl-b)
sign(xl)
sign(a)
sign(p)
a.*(A*xl-b)
p.*x

%% 2 pb quadratiques 
% resolution avec cvx
cvx_begin
variables x1 x2
dual variables d e1 e2;
minimize( .75*x1^2 + .75*x2^2 +.25*(x1+x2)^2 - 6* x1 - 8 *x2 )
subject to
d : x1 + x2 == 5;
e1 : x1 >=0 ;
e2 : x2 >=0 ;
cvx_end

% resolution sous forme matricielle 
Q = 2*[1 0.25 ; 0.25 1];
f = [-6 ; -8];
A = [-1 -1];
b = -5;
tic
cvx_begin
variables x(2)
dual variables d e;
minimize( .5*x'*Q*x + f'*x )
subject to
d : -A*x == -b;
e : x >=0 ;
cvx_end

% verification de la convexité du pb 
eig(Q) % pour que Q soit defini positive et que le pb soit donc convexe, on d'assure que ses vp soient positives.

% lagrangien et verif des kkt 

Q*x + f + A'*d - e
A*x-b
sign(e)
e.*x

% solution optimale 

[Q A'; A 0]\[-f;b]

% on modifie le pb 
Q = 2*[1 0.25 0 ; 0.25 1 .25 ; 0 .25 1];
f = [-6 ; -8 ; -2];
A = [-1 -1 -1];
b = -5;
cvx_begin
variables x(3)
dual variables d e;
minimize( .5*x'*Q*x + f'*x )
subject to
d : A*x == b;
e : x >=0 ;
cvx_end

% verif de l'optimi des kkt
Q*x + f - A'*d - e
-A*x+b
sign(e)
e.*x



