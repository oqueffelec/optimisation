function cout = CoutLogB(alpha,x,y,c)
% p=length(alpha);
% X=[];
% for i=1:p
%     X=[X x.^(i-1)];
% end
r=x*alpha-y;
cout=sum(-c*log(1-r.^2/c^2));
end