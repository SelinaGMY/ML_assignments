load co2.txt -ascii

X = [co2(:,3)';ones(1,length(co2(:,3)))];
Y = co2(:,4)';

% a
s_sigma = 1;
w = [100,0; 0,10000];
w_cov = (X * X'/s_sigma + w^-1)^-1;
w_mean = w_cov * X * Y'/ s_sigma;

% b
res = Y - w_mean' * X;
figure; plot(X(1,:),res); xlabel("time"); ylabel("residual");

% d
% parameters
theta = 3;
tau = 1;
sigma = 2;
phi = 0.5;
zeta = 0.05;
eta = 4;
% kernel
k = @(s,t) theta^2 * (exp(-2*sin(pi*(s-t)/tau)^2/sigma^2) + phi^2*exp(-(s-t)^2/(2*eta^2))) + zeta^2 * (s==t);
% draw from GP
[y_GP,cov_GP] = GP(X(1,:),k);
figure; plot(X(1,:),y_GP); 
title(['\theta=',num2str(theta),' \tau=',num2str(tau),' \sigma=',num2str(sigma),' \phi=',num2str(phi),' \zeta=',num2str(zeta),' \eta=',num2str(eta)])

% % f
% predict
X = co2(:,3)';
Xnew = [X(end):1/12:2035];
Xnew = Xnew(2:(end-1));
k_XXnew = covkernel(X,Xnew,k);
k_XnewXnew = covkernel(Xnew,Xnew,k);
k_XX = covkernel(X,X,k);
mean_new = res * k_XX^(-1) * k_XXnew + w_mean' * [Xnew;ones(1,length(Xnew))];
cov_new = k_XnewXnew - k_XXnew' * k_XX^(-1) * k_XXnew;
std_new = sqrt(diag(cov_new))';
% plot
figure;
hold on
f = [mean_new + std_new; flipud(mean_new - std_new)];
fill = patch([Xnew;flipud(Xnew)],f,[7 7 7]/8); 
data = plot(X,Y,'.');
data_pred = plot(Xnew,mean_new,'r');
lgd = legend([data,data_pred,fill],'data','pred mean','error bar','Location','best');
lgd.FontSize = 20;
title(['\theta=',num2str(theta),' \tau=',num2str(tau),' \sigma=',num2str(sigma),' \phi=',num2str(phi),' \zeta=',num2str(zeta),' \eta=',num2str(eta)],'FontSize',20);
xlabel("Year",'FontSize',20);ylabel("co2",'FontSize',20);

