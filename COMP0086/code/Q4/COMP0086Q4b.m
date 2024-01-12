X = readmatrix("ssm_spins.txt")';

% initialisation
d = 5; k = 4;
A = 0.99 * [cos(2*pi/180) -sin(2*pi/180) 0 0; 
            sin(2*pi/180) cos(2*pi/180) 0 0;
            0 0 cos(2*pi/90) -sin(2*pi/90);
            0 0 sin(2*pi/90) cos(2*pi/90)];
C = [1 0 1 0;
     0 1 0 1;
     1 0 0 1;
     0 0 1 1;
     0.5 0.5 0.5 0.5];
Q = eye(size(A)) - A*A';
R = eye(d,d);
Y_init = zeros(k,1);
Q_init = eye(k,k);

% run EM algorithm
iter = 500;
[Aem,Cem,Qem,Rem,Loglik] = LGSSM_EM(X,A,C,Q,R,Q_init,Y_init,iter);

% plot 
plot(0:iter,Loglik)
hold on
plot([0,iter],[Loglik(1),Loglik(1)],":")
xlabel('EM iteration')
ylabel('loglikelihood')
hold off

% random initialisation
rep = 10;
Arand = zeros(k,k,rep);
Crand = zeros(d,k,rep);
Qrand = zeros(k,k,rep);
Rrand = zeros(d,d,rep);
Loglikrand = zeros(iter+1,rep);

for i = 1:rep
    while(1)
    A = randn(k,k);
    Q = iwishrnd(eye(k),k);
    C = randn(d,k);
    R = iwishrnd(eye(d),d);
    Y_init = randn(k,1);
    Q_init = iwishrnd(eye(k),k);

    if all(eig(C*Q*C'+R)>0) && all(eig(A)>0)
        break
    end    
    end
    [Aran,Cran,Qran,Rran,Loglikran] = LGSSM_EM(X,A,C,Q,R,Q_init,Y_init,iter);
    Arand(:,:,rep) = Aran;
    Crand(:,:,rep) = Cran;
    Qrand(:,:,rep) = Qran;
    Rrand(:,:,rep) = Rran;
    Loglikrand(:,rep) = Loglikran;

    plot(0:iter,Loglikran)
    xlabel('EM iteration')
    ylabel('loglikelihood')
    hold on
end
plot([0,iter],[Loglik(1),Loglik(1)],":")