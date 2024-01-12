% generate data
rng(100)
num = 200;
t = unifrnd(0,2*pi,[1 num]);
n1 = normrnd(0,0.0001,[1,num]);
rng(200)
n2 = normrnd(0,0.0001,[1,num]);
x = sin(t)+n1;
y = cos(t)+n2;

figure
scatter(x,y)

% generate gaussian kernel matrices K & L
sigma = 2;
K = ones(num,num);
for i=1:num
    for j=(i+1):num
        K(i,j) = kernel(x(i),x(j),sigma);
        K(j,i) = K(i,j);
    end
end
L = ones(num,num);
for i=1:num
    for j=(i+1):num
        L(i,j) = kernel(y(i),y(j),sigma);
        L(j,i) = L(i,j);
    end
end

% incomplete cholesky to K
j = 0;
R = zeros(num,num);
d = diag(K);
[a,I(j+1)] = max(d);
eta = 0.000001;
while a > eta
    j = j + 1;
    nu(j) = sqrt(a);
    for i = 1:num
        R(j,i) = (K(I(j),i) - R(:,i)'*R(:,I(j)))/nu(j);
        d(i) = d(i) - R(j,i)^2;
    end
    [a,I(j+1)] = max(d);
end
T = j;
R = R(1:T,:);

% incomplete cholesky to L
j = 0;
Q = zeros(num,num);
d = diag(L);
[b,I(j+1)] = max(d);
eta = 0.000001;
while b > eta
    j = j + 1;
    nu(j) = sqrt(b);
    for i = 1:num
        Q(j,i) = (L(I(j),i) - Q(:,i)'*Q(:,I(j)))/nu(j);
        d(i) = d(i) - Q(j,i)^2;
    end
    [b,I(j+1)] = max(d);
end
T = j;
Q = Q(1:T,:);

% solve COCO, alpha, beta
K_tilde = R'*R;
L_tilde = Q'*Q;
A = zeros(num*2,num*2);
A(1:num,(num+1):end) = 1/num * K_tilde * L_tilde;
A((num+1):end,1:num) = 1/num * L_tilde * K_tilde;
B = blkdiag(K_tilde,L_tilde);
C = B\A;
[V,D] = eig(C); 
[~,col] = find(D==max(max(D)));
eigvec = reshape(V(:,col),[1,num*2]);
alpha = eigvec(1:num);
beta = eigvec((num+1):end);
disp("coco value is")
disp(max(max(D)))

% generate test data 
rng(345)
t = unifrnd(0,2*pi,[1 num]);
n1 = normrnd(0,0.0001,[1,num]);
rng(456)
n2 = normrnd(0,0.0001,[1,num]);
x_test = sin(t)+n1;
y_test = cos(t)+n2;

% apply witness function to test data
f = zeros(1,num);
g = zeros(1,num);
for n=1:num
    fn=0;
    gn=0;
    for i=1:num
        totalf=0;
        totalg=0;
        for j=1:num
            totalf = totalf + kernel(x(j),x_test(n),sigma);
            totalg = totalg + kernel(y(j),y_test(n),sigma);
        end
        meanf = totalf/num;
        meang = totalg/num;
        fn = fn + alpha(i)*(kernel(x(i),x_test(n),sigma)-meanf);
        gn = gn + beta(i)*(kernel(y(i),y_test(n),sigma)-meang);
    end
    f(n) = fn;
    g(n) = gn;
end

% correlation
corr = corrcoef(f,g);

% plot witness functions
figure
scatter(x_test,f); xlabel("x");ylabel("f(x)")
figure
scatter(y_test,g); xlabel("y");ylabel("g(y)")
figure
scatter(f,g); xlabel("f(x)");ylabel("g(y)");title(["correlation: ",corr(2)])

% kernel function
function k = kernel(x1,x2,sigma)
    k = exp(-norm(x1-x2)^2/(2*sigma^2));
end

