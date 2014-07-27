% Copyright by Siddharth Gopal, Carnegie Mellon University, gcdart@gmail.com
% 
% Implements the TCS-GM-* methods given in 
%  Siddharth Gopal and Yiming Yang, Transformation-based Probabilistic Clustering with Supervision,
%  Uncertainty in Artificial Intelligence, 2014
%
%  Inputs : 1. X (PxN matrix) : training examples 
%           2. Y (KxN matrix) : 0-1 matrix of Cluster assignment training examples
%                               In each column, there is exactly one 1, rest are 0s.
%           3. options
%           P = #dims, N = #examples, K = #classes
%  Output : L : linear transformation matrix which can be applied as Lx, where x in Px1.
%  options: 
%           lambda  = regularization parameter, typically test range of values from 1e-5 to 500
%           verbose = 0 (default) or 1 [prints messages after each line]
%           eps     = 1e-3 (default)   [slightly stricter than difference in objective]
%           l0      = _xP matrix ,     [variable L'L initialized as l0'l0]
%           regularization = {1,2,3,4} , A = L'L
%                    where    1 : ||A-I||^2 (works well for most problems)
%                             2 : ||A||^2   (works well for faces, time-series)
%                             3 : ||A||_*   nuclear norm
%                             4 : log(det(A))
%           citer   = maximum number of iterations (default 5000)
%
% RECOMMENDED LAMBDAS (Try these values first and fine-tune on a validation set)
%           lambda = [1024,512,128,64,32,16,8,4,2,1,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,
%                     .001953125,.0009765625,.00048828125, .000244140625,6.103515625e-05,3.0517578125e-05];
% NOTE: If P > 200, consider doing SVD ! 
%        [U,S,V] = svds(X,50); 
%        projected_X = U'*X;
%        L = TCS(projected_X,Y,options);
%        L = L*U';

function [L] = TCS( X , Y , options )
    [P,N] = size(X);

    if isfield(options,'verbose') == 0
        options.verbose = 0;
    end

    if isfield(options,'eps') == 0
        options.eps = 1e-3;
    end

    if isfield(options,'citer') == 0
        options.citer = 5000;
    end

    if isfield(options,'regularization') == 0
        options.regularization = 1;
    else
        if ~(options.regularization == 1 | options.regularization == 2 | options.regularization == 3 | options.regularization == 4);
            fprintf(1,'Unknown regularization option\n');
            return;
        end
    end

    % Incase input is a vector, each element is from 1..K.
    if size(Y,1) == 1 || size(Y,2) == 1
        Y = reshape(Y,1,length(Y));
        YY = sparse(zeros(max(Y),N));
        YY( sub2ind(size(YY),Y,1:N) ) = 1;
        Y = YY;
    end
    Y = sparse(Y);

    % Remove empty rows from Y
    rowsums = sum(Y,2);
    sel = find(rowsums>0);
    Y = Y(sel,:);
    [K,N] = size(Y);

    % Get indices of positive labels
    [dummy,labels] = max(Y);
    label_ind = sub2ind(size(Y),labels,1:N);
    lambda = options.lambda;

    % Initialize
    A = full(eye(P));
    if isfield(options,'l0') == 1
        A = full(options.l0'*options.l0);
    end

    Pr = zeros(K,N);
    M = bsxfun(@rdivide,Y*X',sum(Y,2));
    step = .001;

    % Function-values
    fvalues = zeros(options.citer,1);
    threshold = 20;
    rho = 1.5;
    step_start = 0;

    for it = 1:options.citer
        [F,G] = fg(A);

        for ct = step_start:50;
            step = (1/rho)^ct;
            nA = A + step * G;
            nA = proj(nA);
            nF = f(nA);
            % Do atmost 50 steps for line-search by decreasing step length by 2 each time.
            if nF < F && ~(ct==49 && it <= 10)
                continue;
            else
                step_start = max(ct-2,0);
                A = nA;
                break;
            end
        end

        % No valid step-length
        if ct == 50;
            break;
        end

        nm = norm(G);

        if options.verbose == 1
            fprintf(1,'[%d] Function : %f , Gradient : %f\n',it, F, nm );
        end

        fvalues(it) = F;
        % Break if no improvement in last threshold iterations
        if it > threshold
            denom = min( abs(fvalues(it-threshold:it)) );
            denom = min( denom , 1 );  
            % denom can be left to 1, but we need stricter criteria if objective is small.
            num = abs(fvalues(it-threshold) - fvalues(it));
            if num/denom < options.eps
                break;
            end
        end
    end

    fprintf(1,'[%d] Function : %f , Gradient : %f\n',it, F, nm );

    [A,L] = proj(A);
    [d,pred] = max(Pr);
    [d,tr] = max(Y);

    function [A,L] = proj( A )
        [L,dg] = eig(A);
        dg = real(diag(dg));
        L = real(L);
        dg( find(dg<1e-9) ) = 0;       
        if options.regularization == 3 % nuclear norm
            dg( find(dg<lambda+1e-9) ) = 0;
        end
        L = ( L * diag(sqrt(dg)) )';
        A = L'*L;
    end


    function [F,G] = fg( A )
        a = sum( (A*X).* X , 1 );
        b = sum( (M*A').* M , 2 );
        Pr = -.5*bsxfun(@plus , bsxfun(@plus , -2*M*A*X , a ) , b );
        Pr = exp( bsxfun(@minus,Pr,max(Pr)) );
        Pr = bsxfun(@rdivide,Pr,sum(Pr));

        EPS = 1e-12;
        Pr( Pr < EPS ) = EPS;
        Pr( Pr > 1-EPS ) = 1-EPS;

        G = zeros(P,P);
        for k = 1:K
            temp = Y(k,:) - Pr(k,:);
            X = bsxfun(@minus,X,M(k,:)');
            add = bsxfun(@times,X,temp) * X';
            X = bsxfun(@plus,X,M(k,:)');
            G = G + add;
        end
        G = G ./ N;
        F = sum( log( Pr(label_ind) ) ) / N;
 
        if options.regularization == 2 % plain L2 regularization
            G = -.5*G - lambda*(A);
            F = F - lambda/2*sum(sum( A.^2 ));
        else
            if options.regularization == 4 % log-det 
                G = -.5*G - lambda*( eye(P) - inv(A) );
                F = F - lambda*( trace(A) - log(det(A)) );
            else 
                if options.regularization == 3 % nuclear norm
                    G = -5*G;
                    F = F - lambda*trace(A);
                else
                    G = -.5*G - lambda*(A-eye(P));
                    F = F - lambda/2*sum(sum( (A-eye(P)).^2 ));
                end
            end
        end
    end

    function [F] = f( A )
        a = sum( (A*X).* X , 1 );
        b = sum( (M*A').* M , 2 );
        Pr = -.5*bsxfun(@plus , bsxfun(@plus , -2*M*A*X , a ) , b );

        %for k = 1:K
        %    X = bsxfun(@minus,X,M(k,:)');
        %    Pr(k,:) = -.5*sum(bsxfun(@times,A*X,X));
        %    X = bsxfun(@plus,X,M(k,:)');
        %end
        Pr = exp( bsxfun(@minus,Pr,max(Pr)) );
        Pr = bsxfun(@rdivide,Pr,sum(Pr));

        EPS = 1e-12;
        Pr( Pr < EPS ) = EPS;
        Pr( Pr > 1-EPS ) = 1-EPS;

        F = sum( log( Pr(label_ind) ) ) / N;

        if options.regularization == 2 
            F = F - lambda/2*sum(sum( A.^2 ));
        else
            if options.regularization == 4
                F = F - lambda*( trace(A) -log(det(A)) );
            else
                if options.regularization == 3
                    F = F - lambda*trace(A);
                else
                    F = F - lambda/2*sum(sum( (A-eye(P)).^2 ));
                end
            end
        end
    end


end


