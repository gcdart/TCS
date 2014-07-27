% Copyright by Siddharth Gopal, Carnegie Mellon University, gcdart@gmail.com
% Implements the simple Gaussian mixture model with a single variance parameter for all
% classes and all dimensions using MAP.
% 
%  Inputs : 1. X (PxN matrix) : training examples 
%           2. K              : number of clusters
%           3. options
%  Outputs: 1. params.Z (KxN matrix) : assignement of points to clusters means.
%           2. params.M (KxP matrix) : cluster means
%           3. params.obj            : The objective of the clustering algorithm
%           4. params.sigma2         : The variance parameter.
%  options: 
%           seed               : random seed
%           kmpp = {0 or 1}    : Initialize with K-means++ ? 
%           verbose = {0 or 1} : Print debug messages ?
%           Zstart (KxN matrix): If specified, Z will be initialized from this.
%           Mstart (KxP matrix): If specified, M will be initialized from this.
%           eps = 1e-3         : Convergence criteria (successive difference in objectives)

function [params] = soft_llyod_kmeans( X , K , options )
     [P,N] = size( X ); 
     Xt = X';
     X2 = sum(X.^2);

     rand('twister',options.seed);

     if isfield(options,'verbose') == 0
         options.verbose = 0;
     end

     if isfield(options,'eps') == 0
         options.eps = 1e-3;
     end

     Z = sparse(K,N);
     Z( sub2ind(size(Z),randsample(K,N,true)',1:N) ) = 1;

     if isfield(options,'kmpp') == 1 && options.kmpp == 1
        Z = kmpp( X , K , options ); 
     end

     if isfield(options,'Zstart') == 1
         Z = options.Zstart;
     end

     M = bsxfun(@rdivide,Z*Xt,sum(Z,2));
     if isfield(options,'Mstart') == 1
         M = options.Mstart;
     end         

     sigma2 = sum(var(X,[],2))./P;

     obj = 0;
     for i = 1:options.maxiter
         options.it = i;
         [Z,nobj] = Estep;
         [M,sigma2] = Mstep();

         if abs(nobj-obj) < options.eps
             break;
         end

         if options.verbose == 1
             fprintf(1,'Iteration %d , Objective = %f\n' , i , nobj );
         end
         obj = nobj;
     end

     params.M = M; params.obj = obj;
     params.Z = Z;
     params.sigma2 = sigma2;

function [Z,F] = Estep;
    logZ = -.5*bsxfun(@plus, bsxfun(@plus, -2*M*X, sum(M.^2,2)) , ...
                      X2 )./sigma2 - .5*P*log(sigma2);
    [D,dummy] = max(logZ);
    Z = bsxfun(@minus,logZ,max(logZ));
    Z = exp(Z);
    Z = bsxfun(@rdivide,Z,sum(Z));
    F = -full(sum(sum(Z.*logZ)));
end


function [M,sigma2] = Mstep;
    M = bsxfun(@rdivide,Z*Xt,sum(Z,2));
    M = full(M);
    M(isnan(M)) = 0;    

    sq = bsxfun(@plus, bsxfun(@plus, -2*M*X, sum(M.^2,2)) , X2 );
    sigma2 = sum(sum(Z.*sq))/(N*P);
end

end


