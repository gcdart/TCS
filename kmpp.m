% Copyright by Siddharth Gopal, Carnegie Mellon University, gcdart@gmail.com
% Implements the Kmeans++ algorithm
%  Inputs : 1. X (PxN matrix) : training examples 
%           2. K              : number of clusters
%           3. options
%  Outputs: 1. Z (KxN matrix) : assignement of points to clusters means.
%           2. M (KxP matrix) : cluster means
%
%  options: 
%           seed : random seed
%           verbose = {0 or 1}
function [Z,M] = kmpp( X, K, options )

     [P,N] = size( X );     

     Xt = X';
     X2 = sum(X.^2);
     rand('twister',options.seed);

     Z = sparse(K,N);
     M = zeros(K,P);

     ind = randsample(N,1);
     Z(1,ind) = 1;
     M(1,:) = X(:,ind)';
     M2 = sum(M.^2,2);

     for k = 1:(K-1)
         DMat = bsxfun(@plus, bsxfun(@plus, -2*M(1:k,:)*X, M2(1:k) ) , X2 );

         if k > 1
             D = min(DMat);
         else
             D = DMat;
         end
         D(D<1e-12) = 0;

         if sum(D) == 0
             D = ones(1,length(D));
         end

         ind = randsample(N,1,true,D);
         
         Z(k+1,ind) = 1;
         M(k+1,:) = X(:,ind)';
         M2(k+1,:) = sum( X(:,ind).^2 );
         if options.verbose == 1
             fprintf(1,' Selecting index %d\n',ind);
         end
     end
     

end