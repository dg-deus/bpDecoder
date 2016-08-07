%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% < Belief-Propagation Algorithm(BPA) / Probabilistic Decoder >
%
% Date      : 16.08.07
% Author    : Yongseen Kim
% Version   : 1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [xhat] = decode_bpa(y,No,max_iter)
% y: rcvd vector from the channel
% No: given SNR of BI-AWGNC


% Here, H is the parity check matrix.
% * This version assumes H is regular. You may change H matrix on your own.
load 128x256regular.mat H

% n = # of columns in H = # of vairable nodes
% k = # of rows in H = # of check nodes, **In fact, k = n - dim(H)
[k,n] = size(H);
xhat = zeros(n,1);
ghat = zeros(n,1);

% [r,c] corresponds to indices for non-zero elements in H
[r,c] = ind2sub(size(H), find(H==1));


%% Iniitiliazation
alpha = zeros(k,n);
beta = zeros(k,n);
gamma = (2*y) ./ (No.^2);   % LLR computation, No here is cont.

for i=1:length(r)
    alpha(r(i),c(i)) = gamma(c(i));
end

%% Iteration Loop
for imax_iter = 1:max_iter
    % Check-to-vairable messages, beta computation
    for i=1:k
        cIdx = find(r==i);  % Check node access from 1st check node
        for j=1:length(cIdx)
            signTemp = 1;
            sumTempB = 0;
            
            % Compute sign and absolute values for all variables nodes
            for l=1:length(cIdx)
                alphaValue = alpha(i,c(cIdx(l)));
                if(alphaValue < 0)
                    signTemp = -signTemp;
                end
                sumTempB = sumTempB + log ( (1 + exp(-abs(alphaValue))) / ...
                                    (1 - exp(-abs(alphaValue))) );
            end
            
            % remove each intrinsic information for extrinsic message
            alphaDel = alpha(i,c(cIdx(j)));
            if(alphaDel < 0)
                signTemp = - signTemp;
            end
            sumTempB = sumTempB - log ( (1 + exp(-abs(alphaDel))) / ...
                                    (1 - exp(-abs(alphaDel))) );
            
            % compute beta, extrinsic info
            beta(i,c(cIdx(j))) = signTemp * log ( (1 + exp(-abs(sumTempB))) / ...
                                                (1 - exp(-abs(sumTempB))) );
        end % for
    end % for


    % Variable-to-check messages, A posteriori info, Hard decision, Code check
    for i=1:n
        vIdx = find(c==i);
        
        % Compute alpha
        for j=1:length(vIdx)
            sumTempA = 0;
            
            % Add all messages from check nodes for the i-th variable node
            for l=1:length(vIdx)
                sumTempA = sumTempA + beta(r(vIdx(l)),i);
            end
            
            % remove each intrisinc info.
            sumTempA = sumTempA - beta(r(vIdx(j)),i);
            
            % Compute alpha, extrinsic info
            alpha(r(vIdx(j)),i)= gamma(i) + sumTempA;
        end
        
        % A posteriori info
        for j=1:length(vIdx)
            ghat(i) = ghat(i) + beta(r(vIdx(j)),i);
        end
        ghat(i) = ghat(i) + gamma(i);
        
        % Hard decision
        if(ghat(i) >= 0)
            xhat(i) = 0;
        else
            xhat(i) = 1;
        end
    end
    
    % check whether it is a codeword and stop decoding
    if(sum(mod(H*xhat,2)) == 0)
        break;
    end
    
end %for









