
function [S] = orderS(X,Y,Theta,tf)
     phi=atan2(Y,X);
%      temp1=exp(1j*Theta)+exp(1j*phi);
%      temp2=exp(1j*phi)-exp(1j*Theta);
     S_plus=mean(abs(mean(exp(1j*(phi(9*tf:10*tf,:)+Theta(9*tf:10*tf,:))),2)),1);
     S_minus=mean(abs(mean(exp(1j*(phi(9*tf:10*tf,:)-Theta(9*tf:10*tf,:))),2)),1);
%     S_plus=abs(mean(exp(1j*mean(phi(9*tf:10*tf,:)+Theta(9*tf:10*tf,:),1))));
%     S_minus=abs(mean(exp(1j*mean(phi(9*tf:10*tf,:)-Theta(9*tf:10*tf,:),1))));
    S =max(S_plus,S_minus);
end