function [r] = orderR(Theta,tf)
%     temp=exp(1i*Theta);
%     r=abs(mean(mean(temp(9*tf:10*tf,:),2),1));
%r=abs(mean(mean(exp(1i*Theta(9*tf:10*tf,:)),2),1));
    %r=abs(mean(exp(1j*mean(Theta(9*tf:10*tf,:),1))));
    r=mean(abs(mean(exp(1j*Theta(9*tf:10*tf,:)),2)),1);
end
