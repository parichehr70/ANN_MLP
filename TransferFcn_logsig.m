function [a] = TransferFcn_logsig(w,p,B)
n=w*p + B;
a=1./(1+exp(-n));
end