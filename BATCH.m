close all
clear all
[p t] = iris_dataset;

maximum_epoch = 200;
epochs = 0:maximum_epoch;
learningrate = .001;
valRatio = 0.5;
trainRatio = 0.5;
hidden = 4;
vperformance = zeros(1, maximum_epoch+1);
performance = zeros(1, maximum_epoch+1);

w1 = 2*rand(hidden, size(p,1))-1;
b1 = 2*rand(hidden, 1)-1;
w2 = 2*rand(size(t,1), hidden)-1;
b2 = 2*rand(size(t,1), 1)-1;

valIndex = 1:2:150;
trainIndex = 2:2:150;
% valIndex = 1:2:size(p,2);
% trainIndex = 2:2:size(p,2);

vperformance_mat = power(t(:,valIndex) - TransferFcn_purelin(w2, TransferFcn_logsig(w1, p(:,valIndex), b1), b2), 2);
vperformance(1) = sum(vperformance_mat(:))/length(vperformance_mat(:));

performance_mat = power(t(:,trainIndex) - TransferFcn_purelin(w2, TransferFcn_logsig(w1, p(:,trainIndex), b1), b2), 2);
performance(1) = sum(performance_mat(:))/length(performance_mat(:));

for epoch = 2:(maximum_epoch+1)
    
    dw2 = 0;
    dw1 = 0;
    db2 = 0;
    db1 = 0;
    for index = trainIndex

        a1 = TransferFcn_logsig(w1, p(:,index), b1);
        a2 = TransferFcn_purelin(w2, a1, b2);
        e = t(:,index) - a2;
    
        s2 = -2*e;
        Derivative_sigmoid=[a1(1)*(1-a1(1)) 0 0 0;0 a1(2)*(1-a1(2)) 0 0;0 0 a1(3)*(1-a1(3)) 0;0 0 0 a1(4)*(1-a1(4))];
        s1 = Derivative_sigmoid*w2'*s2;
        

        dw2 = dw2 + s2*a1';
        db2 = db2 + s2;
        dw1 = dw1 + s1*p(:,index)';
        db1 = db1 + s1;

    end 

    w2 = w2 - (learningrate/length(trainIndex))*dw2;
    b2 = b2 - (learningrate/length(trainIndex))*db2;
    w1 = w1 - (learningrate/length(trainIndex))*dw1;
    b1 = b1 - (learningrate/length(trainIndex))*db1;

    vperformance_mat = power(t(:,valIndex) - TransferFcn_purelin(w2, TransferFcn_logsig(w1, p(:,valIndex), b1), b2), 2);
    vperformance(epoch) = sum(vperformance_mat(:))/length(vperformance_mat(:));
    performance_mat = power(t(:,trainIndex) - TransferFcn_purelin(w2, TransferFcn_logsig(w1, p(:,trainIndex), b1), b2), 2);
    performance(epoch) = sum(performance_mat(:))/length(performance_mat(:));   
    
end
plot(epochs, vperformance)
hold on
plot(epochs, performance)
grid on
xlabel('epoch');
ylabel('MSE');