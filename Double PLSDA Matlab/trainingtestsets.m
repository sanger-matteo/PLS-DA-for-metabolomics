function [Xtrain,Xtest,Ctrain,Ctest,Ptrain,Ptest,P] = trainingtestsets(X,Percent); 
%
%
% Makes training and test sets, where all samples from one patient is in
% either training or test
%
% Input:  X          The data matrix (each row being one sample), first
%                    column patientnumber, second column
%                    class/characteristic, then spectra
%         Percent          Percentage in test set 
% Output: Xtrain     Training spectra
%         Xtest      Test spectra
%         Ctrain     Vector of class labels/characteristics for the training set
%         Ctest      Vector of class labels/characteristics for the test set
%         Ptrain     Vector of patient numbers for the training set
%         Ptest      Vector of patient numbers for the test set
% GFG, 2016

    
a = unique(X(:,1));
a2 = shuffle(a);


S = [];
T = [];

NumberOfPatients = length(a2);

P = round(NumberOfPatients(1)*Percent/100);


    
    for j = 1:P
        provenr_test = find(X(:,1) == a2(j));
        S = [S;provenr_test];
    end
    
    
    for j = (P+1):length(a);
        provenr_trening = find(X(:,1) == a2(j));
        T = [T;provenr_trening];
    end
    
    
    testmat = X(S,:);
    treningsmat= X(T,:);
    
    Xtrain = treningsmat(:,3:end);
    Xtest = testmat(:,3:end);
    Ctrain = treningsmat(:,2);
    Ctest = testmat(:,2);
    Ptrain = treningsmat(:,1);
    Ptest = testmat(:,1);
end