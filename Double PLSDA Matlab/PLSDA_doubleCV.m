function [error,sensitivity,specificity,LV_best] = PLSDA_doubleCV(Spectra, ClinicalInfo,patientnumber,percent_outer,percent_inner,n_outer,n_inner); 

% PLSDA with double cross validation, for two groups
% Performs chosen number of inner and outer loops 
% Nb, Autoscales data, change if you only want mean centering
%
% This scrips is applicable also when more than one sample is present for
% each patient. All samples put in either training/test/valudation set.
% Does not balance the number of spectra from group 1 and 2 in
% training/test sets
%
% Calculates models for 10 LV, increase if you think necessary
%
% [error,sensitivity,specificity,LV_best] = PLSDA_doubleCV(Spectra, ClinicalInfo,patientnumber,percent_outer,percent_inner,n_outer,n_inner); 
% 
% Input: Spectra: Input data, spectra or other covariates
%        ClinicalInfo: vector of 1 and 2
%        patientnumber: vector of patientnumbers
%        percent_outer/inner: percentage of samples in test set
%        n_outer: number of outer loops (for instance 20)
%        n_inner: number of inner loops (for instance 20)
%Output: error: vector of errors of outer loop
%        sensitivity: vector of sensitivity of outer loop
%        specificity: vector of specificity of outer loop
%        LV_best: chosen LV for each outer loop
%
% 04.03.2016, GFG, updated 040418,GFG

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts               = plsda('options');
opts.preprocessing = {preprocess('default','autoscale') preprocess('default','meancenter')};
opts.plots         = 'none';
opts.display       =  'off';  
opts.waitbartrigger = [inf]; %ikke popup-vindu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
tic;
spec= [patientnumber,ClinicalInfo,Spectra];
ytre = 1;

while ytre < n_outer+1; %number of outer loops
    
    % Splitting into an outer test and training set:
    [Xtrain,Xtest,Ctrain,Ctest,Ptrain,Ptest] = trainingtestsets(spec,percent_outer);
    indre = 1;
    
        [en_ytre,x] = size(find(Ctest==1));
        [to_ytre,x] = size(find(Ctest==2));
        
        if en_ytre == 0;
        elseif to_ytre == 0;
            ytre = ytre;
        else
    
   
    while indre < n_inner+1; %number of repetitions of inner loop
        
         Xtrain_input = [Ptrain, Ctrain, Xtrain];
        [XtrainIn,XtestIn,CtrainIn,CtestIn,PtrainIn,PtestIn] = trainingtestsets(Xtrain_input,percent_inner);
        
        [en,x] = size(find(CtestIn==1));
        [to,x] = size(find(CtestIn==2));
                

        if en == 0;
        elseif to == 0;
          
            indre = indre; 
        else
       
            
        
        
        for i = 1:10  %PLSDA for LV 1-10, increase if necessary
            modell = plsda(XtrainIn,CtrainIn,i,opts);
            predikert = plsda(XtestIn,CtestIn, modell,opts); 
            errorLVs(indre,i) = predikert.detail.classerrp(1,i);
            
           end
        
        indre = indre + 1;
        end
 
        
    end
        
        errorLVs_mean = mean(errorLVs,1); %find mean classification error over inner loops for each LV 
        Firstmin = find(diff(errorLVs_mean)>0,1,'first'); %find first minimum in classification error
        if isempty(Firstmin)==1
            LV_best(ytre)= length(Firstmin);
        else
            LV_best (ytre) = Firstmin;
        end
    
    modell = plsda(Xtrain,Ctrain,LV_best(ytre),opts);
    predikert = plsda(Xtest,Ctest, modell,opts); 
    error(ytre) = predikert.detail.classerrp(1,LV_best(ytre));
    sensitivity(ytre) = 1 - predikert.detail.misclassedp{1}(2, LV_best(ytre));
    specificity(ytre) =1 - predikert.detail.misclassedp{1}(1, LV_best(ytre));
        
    ytre = ytre+ 1;
        end
    
    
end
end
    