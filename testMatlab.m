clc
clear all
close all

% Load data and divide into training and test sets
data = readmatrix('wslp_f29.xlsx');

X = data(:,1:7); % 8 feature columns
Y = data(:,9); % Output column
m = length(Y);

PredictorNames = ["depth","qc","fs","u2","log(qt/Pa)","log(Rf)","geology","preconsolidation"]';

rng(1) % for reproducibility

% Randomizing data
p = .8;
idx = randperm(m);
xtr = X(idx(1:round(p*m)),:)
length(xtr)
ytr = Y(idx(1:round(p*m)),:)
length(ytr)
xte = X(idx(round(p*m)+1:end),:)
length(xte)
yte = Y(idx(round(p*m)+1:end),:)
length(yte)

% Create a random forest model
Mdl = TreeBagger(100,xtr,ytr,'Method','classification','minleafsize',7,'OOBPred','On',...
    'OOBPredictorImportance','on');

hte = predict(Mdl,xte);
hte = str2double(hte);
accuracy_te = mean(round(hte)==yte); %added ;
confusionchart(yte,round(hte)); %added ;

htr = predict(Mdl,xtr);
htr = str2double(htr);
accuracy_tr = mean(round(htr)==ytr); %added ;
confusionchart(ytr,round(htr)); %added ;

htall = predict(Mdl,X);
htall = str2double(htall);
accuracy_all = mean(round(htall)==Y); %added ;
confusionchart(Y,round(htall)); %added ;
