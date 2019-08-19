% loadload;

% Generate input/output spectrogram
inframes = []; outframes = [];
 [tmp1,fs] = audioread('faem0_si762.wav');
 [tmp1,tmp2] = AUDwrapper(tmp1,fs,5,0);
 tmp2 = tmp2(129:end-128,:);
 inframes  = cat(2,inframes,tmp2);
 outframes = cat(2,outframes,tmp1);
hiddenSize1 = 192;
%% Load OR Train autoencoder 
load autoenc1.mat
% autoenc1 = trainAutoencoder(inframes,hiddenSize1,...
%         'EncoderTransferFunction','logsig',...
%         'DecoderTransferFunction','logsig',...
%         'L2WeightRegularization',0.00001,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.10);
%%


 inframes_b = [inframes; ones(size(inframes,2),1)' ];   

%load toolbox-calculated weights
  W1e = autoenc1.EncoderWeights;
  W1d = autoenc1.DecoderWeights;

%make this value nonzero to introduce weight regularization
reg_param = 0;
lr = .001;
n = 1;

% load toolbox-calculated biases
 enc_biases = autoenc1.EncoderBiases;
 dec_biases = autoenc1.DecoderBiases;
% add biases
 W1e = [W1e enc_biases];
 W1d = [W1d dec_biases];
 
for epoch = 1:100

  Z1 = W1e*inframes_b;
  Y1 = sigmoid_DNN(Z1);
  Y1b = [Y1; ones(size(Y1,2),1)' ];  
  Z2 = W1d*Y1b;
  Y2 = sigmoid_DNN(Z2);
    %Use sigmoid error, with sigmoid function.
    
    delta = Y2-inframes;
    D = abs(Y2-inframes).^2;
    MSE = sum(D(:))/numel(Y2);
    err(epoch) = MSE;
    lambda = delta;
    dE2dY = 2*lambda/size(Y1b,2);
    dE2dZ = dE2dY.*Y2 .*(1-Y2);
    dE2dW = ( Y1b*dE2dZ')'/size(Y1b,2);
    
    %Update decoder weights (layer 2)
    W1d = W1d - lr * dE2dW+(lr*reg_param*W1d / n);
    
    dE2dZ1 = (W1d'*dE2dZ).*Y1b .* (1-Y1b);
    dE2dZ1(size(dE2dZ1,1),:) = [];  
    
    %Update encoder weights (layer 1)
    dE2dW1e =   dE2dZ1*inframes_b';
    W1e = W1e - lr * dE2dW1e+(lr*reg_param*W1e / n);
    
    drawnow;
    subplot(2, 1, 1);
    imagesc(Y2);
    title('Reconstructed Spectrogram')
    subplot(2, 1, 2);
    imagesc(inframes);
    title('Input Spectrogram')

end
figure;
plot(err)
title('MSE by Epoch');


   
