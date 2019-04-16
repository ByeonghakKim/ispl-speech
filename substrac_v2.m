clc;clear
path1='/음성001.m4a'
path2='/음성002.m4a'
[Noise, FS1] = audioread(path1);

[Noisy, FS2] = audioread(path2);



Noisy=Noisy(1:304114,1);



tt=Noisy;
[Noise,~,~] = stft(Noise,hamming(1024),512,1024,44100);

[Noisy,~,~] = stft(Noisy,hamming(1024),512,1024,44100);



Noise_m = abs(Noise(1:(1024/2+1),:));



pha_nosy=zeros(513,592);
d_Noisy=Noisy(1:(1024/2+1),:);
d_Noise=Noise(1:(1024/2+1),:);


Noisy_m = abs(Noisy(1:(1024/2+1),:));

denosing=Noisy_m-Noise_m;
filtered_denosing=zeros(513,592,2);
for i1=1:513
    for i2=1:592
        phase_Noisy=phase(d_Noisy(i1,i2));
        phase_Noise=phase(d_Noise(i1,i2));
        phaes_x=phase_Noisy-phase_Noise;
        Y_phase=atan2(imag(phase_Noisy),real(phase_Noisy));
        N_phase=atan2(imag(phase_Noise),real(phase_Noise));
        X_phase=atan2(imag(phaes_x),real(phaes_x));
        cri=Noisy_m(i1,i2)*cos(Y_phase-X_phase)-Noise_m(i1,i2)*cos(X_phase-N_phase);
        
        if denosing(i1,i2)<=0
            filtered_denosing(i1,i2,1)=0;
            
        else
            filtered_denosing(i1,i2,1)=denosing(i1,i2)*cos(Y_phase)+j*denosing(i1,i2)*sin(Y_phase);
        end
        
        
        
        if cri<=0
            filtered_denosing(i1,i2,2)=0;
        else
            filtered_denosing(i1,i2,2)=denosing(i1,i2)*cos(Y_phase)+j*denosing(i1,i2)*sin(Y_phase);
        end
            
    end
end

[substrac_correlate,~]=istft(filtered_denosing(:,:,1), hamming(1024), hamming(1024), 512, 1024, 44100);
[substrac_uncorrelate,~]=istft(filtered_denosing(:,:,2), hamming(1024), hamming(1024), 512, 1024, 44100);
[Noisy_i,~]=istft(Noisy, hamming(1024), hamming(1024), 512, 1024, 44100);

sprintf('noisy')
sound(Noisy_i,44100)
pause(10)
sprintf('substrac_correlate')
sound(substrac_correlate,44100)
pause(10)
sprintf('substrac_uncorrelate')
sound(substrac_uncorrelate,44100)


subplot(3,1,1)
mesh(Noise_m)
title('Noisy')
view([90,90])
subplot(3,1,2)
mesh(abs(filtered_denosing(:,:,1)))
title('substrac correlate')
view([90,90])
subplot(3,1,3)
mesh(abs(filtered_denosing(:,:,2)))
title('substrac uncorrelate')
view([90,90])

