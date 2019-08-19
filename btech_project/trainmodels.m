function  trainmodels(speech,model)



Fs = 8000;                          % Sampling Frequency
seglength = 160;                    % Length of frames
overlap = seglength/2;              % # of samples to overlap
stepsize = seglength - overlap;     % Frame stepsize
nframes = length(speech)/stepsize-1;
std_energy = 0.5;           % Energy STD gain factor for Voice Activity (VA)
std_zxings = 0.5;           % Zero xing STD gain factor for VA
noiseframes = 50;           % # of frames used to estimate background noise
bufferlength = 10;          
% Initialise Variables
samp1 = 1; samp2 = seglength;           
energy_thresh_buf = zeros(noiseframes,1);
zxings_thresh_buf = zeros(noiseframes,1);
VAbuff = zeros(bufferlength,1);

VA = 0;             % "Voice Activity" flag
DETECT = 0;         % "VA indicator" flag
WORD = 0;           % "Word has been detected" flag
WORDbuff = zeros(seglength,200);
ALLdata = [];


for i = 1:nframes
    % Remove mean from analysis frame
    frame = speech(samp1:samp2)-mean(speech(samp1:samp2));

   
    frame_energy = log(sum(frame.*frame)+eps);
    frame_zxings = zerocross;
    

    if i < noiseframes
        energy_thresh_buf(i) = frame_energy;
        zxings_thresh_buf(i) = frame_zxings;
    elseif i == noiseframes
        energy_thresh = mean(energy_thresh_buf) + ...
            std_energy*std(energy_thresh_buf);
        
       
        xing_thresh = max(mean(zxings_thresh_buf) + ...
            std_zxings*std(zxings_thresh_buf),25);
    else

       
        if frame_energy >= energy_thresh || frame_zxings >= xing_thresh
            DETECT = 1;
        else
            DETECT = 0;
        end
        
          
        if VA   
            
            VAframes = VAframes + 1;        % Increment VAframe counter
            WORDbuff(:,VAframes) = frame;   % Save in buffer
            
            
            VAbuff = circshift(VAbuff,1); 
            if DETECT 
                VAbuff(1) = 1;
            else
                VAbuff(1) = 0; 
            end
          
            if VAbuff(1) 
            
                VAbuff = [1; zeros(bufferlength-1,1)];
            elseif VAbuff(end)
                    
                VA = 0;
                VAframes = VAframes - bufferlength - 1;
                
                if VAframes > 25; 
                    WORD = 1;
                    WORDdata = WORDbuff(:,1:VAframes);
                    WORDbuff = zeros(seglength,200); 
                else
                    WORD = 0;
                    WORDbuff = zeros(seglength,200);
                end
            end                                         
        
        else    
            
            
            if DETECT
                VA = 1;                 
                VAframes = 1;           
                WORDbuff(:,1) = frame; 
                            
               
                VAbuff = [1; zeros(bufferlength-1,1)];
            end            

        end
        
        
        if WORD
            ALLdata = [ALLdata WORDdata];
            WORD = 0;
        end

        
    end

   
    samp1 = samp1 + stepsize;
    samp2 = samp2 + stepsize;
    
end
  
    function numcross = zerocross
        currsum = 0;
        prevsign = 0;

        for kk = 1:seglength
            currsign = sign(frame(kk));
            if (currsign * prevsign) == -1
                currsum = currsum + 1;
            end
            if currsign ~= 0
                prevsign = currsign;
            end
        end

        numcross = currsum;

    end


mfccdata = mfcc(ALLdata,Fs,1);


if exist('MODELS.mat','file')
    load MODELS
end

modelidx = getmodelidx;
models(modelidx).word = model;
options = statset('MaxIter',500,'Display','final');
disp(['Starting GMM Training for: ' model]);
models(modelidx).gmm = gmdistribution.fit(mfccdata',8,'CovType',...
    'diagonal','Options',options);
save MODELS models

    
    function idx = getmodelidx
        switch model
            case 'forward', idx = 1;
            case 'back', idx = 2;
            case 'left', idx = 3;
            case 'right', idx = 4;
            case 'noise', idx = 5;
            otherwise, error('Invalid Word for training');
        end
    end


end
