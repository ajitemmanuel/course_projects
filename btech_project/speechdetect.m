function  speechdetect(speech,std_energy,std_zxings)

%


if nargin == 1
    std_energy = 0.5;       % Energy STD gain factor for Voice Activity (VA)
    std_zxings = 0.5;       % Zero xing STD gain factor for VA
end


seglength = 160;                    % Length of frames
overlap = seglength/2;              % # of samples to overlap
stepsize = seglength - overlap;     % Frame stepsize
nframes = length(speech)/stepsize-1;

noiseframes = 50;           % # of frames used to estimate background noise
bufferlength = 10;          % Min # of non-VA frames to signify a break in 
                            % speech (silence between words)

% Initialise Variables
samp1 = 1; samp2 = seglength;           %Initialise frame start and end
energy_thresh_buf = zeros(noiseframes,1);
zxings_thresh_buf = zeros(noiseframes,1);
VAbuff = zeros(bufferlength,1);

VA = 0;             % "Voice Activity" flag
DETECT = 0;         % "VA indicator" flag
WORD = 0;           % "Word has been detected" flag
outdetect = zeros(size(speech));


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
            
            VAbuff = circshift(VAbuff,1); 
            if DETECT 
                VAbuff(1) = 1;
            else
                VAbuff(1) = 0; 
            end
                       
            
            if VAbuff(1) 
                
                VAbuff = [1; zeros(bufferlength-1,1)];
            elseif VAbuff(end)
                
                endframe = i-bufferlength-1;
                VA = 0;
               
                if (endframe-startframe) > 25; 
                    WORD = 1;
                    outdetect((startframe-1)*stepsize+1:endframe*stepsize) = 1;
                else
                    WORD = 0;
                end
            end                                         
        
        else    
            
            if DETECT
                VA = 1;             
                startframe = i;     
                
                
                VAbuff = [1; zeros(bufferlength-1,1)];
            end            

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

    plot(speech); hold on; plot(outdetect,'r'); hold off; axis([0 4e4 -0.5 1.1]);
end
