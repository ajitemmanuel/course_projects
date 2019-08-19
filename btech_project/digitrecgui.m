function varargout = digitrecgui(varargin)
% DIGITRECGUI M-file for digitrecgui.fig
%      DIGITRECGUI, by itself, creates a new DIGITRECGUI or raises the
%      existing
%      singleton*.
%
%      H = DIGITRECGUI returns the handle to a new DIGITRECGUI or the handle to
%      the existing singleton*.
%
%      DIGITRECGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DIGITRECGUI.M with the given input arguments.
%
%      DIGITRECGUI('Property','Value',...) creates a new DIGITRECGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before digitrecgui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to digitrecgui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help digitrecgui

% Last Modified by GUIDE v2.5 26-Feb-2013 08:41:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @digitrecgui_OpeningFcn, ...
                   'gui_OutputFcn',  @digitrecgui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before digitrecgui is made visible.
function digitrecgui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to digitrecgui (see VARARGIN)

% Choose default command line output for digitrecgui
handles.output = hObject;

load MODELS
handles.models = models;    

handles.seglength = 160;    
handles.std_energy = 1.0;   
handles.std_zxings = 1.0;   
handles.noiseframes = 50;   
handles.bufflength = 10;    


adaptor = 'winsound';
chan = 1;
handles.ai = analoginput(adaptor);
addchannel(handles.ai, chan);

set(handles.ai, 'SampleRate', 8000);
set(handles.ai, 'SamplesPerTrigger', handles.seglength/2);
set(handles.ai, 'TriggerRepeat',inf);
set(handles.ai, 'TriggerType', 'immediate');
set(handles.ai, 'BufferingConfig',[2048,20]);


handles.running = 0;
handles.close = 0;


set(handles.stopbutton,'Enable','off');

handles.plot = plot(handles.plotaxis,(0:3999)/handles.ai.SampleRate, ...
    zeros(4000,1));
xlabel(handles.plotaxis,'Time (s)');

% Update handles structure
guidata(hObject, handles);


% UIWAIT makes digitrecgui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = digitrecgui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in startbutton.
function startbutton_Callback(hObject, eventdata, handles)
% hObject    handle to startbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Initialise Variables
energy_thresh_buf = zeros(handles.noiseframes,1);
zxings_thresh_buf = zeros(handles.noiseframes,1);
VAbuff = zeros(handles.bufflength,1);

VA = 0;             % "Voice Activity" flag
DETECT = 0;         
WORD = 0;           
i = 1;             
WORDbuff = zeros(handles.seglength,200);
prev_frame = zeros(handles.seglength,1);

% Setup flags to start acquisition
handles.running = 1;
handles.close = 0;


% Start acquisition
start(handles.ai)
set(handles.status,'String','STATUS: Running','ForegroundColor',[0 0 1]);
set(handles.stopbutton,'Enable','on');
set(handles.startbutton,'Enable','off');
guidata(hObject,handles); 


while handles.running
  
    newdata = getdata(handles.ai,handles.ai.SamplesPerTrigger);
    frame = [prev_frame(handles.seglength/2+1:end); newdata];

    frame = frame - mean(frame);
    
  
    frame_energy = log(sum(frame.*frame)+eps);
    frame_zxings = zerocross(frame,handles.seglength);
    
 
    if i < handles.noiseframes
        energy_thresh_buf(i) = frame_energy;
        zxings_thresh_buf(i) = frame_zxings;
    elseif i == handles.noiseframes
        energy_thresh = mean(energy_thresh_buf) + ...
            handles.std_energy*std(energy_thresh_buf);

        
        xing_thresh = max(mean(zxings_thresh_buf) + ...
            handles.std_zxings*std(zxings_thresh_buf),25);
    else

  
        if frame_energy >= energy_thresh || frame_zxings >= xing_thresh
            DETECT = 1;
        else
            DETECT = 0;
        end
        
          
        if VA   
            
            VAframes = VAframes + 1;        
            WORDbuff(:,VAframes) = frame;   
            
         
            VAbuff = circshift(VAbuff,1); 
            if DETECT 
                VAbuff(1) = 1;
            else
                VAbuff(1) = 0; 
            end
                       
            
            if VAbuff(1) 
              
                VAbuff = [1; zeros(handles.bufflength-1,1)];
            elseif VAbuff(end)
                    
                VA = 0;
                VAframes = VAframes - handles.bufflength - 1;
               
                if VAframes > 25  
                    WORD = 1;
                    WORDdata = WORDbuff(:,1:VAframes);
                    WORDbuff = zeros(handles.seglength,200); 
                else
                    WORD = 0;
                    WORDbuff = zeros(handles.seglength,200);
                end
            end                                         
        
        else    
            
           
            if DETECT
                VA = 1;                 
                VAframes = 1;           
                WORDbuff(:,1) = frame;   
                
             
                VAbuff = [1; zeros(handles.bufflength-1,1)];
            end            

        end
        
        
        if WORD
            
            mfccdata = mfcc(WORDdata,handles.ai.SampleRate,1);    
            
            
       
            nll = zeros(5,1);
            for nll_idx = 1:5
                [junk,nll(nll_idx)] = ...
                    posterior(handles.models(nll_idx).gmm,mfccdata');
            end

            [nll_VAL,nll_IDX] = min(nll);
            
        
            
             plotdata=reshape(WORDdata,1,numel(WORDdata));
            set(handles.plot,'ydata',plotdata,'xdata', ...
                (1:length(plotdata))/8000);
            axis(handles.plotaxis,'tight');
            
            
            
                wordimg = getword(nll_IDX);
           imshow(wordimg ,'parent',handles.digitaxis);
            [pathstr, name] = fileparts(wordimg);
            if nll_IDX<5
                sercomm(upper(name(1)));
            end
            
            WORD = 0;
        end

        
    end
    

    i = i+1;
    prev_frame = frame;
    
    % Update handles
    handles = guidata(hObject);
end


stop(handles.ai);
set(handles.status,'String','STATUS: Idle','ForegroundColor',[1 0 0]);
set(handles.stopbutton,'Enable','off');
set(handles.startbutton,'Enable','on');


if handles.close
    delete(handles.ai);
    closereq;
end


function numcross = zerocross(frame,seglength)

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

function word = getword(nll_IDX)

switch nll_IDX
    case 1, word = [pwd '\images\forward.bmp'];
    case 2, word = [pwd '\images\back.bmp'];
    case 3, word = [pwd '\images\left.bmp'];
    case 4, word = [pwd '\images\right.bmp'];
    case 5, word = [pwd '\images\noise.bmp'];
    
end

% --- Executes on button press in stopbutton.
function stopbutton_Callback(hObject, eventdata, handles)
% hObject    handle to stopbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


handles.running = 0;
handles.close = 0;
% Update handles structure
guidata(hObject, handles);    


% --- Executes on button press in exitbutton.
function exitbutton_Callback(hObject, eventdata, handles)
% hObject    handle to exitbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% If already running, set appropriate flags to stop the acquision and 
% close the GUI within the start_button callback function.  We don't want 
% to "stop" the ai object in the middle of a "getdata" call, otherwise we
% will see a timeout error message in the command window.  If not running, 
% then it is safe to just close the GUI from this callback.

if handles.running
    handles.running = 0;
    handles.close = 1;
    % Update handles structure
    guidata(hObject, handles);  
else
    delete(handles.ai);
    closereq;
end
