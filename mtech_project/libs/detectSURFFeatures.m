function Pts=detectSURFFeatures(I, varargin)
if isSimMode()
    [Iu8, params] = parseInputs(I,varargin{:});
    PtsStruct=ocvFastHessianDetector(Iu8, params);
    
else
    [I_u8, params] = parseInputs_cg(I,varargin{:});
    
    
    nRows = size(I_u8, 1);
    nCols = size(I_u8, 2);
    numInDims = 2;
    
    
    Iu8 = I_u8';
    
    
    
    
    
    
    
    
    [PtsStruct_Location, PtsStruct_Scale, PtsStruct_Metric, PtsStruct_SignOfLaplacian] = ...
        vision.internal.buildable.fastHessianDetectorBuildable.fastHessianDetector_uint8(Iu8, ...
        int32(nRows), int32(nCols), int32(numInDims), ...
        int32(params.nOctaveLayers), int32(params.nOctaves), int32(params.hessianThreshold));  
    
    PtsStruct.Location        = PtsStruct_Location;
    PtsStruct.Scale           = PtsStruct_Scale;
    PtsStruct.Metric          = PtsStruct_Metric;
    PtsStruct.SignOfLaplacian = PtsStruct_SignOfLaplacian;       
end
if params.usingROI && ~isempty(params.ROI) 
    PtsStruct.Location = bsxfun(@plus,PtsStruct.Location,...
        single([params.ROI(1) params.ROI(2)]) - single(1));
end
Pts = SURFPoints(PtsStruct.Location, PtsStruct);
function flag = isSimMode()
flag = isempty(coder.target);
function [img, params] = parseInputs(I, varargin)
validateattributes(I,{'logical', 'uint8', 'int16', 'uint16', ...
    'single', 'double'}, {'2d', 'nonempty', 'nonsparse', 'real'},...
                   'detectSURFFeatures', 'I', 1); 
if isa(I,'uint8')
    Iu8 = I;
else
    Iu8 = im2uint8(I);
end
sz = size(Iu8);
defaults = getDefaultParametersVal(sz);
parser = inputParser;
parser.CaseSensitive = true;
parser.addParamValue('MetricThreshold', defaults.MetricThreshold, @checkMetricThreshold);
parser.addParamValue('NumOctaves',      defaults.NumOctaves,      @checkNumOctaves);
parser.addParamValue('NumScaleLevels',  defaults.NumScaleLevels,  @checkNumScaleLevels);
parser.addParamValue('ROI',             defaults.ROI, @(x)vision.internal.detector.checkROI(x,sz)); 
parser.parse(varargin{:});
params.nOctaveLayers    = parser.Results.NumScaleLevels-2;
params.nOctaves         = parser.Results.NumOctaves;
params.hessianThreshold = parser.Results.MetricThreshold;
params.ROI              = int32(parser.Results.ROI);
params.usingROI = isempty(regexp([parser.UsingDefaults{:} ''],...
    'ROI','once')); 
if params.usingROI     
    img = vision.internal.detector.cropImage(Iu8, params.ROI);     
else
    img = Iu8;
end
function [img, params] = parseInputs_cg(I, varargin)
validateattributes(I,{'logical', 'uint8', 'int16', 'uint16', ...
    'single', 'double'}, {'2d', 'nonempty', 'nonsparse', 'real'},...
                   'detectSURFFeatures', 'I', 1); 
if isa(I,'uint8')
    Iu8 = I;
else
    
    h_idtc = getImageDataTypeConverter(class(I));
    Iu8 = step(h_idtc,I);
end
defaultsVal   = getDefaultParametersVal(size(Iu8));
defaultsNoVal = getDefaultParametersNoVal();
properties    = getEmlParserProperties();
optarg = eml_parse_parameter_inputs(defaultsNoVal, properties, varargin{:});
MetricThreshold = (eml_get_parameter_value( ...
        optarg.MetricThreshold, defaultsVal.MetricThreshold, varargin{:}));
NumOctaves = (eml_get_parameter_value( ...
        optarg.NumOctaves, defaultsVal.NumOctaves, varargin{:}));
NumScaleLevels = (eml_get_parameter_value( ...
        optarg.NumScaleLevels, defaultsVal.NumScaleLevels, varargin{:}));        
ROI  = eml_get_parameter_value(optarg.ROI, ...
    defaultsVal.ROI, varargin{:});
        
checkMetricThreshold(MetricThreshold);
checkNumOctaves(NumOctaves);
checkNumScaleLevels(NumScaleLevels);
usingROI = optarg.ROI ~=uint32(0);
if usingROI
    vision.internal.detector.checkROI(ROI, size(Iu8));    
end
params.nOctaveLayers    = uint32(NumScaleLevels)-uint32(2);
params.nOctaves         = uint32(NumOctaves);
params.hessianThreshold = uint32(MetricThreshold);
params.usingROI         = usingROI;
params.ROI              = int32(ROI);
if usingROI    
    img = vision.internal.detector.cropImage(Iu8, params.ROI);         
else
    img = Iu8;
end
function h_idtc = getImageDataTypeConverter(inpClass)
persistent h1 h2 h3 h4 h5
inDTypeIdx = coder.internal.const(getDTypeIdx(inpClass));
switch inDTypeIdx
      case 1 
          if isempty(h1) 
              h1 = vision.ImageDataTypeConverter('OutputDataType','uint8'); 
          end
          h_idtc = h1;   
      case 2 
          if isempty(h2) 
              h2 = vision.ImageDataTypeConverter('OutputDataType','uint8'); 
          end
          h_idtc = h2;    
      case 3 
          if isempty(h3) 
              h3 = vision.ImageDataTypeConverter('OutputDataType','uint8');
          end
          h_idtc = h3;  
      case 4 
          if isempty(h4) 
              h4 = vision.ImageDataTypeConverter('OutputDataType','uint8'); 
          end
          h_idtc = h4;  
      case 5 
          if isempty(h5)
              h5 = vision.ImageDataTypeConverter('OutputDataType','uint8'); 
          end
          h_idtc = h5;          
end
function dtIdx = getDTypeIdx(dtClass)
switch dtClass
    case 'double',
        dtIdx = 1;
    case 'single',
        dtIdx = 2;
    case 'uint16',
        dtIdx = 3;
    case 'int16',
        dtIdx = 4;
    case 'logical',
        dtIdx = 5;        
end
function defaultsVal = getDefaultParametersVal(imgSize)
defaultsVal = struct(...
    'MetricThreshold', uint32(1000), ...
    'NumOctaves', uint32(3), ...
    'NumScaleLevels', uint32(4),...
    'ROI',int32([1 1 imgSize([2 1])]));
function defaultsNoVal = getDefaultParametersNoVal()
defaultsNoVal = struct(...
    'MetricThreshold', uint32(0), ... 
    'NumOctaves',      uint32(0), ... 
    'NumScaleLevels',  uint32(0), ...
    'ROI',             uint32(0));
function properties = getEmlParserProperties()
properties = struct( ...
    'CaseSensitivity', false, ...
    'StructExpand',    true, ...
    'PartialMatching', false);
function tf = checkMetricThreshold(threshold)
validateattributes(threshold, {'numeric'}, {'scalar','finite',...
    'nonsparse', 'real', 'nonnegative'}, 'detectSURFFeatures'); 
tf = true;
function tf = checkNumOctaves(numOctaves)
validateattributes(numOctaves, {'numeric'}, {'integer',... 
    'nonsparse', 'real', 'scalar', 'positive'}, 'detectSURFFeatures'); 
tf = true;
function tf = checkNumScaleLevels(scales)
validateattributes(scales, {'numeric'}, {'integer',...
    'nonsparse', 'real', 'scalar', '>=', 3}, 'detectSURFFeatures'); 
tf = true;
