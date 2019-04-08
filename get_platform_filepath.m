function dir=get_platform_filepath()

status = system('systeminfo');
if status==0
    os = 'windows';
else 
    status= system('uname -a');
    if status == 0
        os = 'linux';
    else
        os = 'unknown';
    end
end

% clc

dir = mfilename('fullpath');

switch os
    case 'linux'        
        i=findstr(dir,'/');
    case 'windows'
        i=findstr(dir,'\');
end
        
dir=dir(1:i(end));

cd(dir)

 end
