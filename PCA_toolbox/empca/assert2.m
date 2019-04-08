function [] = assert2(condition,message)

if nargin == 1,message = '';end
if isempty(message),message = 'Assert Failure.'; end
if(~condition) fprintf(1,'!!! %s !!!\n',message); end

