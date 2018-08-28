
%Get path to simulator
display('Initializing MATLAB simulator')

paths = {'modeling','utilities'}; 

path_local_sim_init = pwd;
warned = 0;

for i = 1:length(paths) 
   addpath(genpath(strcat(path_local_sim_init, '/', paths{i}))); 
   if(exist(paths{i}) ~= 7) 
      warning('The path %s was not correctly added.  Make sure that you are in the directory of the simulator!', paths{i}); 
      warned = 1;
   end
end

if(warned == 0) 
   addpath(pwd);
   display('MATLAB simulator initialized successfully!')
end

    
