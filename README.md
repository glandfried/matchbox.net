# matchbox.net
Setup on linux:
- Install mono (needed for pythonnet) https://www.mono-project.com/download/stable/#download-lin
- Install nuget (needed for pythonnet, might need sudo) 
    curl -o /usr/local/bin/nuget.exe https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
- Install infer.net packages and copy all dlls to same folder (so pythonnet can find them) 
mkdir -p ~/dotNet/packages && cd ~/dotNet/packages
export NUGET="mono /usr/local/bin/nuget.exe install -ExcludeVersion"
${NUGET} Microsoft.ML.Probabilistic 
${NUGET} Microsoft.ML.Probabilistic.Compiler 
${NUGET} Microsoft.ML.Probabilistic.Learners
mkdir -p ~/dotNet/libs 
find ~/dotNet/packages/ -type f -name '*.dll' -exec cp -n {} ~/dotNet/libs ';'

- pip install -r requirements.txt