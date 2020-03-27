
python GenerateNeuroMLlite.py -intralaminar # generate json files for NMLlite
python GenerateNeuroMLlite.py -intralaminar -jnml -duration=1000 # Run in jnml just to ensure they run
##python checkNML.py Intralaminar.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interlaminar
python GenerateNeuroMLlite.py -interlaminar -jnml -duration=1000 
#python checkNML.py Interlaminar.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -duration=1000 
#python checkNML.py Interareal.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -3rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -3rois -duration=1000 
#python checkNML.py Interareal_3.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -4rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -4rois -duration=1000 
#python checkNML.py Interareal_4.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -30rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -30rois -duration=1000 
#python checkNML.py Interareal_30.net.nml # Fix generated file...


