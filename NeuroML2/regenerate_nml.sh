
python GenerateNeuroMLlite.py -intralaminar # generate json files for NMLlite
python GenerateNeuroMLlite.py -intralaminar -jnml # Run in jnml just to ensure they run
python checkNML.py Intralaminar.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interlaminar
python GenerateNeuroMLlite.py -interlaminar -jnml
python checkNML.py Interlaminar.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1
python checkNML.py Interareal.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -3rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -3rois
python checkNML.py Interareal_3.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -4rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -4rois
python checkNML.py Interareal_4.net.nml # Fix generated file...


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -30rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -30rois
python checkNML.py Interareal_30.net.nml # Fix generated file...


