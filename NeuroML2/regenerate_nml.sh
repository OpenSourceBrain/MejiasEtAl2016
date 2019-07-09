
python GenerateNeuroMLlite.py -intralaminar # generate json files for NMLlite
python GenerateNeuroMLlite.py -intralaminar -jnml # Run in jnml just to ensure they run


python GenerateNeuroMLlite.py -interlaminar
python GenerateNeuroMLlite.py -interlaminar -jnml


python GenerateNeuroMLlite.py -interareal -stimulate_V1
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -3rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -3rois


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -4rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -4rois


python GenerateNeuroMLlite.py -interareal -stimulate_V1 -30rois
python GenerateNeuroMLlite.py -interareal -nml -stimulate_V1 -30rois


