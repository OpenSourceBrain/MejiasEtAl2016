### Quick hack to ensure generated NML files run on OSB...

import sys

str_out = ''

    
with open(sys.argv[1]) as fp:  
   added = False
   for line in fp:
       #print('L: %s'%line)
       str_out+='%s'%line
       if '</notes>' in line and not added:
           str_out+='''    <include href="Prototypes.xml"/>
    <include href="RateBased.xml"/>
    <include href="NoisyCurrentSource.xml"/>
'''
           added = True

#print str_out

f = open(sys.argv[1],'w')
f.write(str_out)
f.close()

