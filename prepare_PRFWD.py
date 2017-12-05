import sys
import readbinheader
import time 


retrievalstrategy=sys.argv[1]
relativespecpath=sys.argv[2]+'/'
relativespecpathwin=relativespecpath.replace('/','\\')
print relativespecpath
print relativespecpathwin
#_= raw_input("press enter to continue")

binfile=sys.argv[3]
workdirectory=sys.argv[4]+'/'

#############read skeleton prffwd
filenamein=retrievalstrategy+'/PRFFWD.INP'
f=open(filenamein,'r')
data=f.read()
f.close()
print type(data)
####

################################################################
binspec_comodin='%MESFILE'
data=data.replace(binspec_comodin,relativespecpathwin+binfile)
   

#############to be red from bin
Date_comodin='%DATE'
OPD_comodin='%OPD-MAX'
APD_comodin='%APD'
SEMIFOV_comodin='%SEMIFOV'

ILS_comodin='%ILS'

JD_comodin='%JULDATE'
SZA_comodin='%APPELEV'   
AZA_comodin='%AZIMUTH'

print "ATENCION ",relativespecpath+binfile
binspec=readbinheader.readbinheader(relativespecpath+binfile)
print binspec.ILSparams
Date=binspec.Date

OPD=str(binspec.OPDmax)
if (binspec.APD.strip() == '' or binspec.APD.strip() == 'BX' or binspec.APD.strip() == '0'):
	APD=str(1)
if binspec.APD.strip() == 'B3':
	APD = str(4)
if binspec.APD.strip() == 'NB':
	APD = str(7)
if binspec.APD.strip().isdigit():
	APD = str(binspec.APD.strip())
SEMIFOV=str(binspec.SEMIFOV)
print "ILS type"
print type(binspec.ILSparams), len(binspec.ILSparams)
##########################  se cambio esta parte para incluir ILS experimental (celda) para cada estacion UNAM y ALTZ ###########

if (binspec.Location == "CCA") or (binspec.Location == "UNAM"):
	print "LOCATION: ",binspec.Location
	ils_path = "/home/D2_PROFFIT/OPUS/ILS/UNAM/ilsparms_HBr_20140613.dat"
	f = open(ils_path,"r")
	ils_lines = f.readlines()
	f.close()
	ILS = ''
	for jlj in ils_lines:
		ILS = ILS + jlj.strip()+"\n"
	print ILS
if (binspec.Location == "Altzomoni") or (binspec.Location == "ALTZ"):
	print "LOCATION: ",binspec.Location

	try:
		ILS="".join(binspec.ILSparams)
	except:
		ILS=' 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n' \
			+' 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n'  \
			+' 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n'  \
			+' 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n 1.0 0.0 \n' 

########################################################################################################################
JD=str(binspec.JD)
SZA=str(binspec.SZA)
AZA=str(binspec.AZA)

print Date,OPD,APD,SEMIFOV,JD,SZA,AZA

print "ILS: ", ILS
 ############### replace upper comodines
lista=[Date,OPD,APD,SEMIFOV,ILS,JD,SZA,AZA]
lista_comodin=[Date_comodin,OPD_comodin,APD_comodin,SEMIFOV_comodin,ILS_comodin,JD_comodin,SZA_comodin,AZA_comodin]
for i,oldsubstring in enumerate(lista_comodin):
    newsubstring=lista[i]
    try:
        index=data.index(oldsubstring)
        print oldsubstring,newsubstring
	if i == 4:
		print "in loop: ", newsubstring
        data=data.replace(oldsubstring,newsubstring)
    except:
        print 'EXCEPTION: NOT FOUND',oldsubstring
        
        
################## end to be red from binfile

            


###############################################################

DATEFOLDER_comodin='%DATEFOLDER'
PT_comodin='%PT-FILE'
SOLSKAL_comodin='%SOLSKAL'
VMR_CLI_comodin='%VMR-CLI'
VMR_BESTguess_comodin='%VMR-BST'

newsubstring='..\\..\\pt_profiles\\pt_ncep_altz\\'
oldsubstring=SOLSKAL_comodin
print oldsubstring,newsubstring
try:
	data=data.replace(oldsubstring,newsubstring)
except:
	print 'EXCEPTION: NOT FOUND',oldsubstring


#oldsubstring=JD_comodin
#
#index=data.index(oldsubstring)
#wordlen=len(oldsubstring)
#print data[index:index+wordlen]
##### just for check end
#
#newsubstring=' hola hxsagdlkhsagdlkhsagdsaglksaclksacglksagxxxx'
#data=data.replace(oldsubstring,newsubstring)
#####  check  the change
#print data[index:index+wordlen]
fout=open(workdirectory+'inp_fwd/PRFFWD.INP','w')
fout.write(data)
fout.close()
print 'prepare_PRFWD terminated correctly' 
