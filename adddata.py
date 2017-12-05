adder		import numpy as np
import h5py



#filenamehdf='/home/wolf/reanalysisresults/130228.0_084700SE.bin.hdf5'
#filenamebin='/home/DD1_PROFFIT/ALTZ/spec/bin_v2/SE/2013/130228.0_084700SE.bin'

def addbinsec(filenamebin,filenamehdf):
	dtbinsec1=np.dtype([('Location',str,10),('Date',str,8),('Time_eff',float),('Apparent_elev',float),('Azimuth',float),('DUR',float)])#,('Latitude',float),('Longitude',float),('Altitude',float)])
	dtbinsec2=np.dtype([('Filter',int),('OPD',float),('semiFOV',float),('APO',str,10)])
	binsec1arr=np.zeros((1),dtype=dtbinsec1)
	binsec1=binsec1arr[0]

	binsec2arr=np.zeros((1),dtype=dtbinsec2)
	binsec2=binsec2arr[0]

	f=open(filenamebin,'r')
	line=f.readline()
	while line.strip() != '$':
		line=f.readline()
		print line 

	for item in binsec1.dtype.names:
		line=f.readline().replace('\n','')
		binsec1[item]=line
	print binsec1

	while line.strip() != '$':
		line=f.readline()
		print line

	for item in binsec2.dtype.names:
		line=f.readline().replace('\n','').replace('Filter','')


		try:
			binsec2[item]=line
		except:
			print line
	print binsec2


	f.close()

	try:
		h5file=h5py.File(filenamehdf,'r+')
		dset=h5file.create_dataset('binsec1',data=binsec1)
		h5file.flush()
		h5file.close()

	except:
		print 'maybe binsec1 exists'



	try:
	
		h5file=h5py.File(filenamehdf,'r+')
		dset2=h5file.create_dataset('binsec2',data=binsec2)
		h5file.flush()
		h5file.close()
	except:
		print 'maybe binsec1 exists'
	########################################

	try:
	
		h5file=h5py.File(filenamehdf,'r+')
		dset2=h5file.create_dataset('binsec2',data=binsec2)
		h5file.flush()
		h5file.close()
	except:
		print 'maybe binsec2 exists'



###############################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
def gotonextdollar(f):
    while 1:
        character=f.read(1)
        #print character
        if character == '$':
            return f.tell()
        if not character: break
    return -1


def readbincomment(filename):
	fbin=open(filename,'r')
	pos0=gotonextdollar(fbin)
	pos1=gotonextdollar(fbin)
	pos2=gotonextdollar(fbin)
	pos3=gotonextdollar(fbin)
	pos4=gotonextdollar(fbin)
	line=fbin.readline()
	#wolf  4 octubre 2015 latitude has not been red line=fbin.readline()
	commenttypearr=[]
	values=[]
	counter=0

	while line[0] != '$' and counter < 1000:
		counter=counter+1
		print counter
		line=fbin.readline()
		print line
		if line[0] == '%':
			varname=line.split(':')[0][1:].strip()
		
			try:
				svalue=line.split(':')[1].strip()
			except:
				svalue='true'
			try: 
				value=float(svalue)
				commenttypearr.append((varname,float))
			except:
				value=svalue
				commenttypearr.append((varname,str,20))
			values.append(value)
		elif line[0] == '$':
			print 'stop'
		else:
			print line
	print 'despues while'

	fbin.close()
	print 'file is closed'
	commenttype=np.dtype(commenttypearr)
	bin_comment=np.zeros((1),dtype=commenttype)[0]
	rubros=bin_comment.dtype.names
	for i,item in enumerate(rubros):
		bin_comment[item]=values[i]
	#print bin_comment.dtype.names
	return bin_comment

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

def addbincomment(filenamebin,filenamehdf):
	print 'add bincomment'
	try:
		bin_comment=readbincomment(filenamebin)
	except:
		print ' no bincomment'
		########################################

	try:
		print 1
		h5file=h5py.File(filenamehdf,'r+')
		print 2
		dbincomment=h5file.create_dataset('bincomment',data=bin_comment)
		print 3
		h5file.flush()
		print 4
		h5file.close()
	except:
		print 'maybe bincomment exists in filenamehdf'




