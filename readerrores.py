import sys
sys.path.append('/home/python_toolbox/')
import sidereal
import pickle
import retrieval
import readproffwd
import os
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import fileinput


########################## error section
def errortohdf(folder,method,gasarr,targetgas,nlevel,nmw):
	print gasarr
	ngas=len(gasarr)
	porfwdinp=readproffwd.prffwd(folder)
	print 'ngas:',ngas
	ngasall=porfwdinp.ngasall
	print 'ngasall:',ngasall
	
	try:
		ferrinp=open(folder+'/errcalc9.inp','r')
	except:
		print ' no such file '+folder+'/errcalc9.inp try uppercase '+folder+'/errcalc9.inp'.upper()
		ferrinp=open(folder+'/errcalc9.inp'.upper(),'r')
	serrinp=ferrinp.read()
	def findindex(sdata,pattern):
		subserrinp=sdata
		indices=[]
		indexerrinp=1
		oldindex=0
		while len(subserrinp)> 0 and  indexerrinp > 0:	
			try:
				indexerrinp=subserrinp.index(pattern)
				subserrinp=subserrinp[indexerrinp+1:]
				indexerrinp=indexerrinp+oldindex
				oldindex=indexerrinp+1
				indices.append(indexerrinp)
			
			except:
				indexerrinp=-1
			print indexerrinp

		return indices
	indexerrinp=findindex(serrinp,'$')
	print indexerrinp



	errortyp=np.dtype([('ST',np.float32),('SY',np.float32)])
	errorgroupdistribution=np.dtype([('BASE',errortyp),('ILS',errortyp),('LOS',errortyp),('SOLAR',errortyp),('T',errortyp),('SPEC',errortyp)])
	for i,index in enumerate(indexerrinp):
		ferrinp.seek(index,0)
		print i,ferrinp.readline()[0]
		if i == 0:
			print '********** Number of the error patters in T'
			ntemp=int(ferrinp.readline())
			print 'ntemp',ntemp
		elif i==1:
			print '********** Number of partial columns of target species'
			npcols=int(ferrinp.readline())
			print 'npcols',npcols

		elif i==2:

			print '********** Channeling frequencies (4 values expected)'
			nchanel=4
			chanellingf=np.zeros((4))
			for k in range(nchanel):
				chanellingf[k]=float(ferrinp.readline())
			print chanellingf


		elif i==3:
			print '********** error scale '
			errorscaletype=np.dtype([('offset',np.float32,1),('channeling',np.float32,4),('ILSmod',np.float32,1),('ILSphase',np.float32,1),('LOS',np.float32,1),\
			('solar',np.float32,2),('Temp',np.float32,ntemp),('HITint',np.float32,ngasall),('HITgam',np.float32,ngasall)])
			errorscale=np.zeros((1),dtype=errorscaletype)
			print errorscale.dtype.names
			for rubro in errorscale.dtype.names:
				line=ferrinp.readline().replace('d','e')
				print rubro,line
	 			errorscale[rubro]=np.array(line.split(),dtype=np.float32)

			print errorscale

		elif i==4:
			print '********** error groups: '
			errorgrp=np.zeros((1),dtype=errorgroupdistribution)
			for rubro in errorgrp.dtype.names:
				line=ferrinp.readline().replace('d','e')
				line=line.replace('(','')
				line=line.replace(')','')
				print rubro,line
	 			errorgrp[rubro]['ST']=np.array(line.split(','),dtype=np.float32)[0]
				errorgrp[rubro]['SY']=np.array(line.split(','),dtype=np.float32)[1]

			print errorgrp

		elif i==5:
			print '********** error patterns in T  '
			print ntemp
			temppattern=np.zeros((nlevel,ntemp))
			for i in range(nlevel):
				line=ferrinp.readline().replace('d','e')
				vec=np.array(line.split(),dtype=np.float32)
				temppattern[i,:]=vec[:]
		
		elif i==6:
			print '********** Target species for error calculaton'
			itarget=int(ferrinp.readline())

			gas=gasarr[itarget-1]
			print itarget, gas
		elif i==7:
			print '********** partial column operator'
			partialcolumnoperator=np.zeros((nlevel,npcols))
			for i in range(nlevel):
				line=ferrinp.readline().replace('d','e')
				vec=np.array(line.split(),dtype=np.float32)
				partialcolumnoperator[i,:]=vec[:]
		

		else:
			print '**********'


		print ferrinp.readline()

	ferrinp.close()
	errorinpdtype=np.dtype([('gas',str,8),('igas',int),('ntemp',int),('npcols',int),('chfrq',float,4),('goperator',np.float32,(nlevel,npcols)),('Tpattern',np.float32,(nlevel,ntemp)),('errorscale',errorscaletype),('errorgrp',errorgroupdistribution)])
	errorinp=np.zeros((1),dtype=errorinpdtype)[0]
	errorinp['gas']=gas
	errorinp['igas']=itarget
	errorinp['ntemp']=ntemp
	errorinp['npcols']=npcols
	errorinp['chfrq'][:]=chanellingf[:]
	errorinp['goperator'][:,:]=partialcolumnoperator[:,:]
	errorinp['Tpattern'][:,:]=temppattern[:,:]
	errorinp['errorscale']=errorscale
	errorinp['errorgrp']=errorgrp


	f= h5py.File(folder+'.hdf5','r+')

	errh5dataset=f.create_dataset(gas+'/'+method+'/error/errinp',data=errorinp,dtype=errorinpdtype)
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/temppattern',data=temppattern)


	errorgrouptype=np.dtype([('BASE',np.float32, 1),('ILS',np.float32, 1),('LOS',np.float32,1),('SOLAR',np.float32,1),('T',np.float32,1),('SPEC',np.float32, 1),('NOISE',np.float32, 1)])

	STRGP=np.zeros(nlevel,dtype=errorgrouptype)
	SYRGP=np.zeros(nlevel,dtype=errorgrouptype)

	fergrp=open(folder+'/ERGRPST.DAT','r')
	gasnameold=''
	isotop=1	
	for gaserr in gasarr:
		isotop=isotop+1
		print gasnameold,gaserr,'hola'
		if gasnameold != gaserr:
                	isotop=1
		print gaserr 
		print fergrp.readline()
		for ilev in range(nlevel):
			line= fergrp.readline().split()
			for irubro,rubro in enumerate(errorgrouptype.names):
				#print rubro,irubro
				STRGP[rubro][ilev]=float(line[irubro])
		try:
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/GRP_ST_'+gaserr,data=STRGP,dtype=errorgrouptype)
		except:
			print 'isotop',isotop
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/GRP_ST_'+gaserr+'_%i' % (isotop),data=STRGP,dtype=errorgrouptype)
		gasnameold = gaserr
			
	fergrp.close()



	fergrp=open(folder+'/ERGRPSY.DAT','r')
	gasnameold=''
	isotop=1	
	for gaserr in gasarr:
		isotop=isotop+1
		if gasnameold != gaserr:
                	isotop=1
		print gaserr 
		print fergrp.readline()
		for ilev in range(nlevel):
			line= fergrp.readline().split()
			for irubro,rubro in enumerate(errorgrouptype.names):
				print rubro,irubro
				SYRGP[rubro][ilev]=float(line[irubro])
		try:
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/GRP_SY_'+gaserr,data=SYRGP,dtype=errorgrouptype)
		except:
						
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/GRP_SY_'+gaserr+'_%i' % (isotop),data=SYRGP,dtype=errorgrouptype)

		gasnameold = gaserr
			
			
	fergrp.close()
	#### Height error
	erroraltitude=np.zeros((nlevel),dtype=errortyp)
	errorstsy=np.genfromtxt(folder+'/ERDHSTSY.DAT')
	erroraltitude['ST'][:]=errorstsy[:,0]
	erroraltitude['SY'][:]=errorstsy[:,0]
	print 'DUMMY READERROR',erroraltitude
	print 'METHOD RE',method,'GAS RE',gas
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/erraltitude',data=erroraltitude)
	f.flush()

	################## covariance error

	#### Height error
	cov_Height_SY=np.genfromtxt(folder+'/ERCVHSY.DAT')
	cov_Height_ST=np.genfromtxt(folder+'/ERCVHST.DAT')

	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_Height_SY',data=cov_Height_SY)
	f.flush()
	f.close()
	f= h5py.File(folder+'.hdf5','r+')
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_Height_ST',data=cov_Height_ST)
	f.flush()
	
	cov_all_ST=np.genfromtxt(folder+'/ERCVTOST.DAT')
	cov_all_SY=np.genfromtxt(folder+'/ERCVTOSY.DAT')

	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_all_SY',data=cov_all_SY)
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_all_ST',data=cov_all_ST)


	##
	cov_target_ST=np.genfromtxt(folder+'/ERCVSPST.DAT',skip_header=1)
	print 'COV_TARGET_ST',cov_target_ST.shape
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_target_ST',data=cov_target_ST)
	f.flush()
	cov_target_SY=np.genfromtxt(folder+'/ERCVSPSY.DAT',skip_header=1)
	print 'COV_TARGET_SY',cov_target_SY.shape
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_target_SY',data=cov_target_SY)
	f.flush()
	diag_target_ST=np.zeros(nlevel,dtype=float)
	diag_target_SY=np.zeros(nlevel,dtype=float)
	for i in range(nlevel):
		diag_target_ST[i]=np.sqrt(cov_target_ST[i,i])
		diag_target_SY[i]=np.sqrt(cov_target_SY[i,i])
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/diag_target_ST',data=diag_target_ST)
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/diag_target_SY',data=diag_target_SY)

	#################### 
	COV_PCOLS_ST=np.genfromtxt(folder+'/ERCVCLST.DAT')
	COV_PCOLS_SY=np.genfromtxt(folder+'/ERCVCLSY.DAT')
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_PCOLS_ST_target',data=COV_PCOLS_ST)
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/COV_PCOLS_SY_target',data=COV_PCOLS_SY)
	print 'npcols',npcols
	err_ST=np.zeros((npcols))	
	err_SY=np.zeros((npcols))	
	err_TOT=np.zeros((npcols))	
	if npcols > 1:					### JORGE 22/06/2016, IN CASE THERE'S ONLY ONE PARTIAL COLUMN
		for ipcol in range(npcols):	
			err_ST[ipcol]=(COV_PCOLS_ST[ipcol,ipcol])**0.5/10000.0
			err_SY[ipcol]=(COV_PCOLS_SY[ipcol,ipcol])**0.5/10000.0
			err_TOT[ipcol]=(COV_PCOLS_SY[ipcol,ipcol]+COV_PCOLS_ST[ipcol,ipcol])**0.5/10000.0
	else:
		err_ST=(COV_PCOLS_ST)**0.5/10000.0
		err_SY=(COV_PCOLS_SY)**0.5/10000.0
		err_TOT=(COV_PCOLS_SY+COV_PCOLS_ST)**0.5/10000.0

	print 'err_TOT',err_TOT
	#total column avkernel for targetgas
	errcolsen=np.genfromtxt(folder+'/ERCOLSEN.DAT')
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/ERCOLSEN_target',data=errcolsen)
	##################### err jakobian
	errorjakobian=np.genfromtxt(folder+'/JAKERR.DAT')
	errh5dataset=f.create_dataset(gas+'/'+method+'/error/JAKERR',data=errorjakobian)

	################3 errorpattern
	errorpatterntypenames=[]
	for rubro in errorscaletype.names:
		if rubro == 'channeling':
			for i in range(4*2*nmw):
				errorpatterntypenames.append((rubro+'_%i' % (i),np.float32,1))
		elif rubro == 'Temp':
			for i in range(ntemp):
				errorpatterntypenames.append((rubro+'_%i' % (i),np.float32,1))
		elif rubro == 'SOLAR':
			for i in range(ntemp):
				errorpatterntypenames.append((rubro,np.float32,2))
		
		elif rubro == 'HITint':
			for i in range(ngasall):
				errorpatterntypenames.append((rubro+'_%i' % (i),np.float32,1))
		elif rubro == 'HITgam':
			for i in range(ngasall):
				errorpatterntypenames.append((rubro+'_%i' % (i),np.float32,1))
		else:
			errorpatterntypenames.append((rubro,np.float32,1))




	#errorpatterntypenames.append(('Noise',np.float32,1))	
	errorpatterntype=np.dtype(errorpatterntypenames)

	ferrpattern=open(folder+'/ERRPATTS.DAT','r')
	isotop=1
	for gaserr in gasarr:
		isotop=isotop+1
		if gasnameold != gaserr:
                	isotop=1

		errpattern=np.zeros((nlevel),dtype=errorpatterntype)
		print gaserr 
		print ferrpattern.readline()
		for ilev in range(nlevel):
			line= ferrpattern.readline().split()
			for irubro,rubro in enumerate(errorpatterntype.names):
				print rubro,irubro,ilev
				#print line
				try:
					print line[irubro]
				
					errpattern[rubro][ilev]=float(line[irubro])
				except:
					print 'mimodo'	
		
		try:
			
						
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/errpat_'+gaserr,data=errpattern,dtype=errorpatterntype)

		except:
			print gaserr, isotop
			errh5dataset=f.create_dataset(gas+'/'+method+'/error/errpat_'+gaserr+'_%i' % (isotop),data=errpattern,dtype=errorpatterntype)
		gasnameold = gaserr
	ferrpattern.close()


	f.flush()
	f.close()
	return partialcolumnoperator,err_ST,err_SY,err_TOT
# for test
# folder='/home/DD1_PROFFIT/ALTZ/results/CO_NDACC/150511.1/150511.1_073039SD.bin'
# method='CO_NDACC'
# nlevel=41
# targetgas=5
# nmw=3
# gasarr=['H2O','CO2','O3','N2O','CO','CH4']

# errortohdf(folder,method,gasarr,targetgas,nlevel,nmw)
	










