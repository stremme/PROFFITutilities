

import numpy as np
import matplotlib.pyplot as plt
import h5py



#erroranalysis(filename,method,gas, plotswitch=0,ADDHDF=True,printflag=False)



def errorgroup(orgname):
	name=orgname.split('_')[0]
	if name == 'offset' or name == 'channeling' :
		return 'BASE'
	elif name[0:3] == 'ILS':
		return 'ILS'
	elif name[0:3] == 'LOS':
		return  'LOS'
	elif name == 'solar':
		return  'SOLAR'
	elif name[0:4] == 'Temp':
		return  'T'
	elif name[0:3] == 'HIT':
		return  'SPEC'
	else:
		return 'unknown'

def  erroranalysis(filename,method,gas, plotswitch=0,partcolumnoperators=[],ADDHDF=False,printflag=True):
	
	f=h5py.File(filename,'r+')
	
	if partcolumnoperators==[] and ADDHDF==True:
		partcolumnoperators=np.array(f[gas+'/'+method+'/error/partialOP']).T
	else:
		ADDHDF=False
		

	dsetresults=np.array(f[gas+'/'+method+'/result'])

	nlevel=len(dsetresults['z'])
	colerrtype=np.dtype([('BASE',float),('ILS',float),('LOS',float),('SOLAR',float),('T',float),('SPEC',float),('NOISE',float),('ALL',float)])
	coverrtype=np.dtype([('BASE',float,(nlevel,nlevel)),('ILS',float,(nlevel,nlevel)),('LOS',float,(nlevel,nlevel)),('SOLAR',float,(nlevel,nlevel)),('T',float,(nlevel,nlevel)),('SPEC',float,(nlevel,nlevel)),('NOISE',float,(nlevel,nlevel)),('ALL',float,(nlevel,nlevel))])

	allcolerrs=np.zeros((1),dtype=colerrtype)[0]
	allcoverrs=np.zeros((1),dtype=coverrtype)[0]
	STcolerrs=np.zeros((1),dtype=colerrtype)[0]
	STcoverrs=np.zeros((1),dtype=coverrtype)[0]
	SYcolerrs=np.zeros((1),dtype=colerrtype)[0]
	SYcoverrs=np.zeros((1),dtype=coverrtype)[0]

	if not(partcolumnoperators == []):
		npartcols=len(partcolumnoperators)
		STpartcolumnerrs=np.zeros((npartcols),dtype=colerrtype)
		SYpartcolumnerrs=np.zeros((npartcols),dtype=colerrtype)
		allpartcolumnerrs=np.zeros((npartcols),dtype=colerrtype)

	dset_errorinp=f[gas+'/'+method+'/error/errinp']

	if plotswitch==1:
		if printflag: print dset_errorinp.dtype.names
		for name in dset_errorinp.dtype.names:
			if printflag: print name
			if printflag: print dset_errorinp[name]
			if printflag: print dset_errorinp['errorgrp'].dtype.names
	np.array(dset_errorinp)['errorgrp']['T']['ST']
	
	pathdset= gas+'/'+method+'/error/GRP_ST_'+gas
	if plotswitch==1:
		if printflag: print pathdset
	dset_grp_ST=f[pathdset]
	dset_grp_SY=f[gas+'/'+method+'/error/GRP_SY_'+gas]

	if plotswitch==1:
		for name in dset_grp_ST.dtype.names:
			plt.plot(dset_grp_ST[name],dsetresults['z'])
			diag_target_ST=f[gas+'/'+method+'/error/diag_target_ST']
			plt.plot(diag_target_ST,dsetresults['z'],'--')
			plt.legend(dset_grp_ST.dtype.names)
		plt.show()

	dset_errpat=f[gas+'/'+method+'/error/errpat_'+gas]
	dest_parcols=f[gas+'/'+method+'/error/partialcolumns']
	totcolret=dest_parcols['ret'][0]

	
	if plotswitch==1:
		if printflag: print dset_errpat.dtype.names

	airmass=f[gas+'/'+method+'/retrieval']['airmass']
	nlevel=len(airmass)
	covar=np.zeros((nlevel,nlevel),dtype=np.float64)
	covarST=np.zeros((nlevel,nlevel),dtype=np.float64)
	covarSY=np.zeros((nlevel,nlevel),dtype=np.float64)
	covarall=np.zeros((nlevel,nlevel),dtype=np.float64)
	strcolerrortype=[(name,float) for name in dset_errpat.dtype.names]
	for name in dset_errpat.dtype.names:
		error=np.dot(airmass.T,dset_errpat[name])*1E-6*1E-4
		groupname=errorgroup(name)
		alpha_ST=dset_errorinp['errorgrp'][groupname]['ST']
		factor=alpha_ST**2#+(1.0-alpha_ST)**2
		factor2=(1.0-alpha_ST)**2
		cst=factor*np.outer(dset_errpat[name],dset_errpat[name])
		covarST=covarST+cst
		STcoverrs[groupname]=STcoverrs[groupname]+cst
		csy=factor2*np.outer(dset_errpat[name],dset_errpat[name])
		covarSY=covarSY+csy
		SYcoverrs[groupname]=SYcoverrs[groupname]+csy
		allcoverrs[groupname]=allcoverrs[groupname]+cst+csy
	
		SYcolerrs[groupname]=np.sqrt(SYcolerrs[groupname]**2+factor *error**2)

		STcolerrs[groupname]=np.sqrt(STcolerrs[groupname]**2+factor2*error**2)
	
		allcolerrs[groupname]=np.sqrt(allcolerrs[groupname]**2+(factor+factor2)*error**2)

		covarall=covarall+csy+cst

		
		if plotswitch==1:
			if printflag: print name, error,' %5.3f ' % (error/totcolret*100),'%'

	
	COV_target_ST=f[gas+'/'+method+'/error/COV_target_ST']
	COV_target_SY=f[gas+'/'+method+'/error/COV_target_SY']
	cov_noise=COV_target_ST-covarST
	STcoverrs['NOISE']=cov_noise
	allcoverrs['NOISE']=cov_noise
	STcolerrs['NOISE']=np.sqrt(np.dot(airmass.T,np.dot(cov_noise,airmass))*(1E-6*1E-4)**2)
	allcolerrs['NOISE']=STcolerrs['NOISE']

	for groupname in SYcolerrs.dtype.names[:-1]:
		SYcolerrs['ALL']=np.sqrt(SYcolerrs['ALL']**2+SYcolerrs[groupname]**2)
		STcolerrs['ALL']=np.sqrt(STcolerrs['ALL']**2+STcolerrs[groupname]**2)
		allcolerrs['ALL']=np.sqrt(allcolerrs['ALL']**2+allcolerrs[groupname]**2)

	
	covar=covarST+covarSY
	dset_grp_SY=f[gas+'/'+method+'/error/GRP_SY_'+gas]
	noiseerr=np.zeros((nlevel))
	syserr=np.zeros((nlevel))
	for i in range(nlevel):
		noiseerr[i]=np.sqrt(cov_noise[i,i])
		syserr[i]=np.sqrt(covarSY[i,i])

	
	if not(partcolumnoperators == []):
		
		for kkk,goperator in enumerate(partcolumnoperators):
			
			gairmass=goperator*airmass
			if printflag: print goperator
			if printflag: print gairmass
			call=np.zeros((nlevel,nlevel))
			callst=np.zeros((nlevel,nlevel))
			callsy=np.zeros((nlevel,nlevel))
			for groupname in allcolerrs.dtype.names:
				if printflag: print groupname
				cst=STcoverrs[groupname]
				csy=SYcoverrs[groupname]
				cstmcsy=cst+csy
				call=call+allcoverrs[groupname]
				callst=callst+cst
				callsy=callsy+csy
			
				stcol=np.sqrt(np.dot(gairmass.T,np.dot(cst,gairmass))*(1E-6*1E-4)**2)
				sycol=np.sqrt(np.dot(gairmass.T,np.dot(csy,gairmass))*(1E-6*1E-4)**2)
				if printflag: print stcol
				STpartcolumnerrs[kkk][groupname]=stcol
				SYpartcolumnerrs[kkk][groupname]=sycol
				allpartcolumnerrs[kkk][groupname]=np.sqrt(np.dot(gairmass.T,np.dot(cstmcsy,gairmass))*(1E-6*1E-4)**2)
			allpartcolumnerrs[kkk]['ALL']=np.sqrt(np.dot(gairmass.T,np.dot(call,gairmass))*(1E-6*1E-4)**2)
			STpartcolumnerrs[kkk]['ALL']=np.sqrt(np.dot(gairmass.T,np.dot(callst,gairmass))*(1E-6*1E-4)**2)
			SYpartcolumnerrs[kkk]['ALL']=np.sqrt(np.dot(gairmass.T,np.dot(callsy,gairmass))*(1E-6*1E-4)**2)
	
	if plotswitch==1:
		plt.plot(dset_grp_ST['NOISE'],dsetresults['z'])
		plt.plot(noiseerr,dsetresults['z'],'r--')
		#plt.plot(noiseerr2,dsetresults['z'])

		plt.plot(dset_grp_SY['NOISE'],dsetresults['z'],'g-')
		plt.plot(syserr,dsetresults['z'],'k--')
		#plt.plot(noiseerr2,dsetresults['z'])

	
		error_ST=np.dot(airmass.T,np.dot(COV_target_ST,airmass))
		if printflag: print np.sqrt(error_ST)*1E-6*1E-4 # ppm cm2/m2

		error_tot=np.dot(airmass.T,np.dot(covar,airmass))
		if printflag: print 'total de pattern:'
		if printflag: print np.sqrt(error_tot)*1E-6*1E-4 # ppm cm2/m2
		if printflag: print np.sqrt(error_tot)*1E-6*1E-4 /totcolret*100 ,'%'# ppm cm2/m2
		covar=np.array(COV_target_ST)+np.array(COV_target_SY)
		error_tot=np.dot(airmass.T,np.dot(covar,airmass))
		error_noise=np.dot(airmass.T,np.dot(cov_noise,airmass))
		if printflag: print 'total:'
		if printflag: print np.sqrt(error_tot)*1E-6*1E-4 # ppm cm2/m2
		if printflag: print np.sqrt(error_tot)*1E-6*1E-4 /totcolret*100 ,'%'# ppm cm2/m2
		if printflag: print 'Noise:'
		if printflag: print np.sqrt(error_noise)*1E-6*1E-4 # ppm cm2/m2
		if printflag: print np.sqrt(error_noise)*1E-6*1E-4 /totcolret*100 ,'%'# ppm cm2/m2
		plt.show()

		

	if printflag: print 'ADDHDF=',ADDHDF
	
	if ADDHDF:
		if printflag: print 'ADDHDF=',ADDHDF
		dsetname=gas+'/'+method+'/error/allpartcolumnerrs_'+gas
		if printflag: print dsetname
		dset1NEW=f.create_dataset(gas+'/'+method+'/error/allpartcolumnerrs',data=allpartcolumnerrs)
		dset2NEW=f.create_dataset(gas+'/'+method+'/error/STpartcolumnerrs',data=STpartcolumnerrs)
		dset3NEW=f.create_dataset(gas+'/'+method+'/error/SYpartcolumnerrs',data=SYpartcolumnerrs)
		dset4NEW=f.create_dataset(gas+'/'+method+'/error/STcoverrs',data=STcoverrs)
		dset5NEW=f.create_dataset(gas+'/'+method+'/error/SYcoverrs',data=SYcoverrs)
		dset6NEW=f.create_dataset(gas+'/'+method+'/error/ALLcoverrs',data=allcoverrs)
		f.flush()
	
	f.close()
	if not(partcolumnoperators == []):
		return STpartcolumnerrs,SYpartcolumnerrs,allpartcolumnerrs,STcoverrs,SYcoverrs,allcoverrs
	else:
		return STcolerrs,SYcolerrs,allcolerrs,STcoverrs,SYcoverrs,allcoverrs



#STcolerrs,SYcolerrs,allcolerrs,STcoverrs,SYcoverrs,allcoverrs=erroranalysis(filename,method,gas, plotswitch=0,ADDHDF=True)

'''
filename='121124.5_153513SC.bin.hdf5'
method='CH2O_v16'
gas='CH2O'

erroranalysis(filename,method,gas, plotswitch=0,ADDHDF=True,printflag=False)
'''
#for name in allcolerrs.dtype.names:
#	if printflag: print name,':',allcolerrs[name]

