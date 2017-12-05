#!/usr/bin/pytho
import pickle
import sys
sys.path.append('/home/python_toolbox/')
import sidereal
#sys.path.append('/home/data/solar_absorbtion/analysis/')
import pickle
import retrieval as retrieval # wolf
import readproffwd
import os
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import fileinput
import adddata

#folder=sys.argv[1]
#folder='/home/data/solar_absorbtion/altz_retrievals/results/N2O_NDACC/130531/130531SD.0_141700SD.bin'
#folder='/home/data/solar_absorbtion/altz_retrievals/results/CH4_NDACC/121205/121205SC.0018_181634SC.bin'
#print folder
folder=sys.argv[1]
print folder
method=sys.argv[2]
print method
#method='N2O_NDACC'
#method='CH4_NDACC'
#gasname='N2O'
gasname=method.split('_')[0]

print gasname

#gasname=sys.argv[2]nfitparam
###################################################################################################################################

def retrievaltypedef(ngas,nlayer,nmw,nfitparam,nfitparamraw,npt,gases):
    ngas=int(ngas)
    nlayer=int(nlayer)
    nmw=int(nmw)
    nfitparam=int(nfitparam)
    npt=int(npt)
    sgasesprf=[]
    sgascols=[]
    sgasAK=[]
    rubroold=''
    isocounter=1
    for rubro in gases:
        if rubro==rubroold:
            rubro=rubro+'iso%i' % (isocounter)
            isocounter=isocounter+1
        else:
            rubroold=rubro
            isocounter=1
        sgasesprf.append((rubro,np.float32,nlayer))
        sgasAK.append((rubro,np.float32,(nlayer,nlayer)))        
        sgascols.append((rubro,np.float32,1))
    print sgasesprf
    gasprftype=np.dtype(sgasesprf)
    gascoltype=np.dtype(sgascols)
    gasAKtype=np.dtype(sgasAK)
    retrievaltype=np.dtype([('tepoch',np.int),\
			('date',str,10),\
			('time',str,8),\
			('jd_utm6',np.float64,1),\
			('mjd2k_utm6',np.float64,1),\
			('jd_ut',np.float64,1),\
			('mjd2k_ut',np.float64,1),\
			('filename',str,30),\
			('DUR',np.float32,1),\
                       ('sza',np.float64,1),\
                       ('az',np.float64,1),\
                       ('apd',np.float64,1),\
                       ('opd',np.float64,1),\
                       ('semifov',np.float64,1),\
                       ('meanrms',np.float32,1),\
                       ('meansignal',np.float32,1),\
                       ('relative_rms',np.float32,1),\
                       ('meanwshift',np.float32,1),\
                       ('targetgas',np.dtype('a10'),1),\
                       ('totcol',np.float32,1),\
                       ('gas',np.dtype('a10'),ngas),\
                       ('cols',gascoltype),\
                       ('rms',np.float32,nmw),\
                       ('signal',np.float32,nmw),\
                       ('wshift',np.float32,nmw),\
                       ('dof',np.float32,1),\
                       ('dofs',gascoltype),\
                       ('dofmas',np.float32,1),\
                       ('dofall',np.float32,1),\
                       ('retvmr',gasprftype),\
                       ('aprvmr',gasprftype),\
                       ('prfvmr',np.float32,nlayer),\
                       ('pcol',np.float32,nlayer-1),\
                       ('aprpcol',np.float32,nlayer-1),\
                       ('apriori',np.float32,nlayer),\
                       ('AKtot',np.float32,nlayer),\
                       ('numberdensity',np.float32,nlayer),\
                       ('AKvmr',gasAKtype),\
                       ('airmass',np.float32,nlayer),\
                       ('P',np.float32,nlayer),\
                       ('T',np.float32,nlayer),\
                       ('fitparameter',np.float32,nfitparam),\
                       ('fitparameterraw',np.float32,nfitparamraw),\
                       ('KTdyKxraw',np.float32,nfitparamraw),\
                       ('KTdyKx',np.float32,nfitparam),\
                       ('npt',np.int,1)])
    return retrievaltype
    
    
###################################################################################################################################

def diagnosticstypedef(ngas,nlayer,nmw,nfitparam,nfitparamraw,npt):
    ngas=int(ngas)
    nlayer=int(nlayer)
    nmw=int(nmw)
    nfitparam=int(nfitparam)
    npt=int(npt)
    
    diagnosticstype=np.dtype([('fitparameter',np.float32,nfitparam),\
                       ('fitparameterraw',np.float32,nfitparamraw),\
                       ('fulltoraw',np.float32,(nfitparamraw,nfitparam)),\
                       ('gainfull',np.float32,(npt,nfitparam)),\
                       ('gainraw',np.float32,(npt,nfitparamraw)),\
                       ('Kfull',np.float32,(npt,nfitparam)),\
                       ('Afull',np.float32,(nfitparam,nfitparam)),\
                       ('Kraw',np.float32,(npt,nfitparamraw)),\
                       ('Araw',np.float32,(nfitparamraw,nfitparamraw)),\
                       ('R',np.float32,(nfitparamraw,nfitparamraw)),\
                       ('KTdyKxraw',np.float32,nfitparamraw),\
                       ('KTdyKx',np.float32,nfitparam),\
                       ('KTK',np.float32,(nfitparam,nfitparam)),\
                       ('KTKraw',np.float32,(nfitparamraw,nfitparamraw)),\
                       ('rawtofull',np.float32,(nfitparam,nfitparamraw))])
    return diagnosticstype
###################################################################################################################################

def spectypedef(npt,nmw):  
    spectype=np.dtype([('nmw',np.float64,1),\
                       ('ibegin',np.int,nmw),\
                       ('w',np.float32,npt),\
                       ('obs',np.float32,npt),\
                       ('sim',np.float32,npt),\
                       ('dif',np.float32,npt),\
                       ('dw',np.float32,1),\
                       ('wlow',np.float32,nmw),\
                       ('wup',np.float32,nmw),\
                       ('npt',np.int,1)])
    return spectype
###################################################################################################################################

def julian_day(YY,MM,DD,HR,Min,Sec):
    return 367*YY-(7*(YY+((MM+9)/12))/4)+(275*MM/9)+DD+1721013.5-0.5*np.sign((100*YY)+MM-190002.5)+0.5+HR/24.0+Min/(60.0*24.0)+Sec/(3600.0*24.0)



###################################################################################################################################

ret=retrieval.retrieval(folder)
ngas=ret.ngas
gases=ret.gas
nlayer=ret.nlevel
nfitparam=len(ret.fitparameter)
nfitparamraw=len(ret.fit_list_raw)

nmw=ret.par.nmw
ret.spec()
npt=int(0)
for mw in ret.mw:
    npt=npt+int(mw.npt)
print 'npt',npt
###################################################################################################################################
print 'gases'
print ret.fit_list_raw
print ret.fit_list_full

print ngas,nlayer,nmw,nfitparam,npt,gases
print nfitparamraw
retrievaltype=retrievaltypedef(ngas,nlayer,nmw,nfitparam,nfitparamraw,npt,gases)
diagnosticstype=diagnosticstypedef(ngas,nlayer,nmw,nfitparam,nfitparamraw,npt)
spectype=spectypedef(npt,nmw)
newdataarr=np.zeros(1,dtype=retrievaltype)
newdata=newdataarr[0]
diagnewdataarr=np.zeros(1,dtype=diagnosticstype)
diagnewdata=diagnewdataarr[0]
spcnewdataarr=np.zeros(1,dtype=spectype)
spcnewdata=spcnewdataarr[0]
resulttype=np.dtype([('prfvmr',np.float32,1),\
    ('AKtot',np.float32,1),\
    ('apriori',np.float32,1),\
    ('numberdensity',np.float32,1),\
    ('z',np.float32,1)])

resultarr=np.zeros([1],dtype=resulttype)
result=resultarr[0]
resultarr=np.zeros([nlayer],dtype=resulttype)
###################################################################################################################################

print folder
stime=os.path.basename(folder).split('_')[1][0:6]
print stime
ret=retrieval.retrieval(folder)
prffwd=readproffwd.prffwd(folder)
print ret.gas
try:
    ngas=ret.gas.index(gasname)
except:
    print   'hola',ret.gas
    ngas=ret.gas.index(gasname.upper())
columna=ret.pcol[ngas].totcol*1E-4
rms=ret.par.rms
meansignal=ret.par.meansignal
try:
    wshift=ret.par.wshifts
except:
    print 'wshift no exist' ##wshift=0.0
fitparameter=ret.fitparameter
if str(columna) != 'nan':
           
    print columna
    jt=julian_day(ret.datetime.year,ret.datetime.month,ret.datetime.day,ret.datetime.hour,ret.datetime.minute,ret.datetime.second)
#######jt UT ?????????
    # error before 4 octubre 2015 and first NDACC upload jt2000=2451545.0000
    jt2000=2451544.5000
    newdata=newdataarr[0]
    print jt
    newdata['jd_utm6']=jt
    newdata['mjd2k_utm6']=jt-jt2000

    newdata['jd_ut']=newdata['jd_utm6']+6.0/24.0
    #newdata['mjd2k_ut']=newdata['jd_utm6']-jt2000 # error before 4 octubre 2015
    newdata['mjd2k_ut']=newdata['jd_utm6']+6.0/24.0-jt2000 

    newdata['tepoch']=np.int(ret.datetime.strftime('%s'))
    newdata['date']=ret.datetime.strftime('%Y-%m-%d')
    newdata['time']=ret.datetime.strftime('%H:%M:%S')
    newdata['filename']=folder.split('/')[-1]

    newdata['sza']=prffwd.sza*180/np.pi
    newdata['az']=prffwd.az*180/np.pi
    newdata['apd']=prffwd.apd
    newdata['opd']=prffwd.opd
    newdata['semifov']=prffwd.semifov
    newdata['totcol']=columna
    newdata['prfvmr'][:]=ret.prf[ngas].vmr[:]
    newdata['apriori'][:]=ret.prf[ngas].aprvmr[:]
    retpcol= np.dot( ret.vmrtoaircol,ret.prf[ngas].vmr)*1.0E-6 # ppm
    newdata['pcol'][:]=retpcol[:]/10000.0 # m^-> cm ^2
    aprpcol= np.dot( ret.vmrtoaircol,ret.prf[ngas].aprvmr)*1.0E-6 # ppm
    newdata['aprpcol'][:]=aprpcol[:]/10000.0 # m^-> cm ^2
 
    newdata['numberdensity'][:]=ret.prf[ngas].numberdensity[:]
    newdata['AKtot'][:]=ret.Atot[ngas][:]
    try:
	newdata['gas'][:]=ret.gas[:]
    except:
	newdata['gas']=ret.gas[0]
    newdata['targetgas']=gasname
    try:
	print type(ret.par.rms)
    	print type(newdata['rms'])
    	newdata['rms'][:]=ret.par.rms[:]
    except:
	newdata['rms']=ret.par.rms[0]
    newdata['dof']=ret.dof[ngas]

    gaseshdf=newdata['dofs'].dtype.names
    for kgas,gasnameactual in enumerate(gaseshdf):
	newdata['dofs'][gasnameactual]=ret.dof[kgas]
    newdata['dofmas']=ret.dof[kgas+1]
    newdata['dofall']=np.sum(ret.dof)

    for kk in range (nlayer):
        result['prfvmr']=ret.prf[ngas].vmr[kk]
        result['AKtot']=ret.Atot[ngas][kk]
        result['apriori']=ret.prf[ngas].aprvmr[kk]
        result['numberdensity']=ret.prf[ngas].numberdensity[kk]
        result['z']=ret.z[kk]
        resultarr[kk]=result
    
    newdata['meanrms']=np.average(ret.par.rms[:])
    newdata['meansignal']=np.average(ret.par.meansignal[:])
    newdata['relative_rms']=newdata['meanrms']/newdata['meansignal']
    try:
        newdata['meanwshift']=np.average(ret.par.wshifts[:])
    except:
        newdata['meanwshift']=0.0
    try:
	newdata['signal'][:]=ret.par.meansignal[:]
    except:
	newdata['signal']=ret.par.meansignal[0]
    try:
	newdata['wshift'][:]=ret.par.wshifts[:]
    except:
        try:
            newdata['wshift']=ret.par.wshifts[0]
        except:
            newdata['wshift']=0.0
    gaseshdf=newdata['retvmr'].dtype.names

    for igas in range(len(gases)):
        newdata['cols'][igas]=ret.pcol[igas].totcol*1E-4
        print igas, gases[igas],gaseshdf[igas]

        newdata['retvmr'][gaseshdf[igas]][:]=ret.prf[igas].vmr[:]
        newdata['aprvmr'][gaseshdf[igas]][:]=ret.prf[igas].aprvmr[:]
        newdata['AKvmr'][gaseshdf[igas]][:,:]=ret.Avmr_blocks[igas][:,:]
    newdata['airmass'][:]=ret.airmass[:]
    print 'fitparameter:',ret.fitparameter
    newdata['fitparameter'][:]=ret.fitparameter[:]
    newdata['P'][:]=ret.PT_ret.p
    newdata['T'][:]=ret.PT_ret.T
    ret.diagnostic()
    print ret.diagnostics.BTB[5,:]
    obs=np.array([],dtype=np.float32)
    sim=np.array([],dtype=np.float32)
    ret.spec()
    npt=int(0)
    for mw in ret.mw:
        npt=npt+int(mw.npt)
    print 'npt',npt
    newdata['npt']=npt
    for mw in ret.mw:
        obs=np.append(obs,np.array((mw.obs),dtype=np.float32))
        sim=np.append(sim,np.array((mw.sim),dtype=np.float32))
    
    print    len(spcnewdata['obs'])         
    spcnewdata['obs'][0:npt]=obs[0:npt]    
    spcnewdata['sim'][0:npt]=sim[0:npt] 
    spcnewdata['dif'][0:npt]=obs[0:npt]-sim[0:npt]  
    ret.diagnostics.full()     
    K=ret.diagnostics.JAK_full
    Kraw=ret.diagnostics.JAK_raw
    G=ret.diagnostics.gain_full
    Graw=ret.diagnostics.gain_raw
    ak=ret.averagingkernel_init()
    Afull=ret.averagingkernel.kernelfull
    Araw=ret.averagingkernel.kernelraw
    diagnewdata['Afull'][:,:]=Afull[:,:]
    diagnewdata['Araw'][:,:]=Araw[:,:]
    diagnewdata['Kfull'][:,:]=K[:,:]
    diagnewdata['gainfull'][:,:]=G[:,:]
    diagnewdata['Kraw'][:,:]=Kraw[:,:]
    diagnewdata['gainraw'][:,:]=Graw[:,:]
    n=len(K[0,:])
    for i in range(n):
        ii=i
        ki=K[:,i]
        for j in range(ii+1):
            kj=  K[:,j]                  
            kkk=np.dot(ki,kj)
            diagnewdata['KTK'][i,j]=kkk
            diagnewdata['KTK'][j,i]=kkk            
    Kx=np.dot(K,ret.fitparameter)
    dypKx=np.array(obs-sim+Kx,dtype=np.float32)
    KTdypKx=np.dot(K.T,dypKx)  
    newdata['KTdyKx'][:]=KTdypKx[:]
    diagnewdata['KTdyKx'][:]=KTdypKx[:]
    KTdypKxraw=np.dot(Kraw.T,dypKx)  
    diagnewdata['KTdyKxraw'][:]=KTdypKxraw[:]

icount=0
helpvector=[]
helpvector2=[]
rawtofull=np.zeros((nfitparam,nfitparamraw),dtype=float)
fulltoraw=np.zeros((nfitparamraw,nfitparam),dtype=float)
fromvmr=np.zeros((nfitparam,nfitparam),dtype=float)
tovmr=np.zeros((nfitparam,nfitparam),dtype=float)
for i,x in enumerate(ret.fit_list_full[:]):
    if x.split()[0] == ret.fit_list_raw[i-icount].split()[0]:
        #print x
        helpvector2.append(i)
    else:
        icount=icount+1
        #print icount
    helpvector.append([i,i-icount])
print helpvector
print helpvector2
for i,x in enumerate(helpvector2[:-1]):
    
    if ( helpvector2[i+1]-x > 1):
        
        #print 'x',x
        ibegin=x
        iend=helpvector2[i+1]
        gasnamehelp=ret.fit_list_raw[i].split()[0]
        igas=ret.gas.index(gasnamehelp) 
        #print ibegin,igas,gasnamehelp
                     
        for kk in range(nlayer):
            rawtofull[x+kk,i]=ret.prf[igas].firstvmr[kk]
            fulltoraw[i,x+kk]=1.0/(ret.prf[igas].firstvmr[kk]*nlayer)
            fromvmr[x+kk,x+kk]=1.0/ret.prf[igas].firstvmr[kk]
            tovmr[x+kk,x+kk]=ret.prf[igas].firstvmr[kk]

    else:
        rawtofull[x,i]=1.0
        fulltoraw[i,x]=1.0
        tovmr[x,x]=1.0
        fromvmr[x,x]=1.0

lenraw=len(ret.fit_list_raw)
lenfull=len(ret.fit_list_full)

fulltoraw[lenraw-1,lenfull-1]=1.0
rawtofull[lenfull-1,lenraw-1]=1.0

print 'hasata qui'        
fitparameterraw=np.dot(fulltoraw,ret.fitparameter) 
newdata['fitparameterraw'][:]=fitparameterraw[:]
KTdypKxraw=np.dot(rawtofull.T,KTdypKx) 
newdata['KTdyKxraw'][:]=KTdypKxraw[:]
    
diagnewdata['rawtofull'][:,:]=rawtofull[:,:]
diagnewdata['fulltoraw'][:,:]=fulltoraw[:,:]
diagnewdata['fitparameterraw'][:]=fitparameterraw[:]
diagnewdata['fitparameter'][:]=ret.fitparameter[:]


f= h5py.File(folder+'.hdf5','w')
h5dataset=f.create_dataset(gasname+'/'+method+'/retrieval',data=newdata,dtype=retrievaltype)
h5dataset.attrs['levels']=ret.z[:]
h5dataset.attrs['target']=gasname
h5dataset.attrs['retrieval']=method
h5dataset.attrs['fitparameter']=ret.fit_list_full[:]
h5dataset.attrs['fitparameterraw']=ret.fit_list_raw[:]
h5dataset.attrs['retrieval']=method

resultdataset=f.create_dataset(gasname+'/'+method+'/result',data=resultarr,dtype=resulttype)

#diagh5dataset=f.create_dataset(gasname+'/'+method+'/diagnostics',data=diagnewdata,dtype=diagnosticstype)


############### fitparameters
helpfitparameterrawtype=[]
print ret.fit_list_raw
for fitvariable in ret.fit_list_raw:
	helpfitparameterrawtype.append((fitvariable,np.float32))
fitparameterrawtype=np.dtype(helpfitparameterrawtype)
raw_fitted_parameters=np.zeros((1),dtype=fitparameterrawtype)[0]
for ir,rubro in enumerate(fitparameterrawtype.names):
	raw_fitted_parameters[rubro]=fitparameterraw[ir]

fitted_parameters=f.create_dataset(gasname+'/'+method+'/fitparameterraw',data=raw_fitted_parameters,dtype=fitparameterrawtype)

# informacion no necessario
#scph5dataset=f.create_dataset(gasname+'/'+method+'/spec/fullspc',data=spcnewdata,dtype=spectype)

microwindowtype=np.dtype([('obs',np.float32,1),('sim',np.float32,1),('dif',np.float32,1),('w',np.float32,1)])
mwarr=np.zeros([1],dtype=microwindowtype)
mw=mwarr[0]    
for i,x in enumerate(ret.mw):
    nn=len(x.obs)
    mwarr=np.zeros([nn],dtype=microwindowtype)
    for ii in range(nn):
        mw['obs']=x.obs[ii]
        mw['sim']=x.sim[ii]
        mw['dif']=x.res[ii]
        mw['w']=x.w[ii]
        mwarr[ii]=mw
    mwh5dataset=f.create_dataset(gasname+'/'+method+'/spec/microwindow_'+str(i),data=mwarr,dtype=microwindowtype)


try:
	solarscale=np.genfromtxt(folder+'/SOLSKAL.DAT')
	solarscale=f.create_dataset(gasname+'/'+method+'/spec/SOLSKAL',data=solarscale)
except:
	print 'no solar parameter'
f.flush()
f.close()
print'wrote '+folder+'.hdf5'

import readerrores

gasarr=ret.gas
targetgas=ngas
nlevel=ret.nlevel
nmw=len(ret.mw)
print gasarr[ngas]
print folder,method,gasarr,targetgas,nlevel,nmw
partialcoloperators,err_ST,err_SY,err_TOT=readerrores.errortohdf(folder,method,gasarr,targetgas,nlevel,nmw)
# 


APCOL=ret.averagingkernel.Apcol[targetgas]
prfcol=ret.airmass*ret.prf[targetgas].vmr
aprioriprfcol=ret.airmass*ret.prf[targetgas].aprvmr
prfcolh2o=ret.airmassh2o*ret.prf[targetgas].vmr
aprioriprfcolh2o=ret.airmassh2o*ret.prf[targetgas].aprvmr

PAKS=[]
pcols=[]
aprpcols=[]

pcolsh2o=[]
aprpcolsh2o=[]
for i,g in enumerate(partialcoloperators.T):
	print i,g
	PAK=np.dot(g.T,APCOL)
	PAKS.append(PAK)
	pcols.append(np.dot(g.T,prfcol))
	aprpcols.append(np.dot(g.T,aprioriprfcol))
	pcolsh2o.append(np.dot(g.T,prfcolh2o))
	aprpcolsh2o.append(np.dot(g.T,aprioriprfcolh2o))
pcols=np.array(pcols,dtype=np.float32)
aprpcols=np.array(aprpcols,dtype=np.float32)
dpcols=pcols-aprpcols

#,('err_ST',float),('err_SY',float),('err_TOT',float)

pctype=np.dtype([('ret',float),('reth2o',float),('apr',float),('aprh2o',float),('dpcol',float),('err_ST',float),('err_SY',float),('err_TOT',float)])

partialcolumns=np.zeros((len(pcols)),dtype=pctype)
for i,val in enumerate(pcols):
	partialcolumns['ret'][i]=pcols[i]/10000000000.0
	partialcolumns['reth2o'][i]=pcolsh2o[i]/10000000000.0
	partialcolumns['apr'][i]=aprpcols[i]/10000000000.0
	partialcolumns['aprh2o'][i]=aprpcolsh2o[i]/10000000000.0
	partialcolumns['dpcol'][i]=dpcols[i]/10000000000.0
	try:
		partialcolumns['err_ST'][i]=err_ST[i]
		partialcolumns['err_SY'][i]=err_SY[i]
		partialcolumns['err_TOT'][i]=err_TOT[i]
	except:							### JORGE 22/06/2016, IN CASE THERE'S ONLY ONE PARTIAL COLUMN
		partialcolumns['err_ST'][i]=err_ST
		partialcolumns['err_SY'][i]=err_SY
		partialcolumns['err_TOT'][i]=err_TOT


f= h5py.File(folder+'.hdf5','r+')

pOP=f.create_dataset(gasname+'/'+method+'/error/partialOP',data=np.array(partialcoloperators))
pak=f.create_dataset(gasname+'/'+method+'/error/partialAK',data=np.array(PAKS).T)
pcols=f.create_dataset(gasname+'/'+method+'/error/partialcolumns',data=partialcolumns)

f.flush()
f.close()
##########NDACC
##sza astronomical
def sun_angle(lat,lon,alt,jt_UT):
	import ephem
	o = ephem.Observer()
	o.lat = lat*ephem.degree
	o.long =lon*ephem.degree
	alt=alt*1000.0
	o.elev=alt
	o.date = ephem.Date(jt_UT-2415020)
	print o.date
	sun = ephem.Sun(o)
       
	elev=float(sun.alt)
	azimuth=float(sun.az)
        astro_SZA = np.abs(90.0-elev*180/np.pi)
        dn=0.000292*np.exp(-alt/6000.0) # pressure actualisation
	print dn
        SZA_refacted = 180.0/np.pi*np.arcsin(np.sin(np.pi*astro_SZA/180.0)/(1.0+dn))
        SAZ = azimuth*180/np.pi
        print 'self.SZA,self.SZA_refacted,astro_SZA'
        print SAZ,SZA_refacted,astro_SZA
	return    SZA_refacted 
	#return astro_SZA

############################
def sun_angle_astro(lat,lon,alt,mjdt_UT):
	jt2000=2451544.5000
	jt_UT=mjdt_UT+jt2000
	import ephem
	o = ephem.Observer()
	o.lat = lat*ephem.degree
	o.long =lon*ephem.degree
	alt=alt*1000.0
	o.elev=alt
	o.date = ephem.Date(jt_UT-2415020)
	print o.date
	sun = ephem.Sun(o)
       
	elev=float(sun.alt)
	azimuth=float(sun.az)
        astro_SZA = np.abs(90.0-elev*180/np.pi)
	return astro_SZA
         
        
#########################
spcinf=prffwd.sec[3]
print '#####################################################'
binspec=(spcinf[2][0]).replace('\\','/')
binspec = binspec.replace("DD1","D2")		### JORGE 14/10/2015
print binspec

adddata.addbinsec(binspec,folder+'.hdf5')
adddata.addbincomment(binspec,folder+'.hdf5')


#############################################################3

f= h5py.File(folder+'.hdf5','r+')
retrievaldataset=f.get(gasname+'/'+method+'/'+'retrieval')
binsec1=f.get('binsec1')
print 'BINSEC1',binsec1
DATASET_DUMMY = f.get(gasname+'/'+method+'/'+'error/erraltitude')
print 'DUMMY',DATASET_DUMMY
try:
	VMRerr_ST=f.get(gasname+'/'+method+'/'+'error/COV_target_ST')
	print 'VMRerr_ST',VMRerr_ST
except:
	print 'ERROR EN VMR_ERROR_ST'
try:
	VMRerr_SY=f.get(gasname+'/'+method+'/'+'error/COV_target_SY')
	print 'VMRerr_SY',VMRerr_SY
except:
	print 'ERROR EN VMR_ERROR_SY'
print 'METHOD HR',method,'GAS HR',gasname
partialAK=f.get(gasname+'/'+method+'/'+'error/partialAK')
print 'PART_AK',partialAK
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ALTITUDE',data=ret.z/1000.0)
bndz=np.array([ret.z/1000.0,np.append(ret.z[1:]/1000.0,np.max(ret.z)/1000.0)]).T
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ALTITUDE.BOUNDARIES',data=bndz)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ALTITUDE.INSTRUMENT',data=np.min(ret.z)/1000.0)

##################################################################################################################################
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ANGLE.SOLAR_AZIMUTH',data=180.0+retrievaldataset['az'])
print 180.0+retrievaldataset['az']

###########################################################
lat=19.1187 # should in future come from the bin-file
lon=-98.6552 ## should in future come from the bin-file
alt=3.985 ## should in future come from the bin-file
infotext=''
try:
	
	bincomment=f.get('bincomment')
	lon=bincomment['LONGITUDE']
	infotext=infotext+' LONGITUDE'
	alt=bincomment['ALTITUDE']			### JORGE 22/06/2016
	infotext=infotext+' ALTITUDE'
	astronomical_elev=bincomment['ASTRONOMICAL_ELEV']
	infotext=infotext+' ASTRONOMICAL_ELEV'
	lat=bincomment['LATITUDE']
	infotext=infotext+'LATITUDE'
	

except:
	lon=binsec1['Longitude']		### JORGE 22/06/2016
	infotext=infotext+' Longitude'
	alt=binsec1['Altitude']
	infotext=infotext+' Altitude'
	lat=binsec1['Latitude']
	infotext=infotext+'Latitude'
	f.create_dataset('warning1',data='no site information bincomment apart from %s from binsec1' % infotext) ### JORGE 22/06/2016
	print 'exception site information'



###########################################################



dur=binsec1['DUR']

#wolf 3.7.2015
try: 
	retrievaldataset['DUR']=dur
except:
	print 'mimodo dur'

print 'dur',dur
#jd_ut=retrievaldataset['jd_ut']+dur/(2*24*3600) ##### correction for the duration will be corrected in future
jd_ut=retrievaldataset['jd_ut'] ##### changed on 4 octubre 2015 acording to new bin format
mjd2k_UT=retrievaldataset['mjd2k_ut']
sza=sun_angle(lat,lon,alt,jd_ut)
print sza,retrievaldataset['sza']

f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'DATETIME',data=mjd2k_UT) # future easier

try:
	astronomical_sza=np.float(90.0-astronomical_elev)
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ANGLE.SOLAR_ZENITH.ASTRONOMICAL',data=astronomical_sza) # future easier
	sun_angle_astro(lat,lon,alt,mjd2k_UT)
	astronomical_sza_ephm=sun_angle_astro(lat,lon,alt,mjd2k_UT)
	dsza=astronomical_sza-astronomical_sza_ephm
	f.create_dataset('szadifference_ephm',data=dsza)
except:
	f.create_dataset('warning2',data='no astronomical sza')
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'ANGLE.SOLAR_ZENITH.ASTRONOMICAL',data=retrievaldataset['sza']) # future easier


##################################################################################################################################

f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'INTEGRATION.TIME',data=binsec1['DUR'])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'LATITUDE.INSTRUMENT',data=lat)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'LONGITUDE.INSTRUMENT',data=lon)
pcols=np.append(retrievaldataset['pcol'],0.0)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN.PARTIAL_ABSORPTION.SOLAR',data=pcols)
aprpcols=np.append(retrievaldataset['aprpcol'],0.0)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN.PARTIAL_ABSORPTION.SOLAR_APRIORI',data=aprpcols)

f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN_ABSORPTION.SOLAR',data=partialcolumns['ret'][0])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN_ABSORPTION.SOLAR_APRIORI',data=partialcolumns['apr'][0])
AKtot=partialAK[:,0]
print len(AKtot)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN_ABSORPTION.SOLAR_AVK',data=AKtot)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN_ABSORPTION.SOLAR_UNCERTAINTY.RANDOM.STANDARD',data=partialcolumns['err_ST'][0])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.COLUMN_ABSORPTION.SOLAR_UNCERTAINTY.SYSTEMATIC.STANDARD',data=partialcolumns['err_SY'][0])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR',data=retrievaldataset['retvmr'][gasname])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_APRIORI',data=retrievaldataset['aprvmr'][gasname])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_AVK',data=retrievaldataset['AKvmr'][gasname])
##################################################################################################################################
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_UNCERTAINTY.RANDOM.COVARIANCE',data=VMRerr_ST)
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+gasname+'.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_UNCERTAINTY.SYSTEMATIC.COVARIANCE',data=VMRerr_SY)
##################################################################################################################################
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'PRESSURE_INDEPENDENT',data=retrievaldataset['P'])
f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'TEMPERATURE_INDEPENDENT',data=retrievaldataset['T'])

try:

	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'H2O.COLUMN_ABSORPTION.SOLAR',data=retrievaldataset['cols']['H2O'])
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'H2O.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR',data=retrievaldataset['retvmr']['H2O'])
except:
	print 'no h2o'

try:

	binsec4=f.get('binsec4')
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'SURFACE.PRESSURE_INDEPENDENT',data=binsec4['surface_pressure'])
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'SURFACE.TEMPERATURE_INDEPENDENT',data=binsec4['surface_temperature'])
except:

	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'SURFACE.PRESSURE_INDEPENDENT',data=retrievaldataset['P'][0])
	f.create_dataset(gasname+'/'+method+'/NDACC_GMES/'+'SURFACE.TEMPERATURE_INDEPENDENT',data=retrievaldataset['T'][0])
##################################################################################################################################

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################






f.create_dataset('version', data='hdffromretrieval3.py from octubre 4')

f.flush()
f.close()

