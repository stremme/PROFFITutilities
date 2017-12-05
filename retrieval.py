#!/Uin	sr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from datetime import datetime
def hitrangas(i):
    #gas=['xcfgas','H2O','CO2','O3','N2O','CO','CH4','O2','NO','SO2','NO2','NH3','HNO3','OH','HF','HCl','HBr','HI','ClO','OCS','H2CO','HOCl','N2','HCN','CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6','H2S','HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH','CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3','SO2','NO2','NH3','HNO3','OH','HF','HCl','HBr','HI','ClO','OCS','H2CO','HOCl','N2','HCN','CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6','H2S','HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH','CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3']
     gas=['xcfgas','H2O','CO2','O3','N2O','CO','CH4','O2','NO','SO2','NO2','NH3','HNO3','OH','HF','HCl','HBr','HI','ClO','OCS','CH2O','HOCl','N2','HCN','CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6','H2S','HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH','CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3','SO2','NO2','NH3','HNO3','OH','HF','HCl','HBr','HI','ClO','OCS','H2CO','HOCl','N2','HCN','CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6','H2S','HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH','CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3']
     if i==99:
		return 'SIF4'
     return gas[i]


class microwindow:
    def __init__(self, fspec):
        specmatrix=np.genfromtxt(fspec)
        self.w=specmatrix[:,0]
        self.obs=specmatrix[:,1]
        self.sim=specmatrix[:,2]
        self.res=specmatrix[:,3]
        self.rms=np.std( self.res)
        self.npt=len( self.w)
        self.wlow=np.min(self.w)
        self.wup=np.max(self.w)
#####################################################################
class profile:
        def __init__(self, profilemat,ihit,igas,gas):  
            self.z=profilemat[:,0]
            self.aprvmr=profilemat[:,1]
            self.firstvmr=profilemat[:,2]
            self.vmr=profilemat[:,3]
            self.errvmr=profilemat[:,4]
            self.numberdensity=profilemat[:,5]
            self.gas=gas
            self.ihit=ihit
            self.igas=igas
#####################################################################
def profiles(fileprofile):
    import fileinput
    listprofile=[]
    gas=''
    igas=0
    ihit=0
    prfs=[]
    for line in fileinput.input(fileprofile):
        parts=line.split()
        if len(parts) == 1:
            if igas > 0:
                profilemat=np.array(listprofile,dtype=float)
                prf=profile(profilemat,ihit,igas,gas)
                prfs.append(prf)
            parts=line.split('/')
            print parts[len(parts)-1][0:2]
            try:
                ihit=int(parts[len(parts)-1][0:2])
                print ihit
                gas=hitrangas(ihit)
            except:
		parts=line.split('\\')
		try: 
			ihit=int(parts[len(parts)-1][0:2])
                	print ihit
                	gas=hitrangas(ihit)
		except:
                	gas=parts[len(parts)-1].split('_')[0]
                	gas=gas.split('.xsc')[0].upper()
            igas=igas+1
            listprofile=[]
        else:
#            print parts
            listprofile.append(np.array(parts,dtype=float))
    profilemat=np.array(listprofile,dtype=float)
    prf=profile(profilemat,ihit,igas,gas)
    prfs.append(prf)    
    return prfs
#####################################################################################
class partcol:
        def __init__(self, profilemat,ihit,igas,gas):  
            self.zlow=profilemat[:,0]
            self.zup=profilemat[:,1]
            self.pcol=profilemat[:,2]
            self.accol=profilemat[:,3]
            nlayer=len(profilemat[:,3])
            self.totcol=profilemat[nlayer-1,3]           
            self.gas=gas
            self.ihit=ihit
            self.igas=igas     

#######################################################################################  
              
#######################################################################################                    
def partialcol(filepartcol):
    import fileinput
    listprofile=[]
    gas=''
    igas=0
    ihit=0
    partcols=[]
    for line in fileinput.input(filepartcol):
        parts=line.split()
        if len(parts) == 1:
            if igas > 0:
                profilemat=np.array(listprofile,dtype=float)
                prf=partcol(profilemat,ihit,igas,gas)
                partcols.append(prf)
            parts=line.split('\\')

                #print parts[len(parts)-1][0:2]
            try:
                ihit=int(parts[len(parts)-1][0:2])
                #print ihit
                gas=hitrangas(ihit)
            except:
                gas=parts[len(parts)-1].split('_')[0]
            igas=igas+1
            listprofile=[]
        else:
            listprofile.append(np.array(parts,dtype=float))
    profilemat=np.array(listprofile,dtype=float)
    prf=partcol(profilemat,ihit,igas,gas)
    partcols.append(prf)
    return partcols
#####################################################################



def readtodollar(f):
    line=f.readline() 
    line=line.strip()+' '
    while line[0] <> '$':
        line=f.readline()+' '
    return


def readtodollarandreturnline(f):
    line=f.readline() 
    line=line.strip()+' '
    while line[0] <> '$':
        line=f.readline()+' '
    return line

class parameter:
    def __init__(self, faux):
        f=open(faux)
        readtodollar(f)        
        self.iter=int(f.readline())
        self.itermax=int(f.readline())
        readtodollar(f)        
        self.nmw=int(f.readline())        
        readtodollar(f) 
        rms=[]
        meansignal=[]
        for ii in range(self.nmw):
            line=f.readline()
            line=line.split()
            rms.append(float(line[1]))
            meansignal.append(float(line[0]))
        self.rms=np.array(rms)  
        self.meansignal=np.array(meansignal)
                
        readtodollar(f) 
        flags=[]
        for ii in range(5):
            line=f.readline()
            flags.append(line.strip())
        self.flags=flags
        
	fitparamlist=[]			### JORGE WOLF 03/08/2015
        readtodollar(f)         
        if self.flags[0] =='T':
            print 'ils should be red but is not'
	    fitparamlist.append('ils_modulation')
	    line=f.readline()
	    self.ils_modulation = float(line)
	    print self.ils_modulation
            
        if self.flags[1] =='T':
            print 'ils2 should be red but is not'
	    fitparamlist.append('ils_phase')
	    line=f.readline()
	    self.ils_phase = float(line)
	    print self.ils_phase
           
        readtodollar(f)  
        knotslist=[]
        if self.flags[2] =='T':
            #print 'base line knots'
            for ii in range(self.nmw):
                line=f.readline()
                line=f.readline()
                #print line
                knots=np.array(line.split(),dtype=float)
                #print knots
                knotslist.append(knots)
                for kk,knot in enumerate(knots):
                    fitparamlist.append('baseline %i  %i ' % (ii, kk))
        self.knots=knotslist


        readtodollar(f)
        if self.flags[3] =='T':
            print 'chanelling'
            nchanf=int(f.readline())
            self.nchanf=nchanf
            fchan=[]
            Achan=[]		
            for ic in range( nchanf ):
                 fchan.append(float( f.readline()))
            self.fchan=np.array(fchan)
            for ii in range(self.nmw):
                 amplitudes=f.readline().split()
                 for k,a in enumerate(amplitudes):
                    Achan.append(a)
                    fitparamlist.append('Achan %i  %i ' % (ii, k))
            self.Achan=Achan
                    	
			   		
            #		
            #

        lastposwshift=f.tell()  
        readtodollar(f)
        lines=[]
        wshifts=[]
        for ii in range(self.nmw):
            line=f.readline()
            lines.append(line)
            
            try:
            	print lines
            	self.wshifts=np.array(lines,dtype=float) 
            	fitparamlist.append('wshift %i' % (ii))             
            except:
            	print ' no wshift regresa antes el ultimo dollar'           	
            	#self.wshifts=np.zeros(self.nmw)   #	Beatriz wolf 1.9.2015
            	f.seek(lastposwshift)
	readtodollar(f)
        lines=[]
        solar=[]
         
	for ii in range(self.nmw):
            line=f.readline()
            lines.append(line)
            try:
		skale=float(line)
		fitparamlist.append('solar %i' % (ii))
            except: 
		print 'no soloar'          	
        try:
            print lines
            self.solar=np.array(lines,dtype=float)
        except:
            print 'no solar'
        


	self.fitparamlist=fitparamlist
        f.close()


#####################################################################
class PT:
    
    def __init__(self, filept):
        ptmat=np.genfromtxt(filept)
        self.z=ptmat[:,0]
        self.p=ptmat[:,1]
        self.T=ptmat[:,2]   


#diagnostico begin
class averagingkernel:
        "   "
        def __init__(self, folder,prf,pt):

            self.ngas=len(prf)
            self.filename = folder
            self.kernelfull=np.genfromtxt(folder+'/KERN_FUL.DAT')
            self.kernelraw =np.genfromtxt(folder+'/KERN_RAW.DAT')
            
            self.T_flag=False
############## wolf to destinguish between profil retrieval and scaling
            try:
		f=open(folder+'/proffit9.inp')
            except:
		f=open(folder+'/proffit9.inp'.upper())
		
            readtodollar(f)#
            readtodollar(f)#
            readtodollar(f)#
            line=readtodollarandreturnline(f)#
            print line
            line=f.readline()
            print line
            retflags=[]
            for ii in range(self.ngas):
                retflag=int(line.split(',')[0])
                retflags.append(retflag)
                print f.readline()
		line=f.readline()
                print line
                	
            f.close()


####################
       
            f=open(folder+'/DOFS.DAT')
            dof=[]
            nlevels=[]
            for ii in range(self.ngas):
                line=f.readline()
                line=f.readline()
                dof.append(float(line))
                nlevels.append(len(prf[ii].vmr))
            line=f.readline()
            #print 'check for Tretrieval:',line
            line=f.readline()

            for ii in range(self.ngas):
                if dof[ii] == 1.0 and retflags[ii]==1:
                    dof[ii]=dof[ii]+0.0001         
                    print 'no scaling retrieval'
		if dof[ii] == 0.0 and retflags[ii]==0:
                    dof[ii]=1.0         
                    print 'scaling retrieval'

            dof.append(float(line))
            if   self.T_flag == True:
                line=f.readline()
                line=f.readline()
                dof.append(float(line))
                
            print 'ndof', len(dof),dof
            self.dof=dof
            totdof=sum(self.dof)
            self.doftot=totdof
            self.nlevel=max(nlevels)            
            nlevel=self.nlevel
            #print 'nlevel',nlevel
            if   self.T_flag == True:
                nlevels.append(self.nlevel)
                
            npara=len(self.kernelfull)-sum(nlevels)
            nlevels.append(npara)
            
            #print len(nlevels),'levels',nlevels
            self.nlevels=nlevels
            
            def block(istart,jstart,ilevel,jlevel,Amat):
                return Amat[istart:istart+ilevel,jstart:jstart+jlevel]
            
            Afull=self.kernelfull
            #print totdof,np.trace(Afull)
            akmat=[]
            istart=0
            for i in range(len(self.dof)):
                jstart=0
                ilevel=self.nlevels[i]
                akrow=[]
                for j in range(len(self.dof)):
                    #print j
                    jlevel=self.nlevels[j]
                    akblock=block(istart,jstart,ilevel,jlevel,Afull)
                    #print np.trace(akblock)
                    #print i,j,self.dof[i],self.dof[j]
                    akrow.append(akblock)
                    jstart=jstart+jlevel
                akmat.append(akrow)
                istart=istart+ilevel
            self.AK_full_blocks=akmat
            #print len(akmat)
           
           
            Araw=self.kernelraw           
            akmat=[]
            for i in range(len(self.dof)):
                ilevel=self.nlevels[i]
                if self.dof[i] == 1.0:
                    ilevel=1
                akrow=[]
                for j in range(len(self.dof)):
                    jlevel=self.nlevels[j]
                    if self.dof[j] == 1.0:
                        ilevel=j
                    akrow.append(block(i,j,ilevel,jlevel,Araw))
            self.AK_raw_blocks =akmat


            z=np.array(pt.z,dtype=float)
           
            
            dz=np.zeros(nlevel-1,dtype=float)
            skh=np.zeros(nlevel-1,dtype=float)
            for i in range(0,nlevel-1):
                dz[i]=(z[i+1]-z[i])

            p=np.array(pt.p,dtype=float)
            T=np.array(pt.T,dtype=float)
            airmass=np.zeros([nlevel],dtype=float)
            airmassup=np.zeros([nlevel],dtype=float)
            airmassdown=np.zeros([nlevel],dtype=float)
            airmassh2o=np.zeros([nlevel],dtype=float)
            airmassuph2o=np.zeros([nlevel],dtype=float)
            airmassdownh2o=np.zeros([nlevel],dtype=float)

            vmrtoaircol=np.zeros((nlevel-1,nlevel),dtype=float)
            vmrtoaircolh2o=np.zeros((nlevel-1,nlevel),dtype=float)

            H2Ovmrppm=np.zeros([nlevel],dtype=float)


            print prf[0].gas
            gasarr=[item.gas for item in prf]
            try:
            	indexh2O=gasarr.index('H2O')
		H2Ovmrppm[:]=prf[indexh2O].vmr[:]
            except:
		print 'no agua'		
            ph2o=np.zeros([nlevel],dtype=float)
            ph2o[:]=p[:]*(1.0-1.0e-6*H2Ovmrppm[:])
            kboltzmann=1.3806504e-23
            for i in range(nlevel-1):
################ airmass esta definido de airmass[i]=dcol/dvmr[i]
################# asuming linera interpolatcion de VMR, T y log(P)
################# como Frank h. hagamos 50 pasos y evaluamos una columna up y uno down
################
		nfine =50
		dzfine=dz[i]/nfine
		zfine=(np.arange(nfine,dtype=np.float64)+0.5)*dzfine
		skh[i]=dz[i]/np.log(p[i]/p[i+1])
		
		Pfine=p[i]*np.array(np.exp(-zfine/skh[i]),dtype=np.float64)
		Tfine=T[i]+zfine*(T[i+1]-T[i])/dz[i]
		weightingup=1.0-zfine/dz[i]
		weightingdown=zfine/dz[i]
                airmassfino=dzfine*Pfine/(Tfine)
		airmassup[i]=np.sum(weightingup*airmassfino)
		airmassdown[i+1]=np.sum(weightingdown*airmassfino)
		
            for i in range(nlevel-1):
################ airmass esta definido de airmass[i]=dcol/dvmr[i]
################# asuming linera interpolatcion de VMR, T y log(P)
################# como Frank h. hagamos 50 pasos y evaluamos una columna up y uno down
################
		nfine =50
		dzfine=dz[i]/nfine
		zfine=(np.arange(nfine,dtype=np.float64)+0.5)*dzfine
		skh[i]=dz[i]/np.log(ph2o[i]/ph2o[i+1])
		
		Pfine=p[i]*np.array(np.exp(-zfine/skh[i]),dtype=np.float64)
		Tfine=T[i]+zfine*(T[i+1]-T[i])/dz[i]
		weightingup=1.0-zfine/dz[i]
		weightingdown=zfine/dz[i]
                airmassfino=dzfine*Pfine/(Tfine)
		airmassuph2o[i]=np.sum(weightingup*airmassfino)
		airmassdownh2o[i+1]=np.sum(weightingdown*airmassfino)


            for i in range(nlevel):
		airmass[i]=(airmassup[i]+airmassdown[i])/kboltzmann
		airmassh2o[i]=(airmassuph2o[i]+airmassdownh2o[i])/kboltzmann

            for i in range(nlevel-1):
		vmrtoaircol[i,i]=airmassdown[i+1]/kboltzmann
		vmrtoaircol[i,i+1]=airmassup[i]/kboltzmann
            for i in range(nlevel-1):
		vmrtoaircolh2o[i,i]=airmassdownh2o[i+1]/kboltzmann
		vmrtoaircolh2o[i,i+1]=airmassuph2o[i]/kboltzmann
            #plt.plot(skh)  
            #plt.show()  
            U=np.zeros([nlevel,nlevel],dtype=float)
            Uinv=np.zeros([nlevel,nlevel],dtype=float)
            
            for i in range(nlevel):
                U[i,i]=airmass[i]
                Uinv[i,i]=1.0/airmass[i]
            
            self.U=U
            self.airmass=airmass

            self.airmassh2o=airmassh2o
            self.h2ovmrppm=H2Ovmrppm
            self.vmrtoaircol=vmrtoaircol
            self.vmrtoaircolh2o=vmrtoaircolh2o
            AKVvmr=[]
            Apcol=[]
            for i in range(self.ngas):
                Avmr=self.AK_full_blocks[i]
                Avmr=np.array(Avmr[i])
		AKVvmr.append(Avmr)
                AUinv=np.dot(Avmr,Uinv) 
                UAUinv=np.dot(U,AUinv) 
                Apcol.append(UAUinv) 
               
            #print len(Apcol)
            self.Apcol=Apcol
            self.AKVvmr=AKVvmr
            Atot=[]
            gt=np.zeros([nlevel],dtype=float)+1.0
            totoperator=np.dot(self.U,gt)
            self.totoperator=totoperator
            
            for i,A in enumerate(Apcol):
                #print np.trace(A)
                #print self.dof[i]
                AK_tot=np.dot(A.transpose(),gt)
                #plt.plot(AK_tot,z)
                #plt.xlim([-2,5])
                #plt.show()
                Atot.append(AK_tot) 
            self.Atot=Atot
            self.z=z
            #AK_tot=Atot[1]
            #print len(AK_tot)
            #plt.plot(AK_tot,z)
            #plt.xlim([-2,5])
            #plt.show()
            
#diagnostico end

class diagnostics:
        "   "
        def __init__(self, folder):
            self.folder=folder
            self.gain_raw =np.genfromtxt(folder+'/GAIN_RAW.DAT')
            self.BTB=np.genfromtxt(folder+'/BTB.DAT')
            self.JTJ=np.genfromtxt(folder+'/JTJ.DAT')
            self.JAK_raw =np.genfromtxt(folder+'/JAK.DAT')
        def full(self):            
            self.gain_full=np.genfromtxt(self.folder+'/GAIN_FUL.DAT')
            self.JAK_full=np.genfromtxt(self.folder+'/JAKOBI.DAT')
            








#####################################################################
class retrieval:
    "   "
    def __init__(self, folder):
        self.folder = folder
        
        

#spectrumINVSPECA
#        INVSPECB
#        INVSPECC
#        INVSPECD
#        INVSPECE
        
        self.par=parameter(folder+'/AUXPARMS.DAT')
#retrieval      
        self.prf=profiles(folder+'/PROFS.DAT')
        self.pcol=partialcol(folder+'/PARTCOL.DAT')
        #self.dof=doffromfile(folder+'/DOF.DAT')
        
        gas=[]
        igas=[]
        ihit=[]
        gas_level=[]
        for item in self.prf:
            gas.append(item.gas)        
            igas.append(item.igas)       
            ihit.append(item.ihit)
            gas_level.append(len(item.vmr))
                
        self.ngas=len(gas)
        self.gas=gas
        self.igas=igas
        self.ihit=ihit
        self.gas_level=gas_level 
	print 'RET GAS_L',self.gas_level
        self.nlevel=np.max(gas_level)
        pt_ret=PT(folder+'/PT_OUT.DAT')
        self.PT_ret=pt_ret
        ak=averagingkernel(self.folder,self.prf,self.PT_ret)
        self.Atot=ak.Atot
        Avmr_blocks=[]
        for i in range(self.ngas):
            Avmr_blocks.append(ak.AKVvmr[i])
	#wolf    antes        Avmr_blocks.append(ak.AK_full_blocks[i][i])

        self.Avmr_blocks=Avmr_blocks
        self.airmass=ak.airmass
        self.airmassh2o=ak.airmassh2o
        self.vmrtoaircol=ak.vmrtoaircol
        self.vmrtoaircolh2o=ak.vmrtoaircolh2o
        
        self.z=ak.z
        self.dof=ak.dof
###########################################################################3
	
        self.gas_level_raw=copy.deepcopy(self.gas_level)
        for i,dof in enumerate(self.dof):
            if dof ==1.0:
                self.gas_level_raw[i]=1
        fit_list_full=[]
        isotop=1
        gasnameold=''
        print ' alex wolf'
        for i,gasname in enumerate(self.gas):
            print i,gasname,gasnameold,self.gas		
            isotop=isotop+1
            if gasnameold != gasname:
                isotop=1
            print isotop
            for j in range(self.nlevel):
                fit_list_full.append("%s %i %i" % (gasname,isotop,j+1))
            gasnameold = gasname
        for paramname in  self.par.fitparamlist:
            fit_list_full.append(paramname)        
        print 'RET',fit_list_full
        self.fit_list_full=fit_list_full
        fit_list_raw=[]
        isotop=1
        gasnameold=''
        for i,gasname in enumerate(self.gas):
            isotop=isotop+1
            if gasnameold != gasname:
                isotop=1
            for j in range(self.gas_level_raw[i]):
                fit_list_raw.append(gasname+' '+str(isotop).strip()+' '+str(j+1).strip())
            gasnameold = gasname
        print ' alex wolf',fit_list_raw
        for paramname in  self.par.fitparamlist:
            fit_list_raw.append(paramname)
        self.fit_list_raw=fit_list_raw
        print ' alex wolf',self.fit_list_raw

  #################fit_list_full################################################################        
              
        mat_fit_param=np.genfromtxt(folder+'/ACTPARMS.DAT',dtype=float)
        nline=len(mat_fit_param[:,0])
        print len(mat_fit_param[0,:])
        self.fitparameter=mat_fit_param[nline-1,:]
        
        ################################
        try:
            fnamedate=os.path.basename(folder)[0:6]
            fnametime=os.path.basename(folder).split('_')[1][0:6]
            print fnamedate
            print fnametime
            self.datetime=datetime.strptime(fnamedate+fnametime,'%y%m%d%H%M%S')
        except:
            print 'name not dateformat compartible'
            self.datetime=datetime.strptime('140506223000','%y%m%d%H%M%S')
        
    def spec(self):        
        
        self.mw=[]
        caracterflag=['A','B','C','D','E','F','G','H','I','J','K','L','M','O','P','Q','R','S','T','U','V','X','Y','Z']
        for cflag in caracterflag:
            fspec= self.folder+'/INVSPEC'+cflag+'.DAT'
            if os.path.isfile(fspec):
                spec=microwindow(fspec)
                self.mw.append(spec)
                
        self.nmw=len(self.mw)

        
    def averagingkernel_init(self):       
        self.averagingkernel=averagingkernel(self.folder,self.prf,self.PT_ret)
        self.dof=self.averagingkernel.dof
        self.gas_level_raw=copy.deepcopy(self.gas_level)
        for i,dof in enumerate(self.dof):
            if dof ==1.0:
                self.gas_level_raw[i]=1
        fit_list_full=[]
        isotop=1
        gasnameold=''
	print 'AK GAS_L',self.gas_level
        for i,gasname in enumerate(self.gas):
            for j in range(np.max(self.gas_level)):
                fit_list_full.append(gasname+' '+str(isotop).strip()+' '+str(j).strip())
            isotop=isotop+1
            if gasnameold != gasname:
                isotop=1
            gasnameold = gasname
        for paramname in  self.par.fitparamlist:
            fit_list_full.append(paramname)        
        print 'AK',fit_list_full
        #self.fit_list_full=fit_list_full        
        fit_list_raw=[]
        isotop=1
        gasnameold=''
        for i,gasname in enumerate(self.gas):
            isotop=isotop+1
            if gasnameold != gasname:
                isotop=1
            for j in range(self.gas_level_raw[i]):
                fit_list_raw.append(gasname+' '+str(isotop).strip()+' '+str(j+1).strip())
            gasnameold = gasname
        for paramname in  self.par.fitparamlist:
            fit_list_raw.append(paramname)
        self.fit_list_raw=fit_list_raw        
        

    
        #print fit_list_raw
    def diagnostic(self):
        self.diagnostics=diagnostics(self.folder)
        

   
#retrieval end
       
      

#otros inputs PT-file



#
