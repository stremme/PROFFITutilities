#!/bin/sh
###########
## imput for a run
method=$1
#'O3_test'
relativespecpath=$2
#'spec'
spec=$3

resultpath=$4
#resultpath="/home/...../ALTZ/results"
#'120530.0_150221SF.bin'
pp=$5
### clean up input and output
proffitfolder=$pp'_proffit/'
station=$6		###  elige estacion CCA o ALTZ en mayuscula

echo ---------------------------- >> retrievallog.txt
whovariable=$(who i am)
echo $whovariable >> retrievallog.txt
date >> retrievallog.txt
echo $method >> retrievallog.txt
echo $spec >> retrievallog.txt
echo processador/profitfolder: $pp >> retrievallog.txt


echo "1" > $proffitfolder/retrievalstate.txt


rm $proffitfolder/inp_fwd/PRFFWD.INP
rm $proffitfolder/inp_inv/*
rm $proffitfolder/OUT_INV/*
rm $proffitfolder/OUT_FWD/*

########## Date from name
OPUSspec=${spec%_*}
DATEspec=${OPUSspec%'S'*}
DATE=${DATEspec%'.'*}
echo $DATE

############### Forward input
fwdinput=../$station/skeleton/$method
echo prepare_PRFWD.py $fwdinput $relativespecpath $spec $proffitfolder 
python prepare_PRFWD.py $fwdinput $relativespecpath $spec $proffitfolder 
############### inversions strategy
cp ../$station/skeleton/$method/inp_inv/* $proffitfolder/inp_inv/
echo $fwdinput/updateparameter.sh $proffitfolder/inp_fwd/PRFFWD.INP  $DATE
sh $fwdinput/updateparameter.sh $proffitfolder/inp_fwd/PRFFWD.INP  $DATE
########### run proffit in subcarpeta

echo start proffit >> retrievallog.txt
echo ....................... >> retrievallog.txt
cd $proffitfolder
wine proffi96.exe < goahead.txt > log.txt
cd ..

#############
echo ....................... >> retrievallog.txt
echo $spec >> retrievallog.txt

################ save results

mkdir $resultpath/$method
mkdir $resultpath/$method/$DATE
mkdir $resultpath/$method/$DATE/$spec
mv $proffitfolder/OUT_INV/* $resultpath/$method/$DATE/$spec/
mv $proffitfolder/inp_fwd/PRFFWD.INP $resultpath/$method/$DATE/$spec/
mv $proffitfolder/inp_inv/* $resultpath/$method/$DATE/$spec/
mv $proffitfolder/log.txt $resultpath/$method/$DATE/$spec/
mv start_$pp.sh $resultpath/$method/$DATE/$spec/


aqui=$(pwd)
echo $aqui 
echo $aqui >> $aqui/retrievallog.txt
echo start hdffromretrieval >> $aqui/retrievallog.txt
#python hdffromretrieval.py $resultpath$method/$DATE/$spec  $method  > $resultpath$method/$DATE/$spec/loghdf.txt
#python hdffromretrieval2.py $resultpath$method/$DATE/$spec  $method  > $resultpath$method/$DATE/$spec/loghdf.txt
# chane 3 octubre 2015 wolf
python hdffromretrieval3.py $resultpath$method/$DATE/$spec  $method  > $resultpath$method/$DATE/$spec/loghdf.txt

echo hdffromretrieval is finished >> $aqui/retrievallog.txt
date >> retrievallog.txt
echo $spec >> $aqui/retrievallog.txt
echo $aqui >> $aqui/retrievallog.txt
sleep 10
echo hola$DATE
cd $resultpath/$method/$DATE/
echo zipping $spec  >> $aqui/retrievallog.txt
zip -r $spec.zip $spec
rm -r $spec
echo $aqui >> $aqui/retrievallog.txt
date >> $aqui/retrievallog.txt
cd $aqui
echo returnto $aqui  >> $aqui/retrievallog.txt
echo -end  $spec--- >> $aqui/retrievallog.txt
echo "0" > $proffitfolder/retrievalstate.txt
exit

