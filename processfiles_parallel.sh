#!/bin/bash

# Wolf: 0-9
# Alex:10-19
# Cesar:20-29
# Beatriz:30-39
# Ruben:40-49
# Alan:50-59
procmax=10
pp=0
ppp=0
pppsegment=10
#relativespecpath=/home/data/solar_absorbtion/binspec/altz_bin/SC/
#relativespecpath=/home/data/solar_absorbtion/binspec/altz_bin/SF
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v2/SF/2013/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SD/2015/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SE/2014/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SD/2014/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SE/2013/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SD/2013/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SE/2012/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SD/2012/
#relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/SC/2015/
relativespecpath=/home/D2_PROFFIT/ALTZ/spec/bin_v3/$1/$2/
echo $relativespecpath


station=ALTZ
resultpath=/home/STG_03/PROFFIT_results/$station/

method=$3
FILES=$relativespecpath/*
echo $FILES > listatodo.txt

fstop=/home/wolf/stop
rm $fstop
sleep 1
cd /home/D3_PROFFIT/PROFFIT/
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
do
ppp=$(($i % $procmax))
ppp=$(($pppsegment+$ppp))
echo 0  >  $ppp"_proffit/retrievalstate.txt"
echo  $ppp"_proffit/retrievalstate.txt"
done

sleep 2

for f in $FILES
do
  echo "Processing $f file..."
  fff=$(basename $f)
  fresult="$resultpath/$method/*/"${fff}.hdf5
  echo $fresult
  if [ -e ${fresult} ]
  then
  echo $fresult exist
  continue
  fi

  fzipresult="$resultpath/$method/*/"${fff}.zip
  echo $fzipresult
  if [ -e ${fzipresult} ]
  then
  echo $fresult exist
  continue
  fi
  if [ -e ${fstop} ]
  then
  echo $fstop exist
  continue
  fi


  while :
  do
  pp=$(($pp+1))
  echo $pp
  pp=$(( $pp  % $procmax ))
  ppp=$(($pppsegment+$pp))

  echo $ppp
  statefile=$ppp"_proffit/retrievalstate.txt"
  echo $statefile
  state=$(cat $statefile)
  echo state $state
echo  "$state" -eq 0  
  if [  "$state" -eq 0  ];then
  echo "retrieving" $ppp
  echo 1 > $statefile 	
  sleep 0.5
  #echo   "rem xdotool windowminimize $(xdotool getactivewindow)" > start_$pp.sh  
  echo   sh runmethode_pp.sh $method $relativespecpath $fff $resultpath $ppp $station > start_$ppp.sh
  echo   "echo 0 > "$statefile >> start_$ppp.sh
 #### echo   "xdotool windowminimize $(xdotool getactivewindow)" >> start_$pp.sh
  %%%xterm -e "sh start_"$ppp".sh" &
  sh start_$ppp.sh &
#sh runmethode_pp.sh $method $relativespecpath $fff $pp $resultpath
  state = ''
  break
  fi
  echo waiting
  sleep 1.0
  done

done
