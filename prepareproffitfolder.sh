# argument 1: procesor , 2: profiltexecutable
mkdir $1_proffit 
mkdir $1_proffit/inp_fwd
mkdir $1_proffit/inp_inv 
mkdir $1_proffit/OUT_FWD
mkdir $1_proffit/OUT_INV
cp proffit/inp_fwd/*    $1_proffit/inp_fwd/
cp $2 $1_proffit/
