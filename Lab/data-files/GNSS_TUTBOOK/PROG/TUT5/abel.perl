#!/usr/bin/perl

# ---------------------------------------------------------------
# This program computes the electron density "Ne(p)" profile from
# from GPS L1-L2 carrier measurements "over a continuous arc of 
# data" (corresponding to a Radio Occultation).
#
# INPUT data: [p(n),L1-L2(n)]
# ---------------------------
#   with p in km and L1-L2 in meters (of L1-L2 delay), 
#   where the impact parameter must be sorted from larger 
#   to lower values.
#   Note: Only measurements with negative elevation must be 
#         given (i.e., occultation).
#
# OUTPUT data: [n,p(n),L1-L2(n),Ne(n)]
# ------------------------------------
#   where Ne is given in e-/m3.
#
# --------------------------------------------------------------

$,=" ";
$\="\n";

# ------------ Reading the data files and storing the data ---
#              in the vectors "stec", "r", and "p_"
# ------------------------------------------------------------
while (<STDIN>) {
chop;
@d=split(' ',$_,3);

# Creating the vector "p_" with the impact parameters.
push(@p_,$d[0]);

if ($nlayer==0) {

# To cancel the carrier ambiguity, the STEC
# will be referred to the first value of L1-L2.
# That is: STEC0:=L1(0)-L2(0)
$l0=$d[1];

# Creating the vector "r_" with the radius of 
# the spherical layers (like onions)
$r_[0]=$d[0]+1;
                } else {

$r_[$nlayer]=0.5*($p_[$nlayer]+$p_[$nlayer-1])
                       }

# Creating the vector "stec" with the STEC-STEC0 values:
push(@stec,$d[1]-$l0);

$nlayer++;
                }


# ------------- Computing the Abel inversion --------
#  That is estimating the electron density profile
# ---------------------------------------------------

### Once created the vectors "stec", "r", and "p_"
### the program computes the "Ne"  applying the 
### numerical algorithm defined in the slides.

for $i (0..$#p_) {

$stec0=0;
for $j (0..$i-1) {
$l1=sqrt($r_[$j]**2-$p_[$i]**2);
$l0=sqrt($r_[$j+1]**2-$p_[$i]**2);
$stec0+=$Ne[$j]*($l1-$l0)*2;
                 }
$Ne[$i]=($stec[$i]-$stec0)/(2*sqrt($r_[$i]**2-$p_[$i]**2));
print $i,$p_[$i],$stec[$i],$Ne[$i]*1e14/1.05;


                 }
