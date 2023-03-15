#!/usr/bin/perl

# ---------------------------------------------------------------------------------------------
# This program uses the “RO.obs” as input data and computes the following combinations 
# of RO measurements:
#============================================================================================
#                            |<------ LEO ------>|<------- GPS ------->|
#YY DoY HH.HH CODE PRN  elev  r_LEO AR_LEO DEC_LEO  r_GPS AR_GPS DEC_GPS   L1  L2   L1-L2 arc
#                       (deg)  (km) (Deg)   (Deg)    (km)  (Deg)  (Deg)   (cycles)   (m)
# 1  2    3    4    5    6       7    8       9       10     11     12     13  14     15  16
#============================================================================================
#
# Where 
#    Rho: is the Euclidean Distance between GPS and LEO  
#      d: means differences in time 
#     DD: means Double Differences between GPS and LEOs satellites.
#
#   Note: the GPS PRN13 and LEO “l252” as used a reference satellites.
#           (the rays between these satellites are not in occultation).
#
#   The results are computed for the RO between GPS PRN02
#   and LEO  “l241” (the same occultation as in previous cases)
# ---------------------------------------------------------------------------------------------


$,=" ";
$\="\n";
$[=1;

$c=299792458;
$f0=10.23e6;$f1=154*$f0;$f2=120*$f0;
$g2r=atan2(1,1)/45.;


### Initial and Final Times
$t0=16200;
$t1=16500;

### Reference Satellites (GPS0:=sat0, LEO0:=leo0) to compute the Double Differences.
$sat=2;
$leo="l241";
$sat0=13;
$leo0="l251";



open (F1,"cat RO.obs|");

while (<F1>) {
chop;
@d=split(' ',$_,30);
$t=int($d[3]*3600+0.5);
if ($t>$t0 && $t<$t1 && ($d[5]==$sat || $d[5]==$sat0) && ($d[4] eq $leo || $d[4] eq $leo0) ) {
$r_l[1]=$d[7]*cos($d[9]*$g2r)*cos($d[8]*$g2r);
$r_l[2]=$d[7]*cos($d[9]*$g2r)*sin($d[8]*$g2r);
$r_l[3]=$d[7]*sin($d[9]*$g2r);
$r_g[1]=$d[10]*cos($d[12]*$g2r)*cos($d[11]*$g2r);
$r_g[2]=$d[10]*cos($d[12]*$g2r)*sin($d[11]*$g2r);
$r_g[3]=$d[10]*sin($d[12]*$g2r);

$rho=0;
for $i (1..3) {$rho=$rho+($r_l[$i]-$r_g[$i])**2}

### p_ is the Impact Parameter in function pf time 
$p_{$t." ".$d[4]." ".$d[5]}=$d[7]*cos($d[6]*$g2r);
$l1{$t." ".$d[4]." ".$d[5]}=$d[13]*$c/$f1;
$l2{$t." ".$d[4]." ".$d[5]}=$d[14]*$c/$f2;
$Rho{$t." ".$d[4]." ".$d[5]}=sqrt($rho)*1000.;
##  end if
              }
## end while
             }


### Time Differences computation:
foreach $key  (sort{$a<=>$b} (keys %l1) ) {
@d=split(' ',$key,9);
$k_=eval($d[1]-1)." ".$d[2]." ".$d[3];
if ($l1{$k_}) {
$dl1{$key}=$l1{$key}-$l1{$k_};
$dl2{$key}=$l2{$key}-$l2{$k_};
$dlc{$key}=($dl1{$key}*$f1**2-$dl2{$key}*$f2**2)/($f1**2-$f2**2);
$dRho{$key}=$Rho{$key}-$Rho{$k_};
              }
                                          }
### Double Differences (of time differences) computation:
foreach $key ( sort{$a<=>$b} (keys %dl1) ) {
@d=split(' ',$key,9);
if ($d[2] eq $leo && $d[3]== $sat) {
$k2=$d[1]." ".$leo." ".$sat0;
$k3=$d[1]." ".$leo0." ".$sat;
$k4=$d[1]." ".$leo0." ".$sat0;
if ($dl1{$k2} && $dl1{$k3} && $dl1{$k4}) {
$DDdl1=$dl1{$key}-$dl1{$k2}-$dl1{$k3}+$dl1{$k4};
$DDdl2=$dl2{$key}-$dl2{$k2}-$dl2{$k3}+$dl2{$k4};
$DDdlc=$dlc{$key}-$dlc{$k2}-$dlc{$k3}+$dlc{$k4};
$DDdRho=$dRho{$key}-$dRho{$k2}-$dRho{$k3}+$dRho{$k4};

### Printing Results:
printf "%s %8.2f %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e \n",$key,$p_{$key},$dRho{$key},$dl1{$key},$dl2{$key},$dlc{$key},$dl1{$key}-$dlc{$key},$dl2{$key}-$dlc{$key},$DDdRho,$DDdl1,$DDdl2,$DDdlc
                                         }
                                        }

                                          }
