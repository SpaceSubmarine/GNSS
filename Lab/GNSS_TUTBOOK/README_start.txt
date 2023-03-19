How to Start up the laboratory session:
---------------------------------------

For instance, for TUTORIAL-1:
                  ==========

  In the "GNSS_TUTBOOK" directory:

  - Create the following directories:

    mkdir WORK
    mkdir WORK/TUT1
    mkdir WORK/TUT1/FIG

  - Go to the "WORK/TUT1" directory
    cd WORK/TUT1

  - Copy the program files:
    cp ../../PROG/TUT1/* .

  - Copy the Data files:
    cp ../../FILES/TUT1/* .
    cp -d /bin/gLAB_linux /bin/gLAB_GUI.py /bin/graph.py .

  - Uncompress the data files:
    gzip -df *.Z *.gz

  - Start "Applied Lecture" exercises following the
    instructions of slides "Tutorial_1_v2.0.0.pdf".

    Note: See also the notepad "ntpd_tut1.txt"
          (you can find this notepad file is in the directory "GNSS_TUTBOOK").




NOTEPAD file:
=============
 The "ntpd_tut1.scr" file contains the command line sentences 
 of the exercises for the Applied Lecture.
 These notepad are provided to help the user in the sentence
 writing, and the user can "COPY" and "PASTE" such sentences in the 
 working terminal.

 For those exercises where gLAB is executed using the Graphic 
 User Interface (GUI), the equivalent "command line gLAB sentence"
 and the associated "gLAB configuration file" (i.e. gLAB.cfg), 
 if needed, are also provided in the "FILES/TUT1" directory. 


 NOTE:
   The file "ntpd_tut1.scr" is a UNIX script that executes 
   automatically the entire session when they are executed from the
   "GNSS_TUTBOOK" directory. That is:

   In the "GNSS_TUTBOOK" directory, execute, for instance:
   
   ./ntpd_tut1.scr

   The following directories will be created:
   WORK/
   WORK/FIG

   Where the directory "FIG" contains the plots.





TUTORIAL0:  Basic Training on UNIX (LINUX) commands and environment
=========   Introduction to GNSS standard Data Files Format.

The laboratory sessions of these Tutorials are developed on a UNIX 
(Linux) Operating System (OS). Therefore, an introductory Tutorial 
on the UNIX environment and software tools has been considered useful 
to give the reader a basis for the development of the exercises. 
Among the tools, additional background on the GNSS standard file 
formats is also provided as complementary information. Readers with 
some knowledge of UNIX (Linux) OS and GNSS data file formats can skip 
this Tutorial_0_v0.0.pdf.

