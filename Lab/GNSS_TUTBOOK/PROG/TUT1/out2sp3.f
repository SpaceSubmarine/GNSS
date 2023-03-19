c234567
        program out2sp3

        parameter (nts=99,neph=10000)
        implicit double precision (a-h,o-z)
        character satw*3,wpv*3,wclk*3,tsys*3,gout*18,mess*6,
     *            line*500
        dimension pvt(nts,neph,10)

c       ------------------------------------------------------
c       This program generates a SP3 file from the gLAB output
c       file "gLAB.out".
c
c         Execute: out2sp3 gLAB.out
c
c
c       Note: 
c         It uses the subroutine "sub_wrt" from "GLOeph2sp3.f"
c         program to write the SP3 file format.
c 
c       ...............
c       @ gAGE/UPC (group of Astronomy and Geomatics)
c       ------------------------------------------------------


c       -- Parameters definition --
               c=299792458.d0
c       ---------------------------

c       ------ Initialise ---------
        gout=" "
        mess=" "
        nephem=0
        xdt_out=0.d0
        satw="COM"
        idt_out=30
        satw="COM"
        wpv="NO" 
        wclk="NO"
        tsys="GPS"
        do i=1,nts
         do j=1,neph
          do k=1,10
            pvt(i,j,k)=0.d0
          enddo
         enddo
        enddo
c       ---------------------------


        call getarg(1,gout) 

        if ((gout.eq." ").or.(gout.eq."-help")) then
         print *, "Execute:  out2sp3 gLAB.out"
         goto 100
        endif

        open(10,file=gout)
        open(20,file="orb.sp3")
20      continue
        read(10,'(a)',end=30,err=20) line
        if(line(1:6).eq."OUTPUT") then
         read(line,*,err=20) mess,iyear0,idoy,secd,cov,x,y,z
         else
          goto 20
        endif
        isat = 09


c       ---- Identifying the change of ephemeris block time: -------
        call sub_doy2cal(iyear0,idoy,secd,iyear,imonth,iday,
     *               ihh,imm,xss,nw,sw)


        tw=dble(nw)*604800.d0+sw

        if (tw-tw0.gt.0.d-9) then
             nephem=nephem+1
             if (tw0.gt.1.d0) then
               xdt_out=xdt_out+tw-tw0
             endif
        endif
        if (nephem.gt.neph) then
           print *, "ERROR nephem.gt.nepn",nephem,neph
           goto 20
        endif

        tw0=tw
c       -----------------------------------------------------------

        if (isat.eq.0) then 
          print *, "PRN=0: Skip this satellite",isat
          goto 20
        endif
        pvt(isat,nephem,1)=dble(nw)
        pvt(isat,nephem,2)=sw
        pvt(isat,nephem,3)=x*1.d-3
        pvt(isat,nephem,4)=y*1.d-3
        pvt(isat,nephem,5)=z*1.d-3
        pvt(isat,nephem,9)=999999.999999d0
        pvt(isat,nephem,10)=999999.999999d0
        
        goto 20
30      continue

c       ... time interval estimation ........
        xnephem=dble(nephem)-1.d0
        idt_out=nint(xdt_out/xnephem) 
c       .....................................

       
c       ------- Writing SP3 file --------------------------------
        call sub_wrt(pvt,nephem,satw,wpv,tsys,idt_out)
c       ----------------------------------------------------------

100     continue
        close(10)
        close(20)

        end



c      ------------------------------------------------------
       subroutine sub_wrt(pvt,nephem,satw,wpv,tsys,idt_out)
       parameter (nts=99,neph=10000)
       implicit double precision (a-h,o-z)
       character satw*3,wpv*3,tsys*3
       dimension pvt(nts,neph,10),ifsat(nts),ifsat_all(nts),nsi(nts)


       c=299792458.d0

       k=0
       nwtot=0
       nstot=0
       nstot_all=0
       do i=1,nts
         nsi(i)=0
         ifsat(i)=0
         ifsat_all(i)=0
       enddo

c         ////////////////////////////////////////////////////////////////////
c         ===================== Preparing SP3 file header ===================

c          Checking the satellites appearing at "at least one time" [ifsat(i)]
c          and at all the epochs [ifsat_all(i)].

           do l=1,nephem
             do i=1,nts
               iflg=int(pvt(i,l,1))
               if (iflg.gt.0) then
                 ifsat(i)=1
                 ifsat_all(i)=ifsat_all(i)+1
               endif
             enddo
           enddo

           do i=1,nts
            nstot=nstot+ifsat(i)
            if (ifsat_all(i).eq.nephem) then
               ifsat_all(i)=1
               nstot_all=nstot_all+1
             else
               ifsat_all(i)=0
            endif
           enddo


c          -------- Satellites to write -------------------
           if ((satw.eq."COM").or.(satw.eq."com")) then
             do i=1,nts
               ifsat(i)=ifsat_all(i)
               nstot=nstot_all
             enddo
             if (nstot.eq.0) then
               print *,"********************************************"
               print *,"WARNING: No common satellites in all epochs."
               print *,"   Try with  [satw='ALL'] in  GLOeph2sp3.nml"
               print *,"********************************************"
               print *, "STOP my self"
               goto 100
             endif
           endif

           do i=1,nts
            if (ifsat(i).eq.1) then
               k=k+1
               nsi(k)=i
             endif
           enddo
           if (k.ne.nstot) print *, "ERROR",k,nstot
c          -------------------------------------------------

c          ---- Total Number of epochs to write in the file ----------------
           do l=1,nephem
             nflg=0
             do i=1,nts
              if (nflg.eq.0) then
               iflg=int(pvt(i,l,1))*ifsat(i)
               if (iflg.gt.0) then
                 nwtot=nwtot+1
                nflg=1
               endif
              endif
             enddo
           enddo
c          ------------------------------------------------------------------

c          -------------- Saving the initial time -------------------
           do l=1,nephem
c            .... Selection the first valid time in the file .....
             do i=1,nts
               iflg=int(pvt(i,l,1))*ifsat(i)
               if (iflg.gt.0) then
                nwe=nint(pvt(i,l,1))
                swe=pvt(i,l,2)
                call sub_nwsw2cal(nwe,swe,iyear_e0,imonth_e0,
     *             iday_e0,ihh_e0,imm_e0,xss_e0,doy_e0,secd_e0,
     *             nw_e0,sw_e0,xjd_e0,xmjd_e0)
                   mjd_e0=int(xmjd_e0)
                   frac0=secd_e0/86400.d0
                goto 10
              endif
             enddo
           enddo
10         continue
c          ---------------------------------------------------------------

c        --------  Writing the SP3 file header ---------------------------
         write(20,'(a3,i4,4(1x,i2),1x,f11.8,1x,i7,1x,a5,1x,
     *              a5,1x,a3,1x,a4)')
     *    "#cV",iyear_e0,imonth_e0,iday_e0,ihh_e0,imm_e0,
     *    xss_e0,nwtot,"d","IGS00","BCT","UPC"

          write(20,'(a2,1x,i4,1x,f15.8,1x,f14.8,1x,i5,1x,f15.13)')
     *    "##",nw_e0,sw_e0,dble(idt_out),mjd_e0,frac0

c         ..................................................
          if (nstot.lt.17) then
            write(20,'(a2,2x,i2,3x,17(a1,i2))')
     *      "+ ",nstot,("L",nsi(i),i=1,nstot),(" ",nsi(i),i=nstot+1,17)
            write(20,'(a2,7x,17(1x,i2))') "+ ",(0,i=1,17)
          endif
          if (nstot.eq.17) then
            write(20,'(a2,2x,i2,3x,17(a1,i2))')
     *      "+ ",nstot,("L",nsi(i),i=1,17)
            write(20,'(a2,7x,17(1x,i2))') "+ ",(0,i=1,17)
          endif
          if (nstot.gt.17) then
            write(20,'(a2,2x,i2,3x,17(a1,i2))')
     *      "+ ",nstot,("L",nsi(i),i=1,17)
            write(20,'(a2,7x,17(a1,i2))')
     *      "+ ",("L",nsi(i),i=18,nstot),(" ",nsi(i),i=nstot+1,34)
          endif
          write(20,'(a2,7x,17(1x,i2))') "+ ",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "+ ",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "+ ",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "++",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "++",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "++",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "++",(0,i=1,17)
          write(20,'(a2,7x,17(1x,i2))') "++",(0,i=1,17)
          write(20,'(a8,1x,a3,1x,a47)')
     * "%c L  cc",tsys,"ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
           write(20,'(a60)')
     *  "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
           write(20,'(a60)')
     *  "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000"
           write(20,'(a60)')
     *  "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000"
         write(20,'(a60)')
     *  "%i    0    0    0    0      0      0      0      0         0"
         write(20,'(a60)')
     *  "%i    0    0    0    0      0      0      0      0         0"
         write(20,'(a60)')
     *  "/* NOTE: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
         write(20,'(a60)')
     *  "/* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
         write(20,'(a60)')
     *  "/* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
         write(20,'(a60)')
     *  "/* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
c        ------------------------------------------------------------------


c        =====================  Writing the SP3 data blocks file =========
c         --------------- Writing the time blocks ---------------------
           do l=1,nephem
c            .....writing time (for each block of satellites).....
             nflg=0
             do i=1,nts
              if (nflg.eq.0) then
               iflg=int(pvt(i,l,1))*ifsat(i)
               if (iflg.gt.0) then
                nw=nint(pvt(i,l,1))
                sw=pvt(i,l,2)
                call sub_nwsw2cal(nw,sw,iyear,imonth,iday,ihh,imm,xss,
     *                          doy,secd,nweek,sweek,xjd,xmjd)
                write(20,'(a2,1x,i4,4(1x,i2),1x,f11.8)')
     *               "* ",iyear,imonth,iday,ihh,imm,xss
                nflg=1
               endif
              endif
             enddo
c            ........................................................
             do i=1,nts
              iflg=int(pvt(i,l,1))*ifsat(i)
              if (iflg.gt.0) then
               x=pvt(i,l,3)
               y=pvt(i,l,4)
               z=pvt(i,l,5)
               vx=pvt(i,l,6)
               vy=pvt(i,l,7)
               vz=pvt(i,l,8)
               clock=pvt(i,l,9)
               dclock=pvt(i,l,10)
               write(20,'(a2,i2,4(1x,f13.6))')
     *               "PL",i,x,y,z,clock
               if ((wpv.eq."YES").or.(wpv.eq."yes")) then
                 write(20,'(a2,i2,4(1x,f13.6))')
     *               "VL",i,vx,vy,vz,dclock
               endif
              endif
             enddo
           enddo
           write(20,'(a3)') "EOF"
c          -----------------------------------------------------------

100    continue
       return
       end
c      ------------------------------------------------------

         subroutine sub_nwsw2cal(nw,sw,iyear,imonth,iday,ihh,imm,xss,
     *        doy,secd,nweek,sweek,xjd,xmjd)
         implicit double precision (a-h,o-z)

c        --------------------------------------------------------
c        Compute the DOY, GPS_WEEK SEC_WEEK from YEAR MONTH DAY sec_of_day
c
c         nw, sw  -->|ws2cal|--> iyear,imonth, iday, ihh, imm, sec
c                                     doy, secd, nweek, sweek, xjd, xmjd
c
c
c        Execute:
c
c        echo xxxxxxxxxx | cal2cal
c          2009 10 10 5 6 7.00 283 18367.00 1552 536767.00 2455114.712 55114.212
c
c
c        1) Until February 28th of 2100 it calculate consistent doys
c           days. Outside this range there will be a discrepancy which
c           increases by one day for every non-leap century year.
c        2) If the year is given with only two digits, the range is
c           between 1980 to 2079
c        --------------------------------------------------------


c        ..... seconds of day ......................................
         secd=dmod(sw,86400.d0)
c        ...........................................................

c        ..... GPS day: (1980jan6.0 => JD=2444244.5 => id_GPS=1.0)...

         xjd=dble(nw)*7.d0+sw/86400.d0+2444244.5d0
         xmjd=xjd-2400000.5d0
c        ............................................................


c        ---------- Julian Day to GPS WEEK (nweek, secw) ----------
c        GPS day: (1980jan6.0 => JD=2444244.5 => id_GPS=1.0 )
         d_GPS=xjd-2444244.5d0
         id_GPS=int(d_GPS)
c        Day of week:
         idw=mod(id_GPS,7)
c        Number of GPS week:
         nweek=(id_GPS-idw)/7
c        seconds into the week:
         sweek=dble(idw)*86400.d0+secd
c        ----------------------------------------------------------


c        ----------- Julian Day to Cal (YYY, MM, DD, hh, mm, sec)---

         a=xjd+0.5d0
         ia=int(a)
         ib=ia+1537
         c=(dble(ib)-122.1d0)/365.25d0
         ic=int(c)
         c=dble(ic)
         d=365.25d0*c
         id=int(d)
         e=dble(ib-id)/30.6001d0
         ie=int(e)
         xe=30.6001*dble(ie)
         iday=ib-id-int(xe)
         ye=dble(ie)/14.d0
         imonth=ie-1-12*int(ye)
         ym=dble(7+imonth)/10.d0
         iyear=ic-4715-int(ym)

c        hh, mm, sec ...................
         hh=secd/3600.d0
         ihh=int(hh+1.d-10)
         xmm=(hh-dble(ihh))*60.d0
         imm=int(xmm+1.d-10)
         xss=(xmm-dble(imm))*60.d0

c       ------------ Cal 2 Doy -------------------------------------
        xy=dble(iyear)
        if (imonth.le.2) then
           xy=xy-1.d0
        endif

        idoy=int(xmjd)-int(365.25d0*(xy-1.d0))+678591
c       -----------------------------------------------------------

c        write(*,'(i4,4(1x,i2),1x,f18.8,2(1x,i4,1x,f15.7),1x,
c     *      f15.7,1x,f15.7)') iyear,imonth,iday,ihh,imm,xss,
c     *                        idoy,secd,nweek,sweek,xjd,xmjd



         return
         end

c        ------------------------------------------------------

         subroutine sub_doy2cal(iyear0,idoy,secd,iyear,imonth,
     *                      iday,ihh,imm,xss,nweek,sweek)
         implicit double precision (a-h,o-z)

c        --------------------------------------------------------
c        Compute the  YEAR MONTH DAY HH MM SS from the elapsed
c        seconds since from 6.0 Jan 1980 (GPS week=0 sec=0)
c
c        1) Until February 28th of 2100 it calculates consistent doys
c           (days). Outside this range there will be a discrepancy which
c           increases by one day for every non-leap century year.
c        2) If the year is given with only two digits, the range is
c           between 1980 to 2079
c
c
c       NOTE: This subroutine is basically THE SAME THAN  sub_gpsws2cal
c             but, without needing the "year", BECAUSE IT ASSUMES THAT
c             THE "week" INCLUDES THE ROLLOVER!!
c        ........................................................
c
c              @ gAGE/UPC (group of Astronomy and Geomatics)
c        --------------------------------------------------------

         xy=dble(iyear0)
c        two digits iyear control ...........
         if (xy.lt.100.d0) then
            if (xy.lt.80.d0) then
               xy=xy+2000.d0
             else
               xy=xy+1900.d0
            endif
         endif
c        ....................................

c        Julian day (xjd) at "iyear, idoy, secd" ....................
         jd=int(365.25*(xy-1.d0))+428+idoy
         xjd=dble(jd)+1720981.5d0+secd/86400.d0
c        ...........................................................



c        ---------- Julian Day to GPS WEEK (nweek,secw) ----------
c        GPS day: (1980jan6.0 => JD=2444244.5 => id_GPS=1.0 )
         d_GPS=xjd-2444244.5d0
         id_GPS=int(d_GPS)
c        Day of week:
         idw=mod(id_GPS,7)
c        Number of GPS week:
         nweek=(id_GPS-idw)/7
c        seconds into the week:
         sweek=dble(idw)*86400.d0+secd
c        ----------------------------------------------------------


c        ----------- Julian Doy to Cal (YYY, MM, DD, hh, mm, sec)---

         a=xjd+0.5d0
         ia=int(a)
         ib=ia+1537
         c=(dble(ib)-122.1d0)/365.25d0
         ic=int(c)
         c=dble(ic)
         d=365.25d0*c
         id=int(d)
         e=dble(ib-id)/30.6001
         ie=int(e)
         xe=30.6001*dble(ie)
         iday=ib-id-int(xe)
         ye=dble(ie)/14.d0
         imonth=ie-1-12*int(ye)
         ym=dble(7+imonth)/10.d0
         iyear=ic-4715-int(ym)

c        hh, mm, sec ...................
         hh=secd/3600.d0
         ihh=int(hh+1.d-10)
         xmm=(hh-dble(ihh))*60.d0
         imm=int(xmm+1.d-10)
         xss=(xmm-dble(imm))*60.d0

cc        write(*,'(i4,4(1x,i2),1x,f18.8,2(1x,i4,1x,f15.7),1x,
cc     *      f15.7,1x,f15.7)') iyear,imonth,iday,ihh,imm,xss,
cc     *                        idoy,secd,nweek,sweek,xjd,xmjd

c         goto 10
100      continue

         return
         end
c        ------------------------------------------------------------



c           f77 -o out2sp3 out2sp3.f

