#include <stdio.h>
#include <math.h>

double mtot,x,y,z,rp,rcut,com_new[3];
long i,npart_new=0; 
int j;

rcut = double(r);
npart_new = npart;

while(npart_new > min_particles)
  {
    for(i=0;i<3;i++) com_new[i] = 0.0;
    mtot=0.0;
    npart_new = 0;
    
    for(i=0;i<npart;i++)
      {
        rp = 0.0;
        for(j=0;j<3;j++) rp += (pos[i*3+j]-com[j])*(pos[i*3+j]-com[j]);
        rp = sqrt(rp);

        if(double(rp) < double(rcut))
          {
            for(j=0;j<3;j++) com_new[j] += mass[i]*pos[i*3+j];
            mtot+=mass[i];
            npart_new++;
          }
      }
    if (npart_new > 0)
      {
        for(i=0;i<3;i++) com[i] = com_new[i]/mtot;
        if(verbose)    
          fprintf(stderr, "[ %.13f %.13f %.13f] %.13f %d\n", com[0],com[1],com[2],rcut,npart_new);
        rcut *= 0.7;
      }
    else break;
  }


