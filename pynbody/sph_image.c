//
// C code snippet for SPH image rendering.
// Will be compiled by scipy.weave
// The Kernel object will supply the following macros
//  KERNEL(x,y,z,h) - evaluates the kernel for a specified cartesian offset and smoothing length
//                    (supplying x,y,z rather than d allows anisotropic (e.g. derivative) kernels)    
//  MAX_D_OVER_H    - the largest value of d/h for which the kernel can be non-zero
//  Z_CONDITION(dz) - a particle inclusion condition given the offset of the particle
//                    in the direction towards the viewer

float dx = (x2-x1)/nx;
float dy = (y2-y1)/ny;
float x_start = x1+dx/2;
float y_start = y1+dy/2;
int n_part = x_array->dimensions[0];
int nn=0;

for(int i=0; i<n_part; i++) {
  float x_i=X1(i), y_i=Y1(i), z_i=Z1(i), sm_i=SM1(i), qty_i=QTY1(i)*MASS1(i)/RHO1(i);

  if(  Z_CONDITION(z_i-z1, sm_i) && x_i>x1-2*sm_i && x_i<x2+2*sm_i && y_i>y1-2*sm_i && y_i<y2+2*sm_i) {
    if(sm_i/dx<1 && sm_i/dy<1) {
      int x_pos = int((x_i-x1)/dx);
      int y_pos = int((y_i-y1)/dy);
      float z_pixel = z1;
      float x_pixel = dx*x_pos + x_start;
      float y_pixel = dy*y_pos + y_start;
      if(x_pos>=0 && x_pos<nx && y_pos>=0 && y_pos<ny)
	RESULT2(x_pos,y_pos)+=qty_i * (KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
      
    } else {
      int x_pix_start = (x_i-x1-MAX_D_OVER_H*sm_i)/dx;
      int x_pix_stop = (x_i-x1+MAX_D_OVER_H*sm_i)/dx;
      if(x_pix_start<0) x_pix_start=0;
      if(x_pix_stop>=nx) x_pix_stop=nx-1;
      float z_pixel = z1;

      for( int x_pos = x_pix_start; x_pos<=x_pix_stop; x_pos++) {
	int y_pix_start = (y_i-y1-MAX_D_OVER_H*sm_i)/dy;
	int y_pix_stop = (y_i-y1+MAX_D_OVER_H*sm_i)/dy;
	if(y_pix_start<0) y_pix_start=0;
	if(y_pix_stop>=ny) y_pix_stop=ny-1;
	float x_pixel = dx*x_pos + x_start;

	for( int y_pos =y_pix_start ; y_pos<=y_pix_stop; y_pos++) {
	  
	  float y_pixel = dy*y_pos + y_start; 

	  /* Debug help
	  if(nn<100) 
	    printf("%d %d %f %f %f\n", int((y_i-y1)/dy), y_pos, y_pixel, y_i, sm_i);
	  nn++;
	  */

	  RESULT2(x_pos,y_pos)+=qty_i*(KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
	}
      }
    }
  }
 }
