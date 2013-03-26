//
// C code snippet for projecting SPH particles onto a 3d grid
// Will be compiled by scipy.weave
// The Kernel object will supply the following macros
//  KERNEL(x,y,z,h) - evaluates the kernel for a specified cartesian offset and smoothing length
//                    (supplying x,y,z rather than d allows anisotropic (e.g. derivative) kernels)    
//  MAX_D_OVER_H    - the largest value of d/h for which the kernel can be non-zero

float dx = (x2-x1)/nx;
float dy = (y2-y1)/ny;
float dz = (z2-z1)/nz;

#define pixel_dx dx
#define X_TO_XI(xpos) int((xpos-x1)/dx)
#define XI_TO_X(xpix) (dx*xpix+x_start)
#define Y_TO_YI(ypos) int((ypos-y1)/dy)
#define YI_TO_Y(ypix) (dy*ypix+y_start)
#define Z_TO_ZI(zpos) int((zpos-z1)/dz)
#define ZI_TO_Z(zpix) (dz*zpix+z_start)


float x_start = x1+dx/2;
float y_start = y1+dy/2;
float z_start = z1+dz/2;
int n_part = x_array->dimensions[0];
int nn=0;
bool abort=false;

using std::abs;
using std::sqrt;

#ifdef THREAD
Py_BEGIN_ALLOW_THREADS;
#endif

for(int i=0; i<n_part; i++) {
  if(abort) continue;
  float x_i=X1(i), y_i=Y1(i), z_i=Z1(i), sm_i=SM1(i), qty_i=QTY1(i)*MASS1(i)/RHO1(i);

#ifdef SMOOTH_RANGE
  if((sm_i<pixel_dx*smooth_lo) || (sm_i>pixel_dx*smooth_hi)) continue;
#endif

#ifndef THREAD
  if(i%1000==0) {
    if(PyErr_CheckSignals()!=0)
      abort = true;
  }
#endif

    {
 
      if( (MAX_D_OVER_H*sm_i/pixel_dx<1 && MAX_D_OVER_H*sm_i/pixel_dx<1)) {
      
      int x_pos = X_TO_XI(x_i);
      int y_pos = Y_TO_YI(y_i);
      int z_pos = Z_TO_ZI(z_i);
      float x_pixel = XI_TO_X(x_pos);
      float y_pixel = YI_TO_Y(y_pos);
      float z_pixel = ZI_TO_Z(z_pos);

      if(x_pos>=0 && x_pos<nx && y_pos>=0 && y_pos<ny && z_pos>=0 && z_pos<nz)
	  RESULT3(x_pos,y_pos,z_pos)+=qty_i * (KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
      
    } else {

      int x_pix_start = X_TO_XI(x_i-MAX_D_OVER_H*sm_i);
      int x_pix_stop = X_TO_XI(x_i+MAX_D_OVER_H*sm_i);
      if(x_pix_start<0) x_pix_start=0;
      if(x_pix_stop>=nx) x_pix_stop=nx-1;
      float z_pixel = z1;

      for( int x_pos = x_pix_start; x_pos<=x_pix_stop; x_pos++) {
	int y_pix_start = Y_TO_YI(y_i-MAX_D_OVER_H*sm_i);
	int y_pix_stop = Y_TO_YI(y_i+MAX_D_OVER_H*sm_i);
	if(y_pix_start<0) y_pix_start=0;
	if(y_pix_stop>=ny) y_pix_stop=ny-1;
	float x_pixel = XI_TO_X(x_pos);

	for( int y_pos =y_pix_start ; y_pos<=y_pix_stop; y_pos++) {
	  
	    int z_pix_start = Z_TO_ZI(z_i-MAX_D_OVER_H*sm_i);
	    int z_pix_stop = Z_TO_ZI(z_i+MAX_D_OVER_H*sm_i);
	    if(z_pix_start<0) z_pix_start=0;
	    if(z_pix_stop>=nz) z_pix_stop=nz-1;
	    float y_pixel = YI_TO_Y(y_pos);
	    for( int z_pos =z_pix_start ; z_pos<=z_pix_stop; z_pos++) {
		float z_pixel = ZI_TO_Z(z_pos);
		float result = qty_i*(KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
		RESULT3(x_pos,y_pos,z_pos)+=result;
		}
	    }
      }
    }
  }
 }

#ifdef THREAD
Py_END_ALLOW_THREADS;
#endif
