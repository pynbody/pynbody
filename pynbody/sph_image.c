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

#ifdef PERSPECTIVE
float ddx = dx/z_camera;
float mid_x = (x2+x1)/2;
float mid_y = (y2+y1)/2;

#define X_TO_XI(xpos) (int((xpos-mid_x)/pixel_dx+nx/2))
#define XI_TO_X(xpix) (pixel_dx*(xpix-nx/2)+mid_x)
#define Y_TO_YI(ypos) (int((ypos-mid_y)/pixel_dx+ny/2))
#define YI_TO_Y(ypix) (pixel_dx*(ypix-ny/2)+mid_y)

#else
#define pixel_dx dx
#define pixel_dy dy
#define X_TO_XI(xpos) int((xpos-x1)/dx)
#define XI_TO_X(xpix) (dx*xpix+x_start)
#define Y_TO_YI(ypos) int((ypos-y1)/dy)
#define YI_TO_Y(ypix) (dy*ypix+y_start)
#endif 


float x_start = x1+dx/2;
float y_start = y1+dy/2;
int n_part = x_array->dimensions[0];
int nn=0;
bool abort=false;

using std::abs;
using std::sqrt;

#ifdef THREAD
Py_BEGIN_ALLOW_THREADS;
#endif

int smooth_lo_i = smooth_lo;
int smooth_hi_i = smooth_hi;

for(int i=0; i<n_part; i++) {
  if(abort) continue;
  float x_i=X1(i), y_i=Y1(i), z_i=Z1(i), sm_i=SM1(i), qty_i=QTY1(i)*MASS1(i)/RHO1(i);


#ifndef THREAD
  if(i%1000==0) {
    if(PyErr_CheckSignals()!=0)
      abort = true;
  }
#endif



#ifdef PERSPECTIVE
#ifndef SMOOTH_IN_PIXELS
  if(z_i>0.99*z_camera) continue;
  if(z_i>0.8*z_camera)
    qty_i*=exp((0.8-z_i/z_camera)/0.1);
#endif
  if(z_i<-z_camera*100) continue;
  if(z_i<-z_camera)
    qty_i/=-z_i/z_camera;
// originally : qty_i*=exp((1.0+z_i/z_camera));
  float pixel_dx = (z_camera-z_i)*ddx;

#ifdef SMOOTH_IN_PIXELS
  sm_i*=pixel_dx;
#endif
  
  if(sm_i < pixel_dx*0.55) sm_i = pixel_dx*0.55;

#ifdef SMOOTH_RANGE
    if((sm_i<pixel_dx*smooth_lo) || (sm_i>pixel_dx*smooth_hi)) continue;
#endif

#else
#ifdef SMOOTH_RANGE
    if((sm_i<pixel_dx*smooth_lo) || (sm_i>pixel_dx*smooth_hi)) continue;
#endif
  if(  (Z_CONDITION(z_i-z1, sm_i)) && x_i>x1-2*sm_i && x_i<x2+2*sm_i && y_i>y1-2*sm_i && y_i<y2+2*sm_i) 
#endif
 


#ifndef PERSPECTIVE
      if( (MAX_D_OVER_H*sm_i/pixel_dx<1 && MAX_D_OVER_H*sm_i/pixel_dy<1)) {
      
      float z_pixel = z1;

      int x_pos = X_TO_XI(x_i);
      int y_pos = Y_TO_YI(y_i);
      float x_pixel = XI_TO_X(x_pos);
      float y_pixel = YI_TO_Y(y_pos);

      if(x_pos>=0 && x_pos<nx && y_pos>=0 && y_pos<ny)
	RESULT2(y_pos,x_pos)+=qty_i * (KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
      
    } else {
#endif

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
	  
	  float y_pixel = YI_TO_Y(y_pos);
	  float result = qty_i*(KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
	  RESULT2(y_pos,x_pos)+=result;
	}
      }
#ifndef PERSPECTIVE
    }
#endif
 }

#ifdef THREAD
Py_END_ALLOW_THREADS;
#endif
