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
#define X_TO_XI(xpos) int((xpos-x1)/dx)
#define XI_TO_X(xpix) (dx*xpix+x_start)
#define Y_TO_YI(ypos) int((ypos-y1)/dy)
#define YI_TO_Y(ypix) (dy*ypix+y_start)
#endif 


float x_start = x1+dx/2;
float y_start = y1+dy/2;
int n_part = x_array->dimensions[0];
int nn=0;

using std::abs;
using std::sqrt;


for(int i=0; i<n_part; i++) {
  float x_i=X1(i), y_i=Y1(i), z_i=Z1(i), sm_i=SM1(i), qty_i=QTY1(i)*MASS1(i)/RHO1(i);

#ifdef PERSPECTIVE
  if(z_i>0.9*z_camera) continue;
  float pixel_dx = (z_camera-z_i)*ddx;
  
#else
  if(  (Z_CONDITION(z_i-z1, sm_i)) && x_i>x1-2*sm_i && x_i<x2+2*sm_i && y_i>y1-2*sm_i && y_i<y2+2*sm_i) 
#endif

    {
 
      if( (sm_i/dx<1 && sm_i/dy<1)) {
      
      float z_pixel = z1;

      int x_pos = X_TO_XI(x_i);
      int y_pos = Y_TO_YI(y_i);
      float x_pixel = XI_TO_X(x_pos);
      float y_pixel = YI_TO_Y(y_pos);

      if(x_pos>=0 && x_pos<nx && y_pos>=0 && y_pos<ny)
	RESULT2(y_pos,x_pos)+=qty_i * (KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
      
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
	  
	  float y_pixel = YI_TO_Y(y_pos);

	  RESULT2(y_pos,x_pos)+=qty_i*(KERNEL(x_i-x_pixel, y_i-y_pixel, z_i-z_pixel ,sm_i));
	}
      }
    }
  }
 }
