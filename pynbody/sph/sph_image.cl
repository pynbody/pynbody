//
// OpenCL implementation of SPH image rendering
//


// The Kernel object will supply the following macros
//  KERNEL(x,y,z,h) - evaluates the kernel for a specified cartesian offset and smoothing length
//                    (supplying x,y,z rather than d allows anisotropic (e.g. derivative) kernels)    
//  MAX_D_OVER_H    - the largest value of d/h for which the kernel can be non-zero
//  Z_CONDITION(dz) - a particle inclusion condition given the offset of the particle
//                    in the direction towards the viewer

typedef struct param
{
        float dx;
        float dy;
        float x1;
        float y1;
        int n_particles;
} param;

__kernel void render(__global const float *sm, __global const float *qty,
                     __global const float *pos,             
                     __constant struct param* params,
                     __global float *dest,
                     local float *local_pos_cache,
                     local float *local_sm_cache,
                     local float *local_weight_cache) {
         
         int nx = get_global_size(0);
         int ny = get_global_size(1);
         int X = get_global_id(0);
         int Y = get_global_id(1);
         
         float dx = params->dx;
         float dy = params->dy;
         float x1 = params->x1;
         float y1 = params->y1;
         int n_particles = params->n_particles;

         float sum=0;
         int n_block = get_local_size(0)*get_local_size(1);

         
         //event_t event = async_work_group_copy(local_pos_cache, pos, 3000, 0);
         //wait_group_events(1, &event);
         int copy_offset = get_local_id(1)+get_local_id(0)*get_local_size(1);
         for(int block_offset=0; block_offset<n_particles; block_offset+=n_block) {

           local_pos_cache[copy_offset*3] = ((pos[(block_offset+copy_offset)*3]-x1)/dx);
           local_pos_cache[copy_offset*3+1] =  ((pos[(block_offset+copy_offset)*3+1]-y1)/dy);
           local_pos_cache[copy_offset*3+2] =  ((pos[(block_offset+copy_offset)*3+2]-y1)/dy);
           local_sm_cache[copy_offset] = sm[block_offset+copy_offset]/dx;
           local_weight_cache[copy_offset] = qty[block_offset+copy_offset];
           barrier(CLK_LOCAL_MEM_FENCE);

           for(int r=0; r<n_block; r++) {
             //if((local_pos_cache[r*2]==X)&&(local_pos_cache[r*2+1]==Y))              
             //  sum+=rho[r];
             float dX = (X-local_pos_cache[r*3]);
             float dY = (Y-local_pos_cache[r*3+1]);
             float dZ = (local_pos_cache[r*3+2]);
             float h = local_sm_cache[r];
             sum+=local_weight_cache[r]*(DISTANCE(dX,dY,dZ)>h && h>0)?(1.0/(h*h*h)):0;
             
           }

           barrier(CLK_LOCAL_MEM_FENCE);
         }
         
         dest[X+nx*Y]=sum;
                  
}

