#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI      3.14159265358979323846
#define TWOPI   (2.0 * PI)
#define HALFPI  (0.5 * PI)

/* Convert angular coordinates (theta, phi) to a unit vector */
void ang2vec(double theta, double phi, double *vec) {
    vec[0] = sin(theta) * cos(phi);
    vec[1] = sin(theta) * sin(phi);
    vec[2] = cos(theta);
}

/* Compute the angular distance between two unit vectors */
double angdist(double *vec1, double *vec2) {
    double dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    return acos(dot);
}

/* Compute the ring number corresponding to a given z */
size_t ring_num_from_z(size_t nside, double z) {
    size_t nring = 4 * nside;
    size_t iring;
    double nside_double = (double)nside;

    if (z > 2.0 / 3.0) {  // North Polar cap
        iring = (size_t)(nside_double * sqrt(3.0 * (1.0 - z)) + 0.5);
    } else if (z >= -2.0 / 3.0) {  // Equatorial region
        iring = (size_t)(nside_double * (2.0 - 1.5 * z) + 0.5);
    } else {  // South Polar cap
        iring = (size_t)(nring - nside_double * sqrt(3.0 * (1.0 + z)) + 0.5);
    }

    if (iring < 1) iring = 1;
    if (iring > nring) iring = nring;
    return iring;
}

/* Convert ring number to z coordinate */
double z_from_ring_num(size_t nside, size_t iring) {
    size_t nl4 = 4 * nside;
    if (iring < 1 || iring > nl4) {
        fprintf(stderr, "Error: Ring number out of bounds.\n");
        exit(1);
    }

    double z;
    if (iring <= nside) {  // North Polar cap
        double tmp = iring;
        z = 1.0 - (tmp * tmp) / (3.0 * nside * nside);
    } else if (iring <= 3 * nside) {  // Equatorial region
        z = (2 * (double) nside - (double) iring ) / (1.5 * nside);
    } else {  // South Polar cap
        double tmp = nl4 - iring;
        z = -1.0 + (tmp * tmp) / (3.0 * nside * nside);
    }
    return z;
}

/* Convert ring number and phi index to pixel index in RING scheme */
size_t ring_and_phi_index_to_pixel_index(size_t nside, size_t iring, size_t iphi) {
    size_t npix = 12 * nside * nside;
    size_t ncap = 2 * nside * (nside + 1);
    size_t ipix;

    if (iring < 1 || iring > 4 * nside) {
        fprintf(stderr, "Error: Ring number out of bounds.\n");
        exit(1);
    }

    size_t nr;
    if (iring <= nside) {  // North Polar cap
        nr = iring;
        ipix = nr * (nr - 1) * 2 + iphi - 1;
    } else if (iring <= 3 * nside) {  // Equatorial region
        nr = nside;
        ipix = ncap + (iring - nside - 1) * 4 * nside + iphi - 1;
    } else {  // South Polar cap
        nr = 4 * nside - iring;
        ipix = npix - nr * (nr - 1) * 2 + iphi - 1 - 4*nr;
    }

    // ipix = (ipix + npix) % npix;  // Ensure ipix is within bounds
    return ipix;
}

/* Query_disc function using the ring-based approach. Also returns an angular distance to the centre */
size_t query_disc_c(size_t nside, double* vec0, double radius, size_t *listpix, double *distpix) {


    double vecnorm = sqrt(vec0[0]*vec0[0] + vec0[1]*vec0[1] + vec0[2]*vec0[2]);
    vec0[0] /= vecnorm;
    vec0[1] /= vecnorm;
    vec0[2] /= vecnorm;

    double z0 = vec0[2];
    double sin_theta0 = sqrt(1-z0*z0);
    double phi0 = atan2(vec0[1], vec0[0]);
    double theta0 = acos(z0);

    double thetamin = theta0 - radius;
    double thetamax = theta0 + radius;

    if(thetamin<0.0) thetamin = 0.0;
    if(thetamax>PI) thetamax = PI;
    double zmax = cos(thetamin);
    double zmin = cos(thetamax);

    size_t irmin = ring_num_from_z(nside, zmax);
    size_t irmax = ring_num_from_z(nside, zmin);

    size_t npix_alloc = 4 * nside * nside;  // Estimate maximum number of pixels

    size_t count = 0;
    double twopi = TWOPI;

    for (size_t iring = irmin; iring <= irmax; iring++) {
        double z = z_from_ring_num(nside, iring);
        double sin_theta = sqrt(1.0 - z * z);
        double cos_dphi = (cos(radius) - z * z0) / (sin_theta * sin_theta0);

        double dphi;
        if (cos_dphi >= 1.0) {
            dphi = 0.0;
        } else if (cos_dphi <= -1.0) {
            dphi = PI;
        } else {
            dphi = acos(cos_dphi);
        }

        size_t ip_lo, ip_hi;

        size_t n_in_ring;
        size_t shift;
        if (iring <= nside) {
            n_in_ring = 4 * iring;
            shift = 1;
        } else if (iring <= 3 * nside) {
            n_in_ring = 4 * nside;
            shift = 2 - (iring - nside + 1) % 2;
        } else {
            n_in_ring = 4 * (4 * nside - iring);
            shift = 1;
        }

        if(dphi>HALFPI) {
            ip_lo = 1;
            ip_hi = n_in_ring;
        } else {

            double phi_low = phi0 - dphi;
            double phi_high = phi0 + dphi;

            if (phi_low < 0.0) phi_low += twopi;
            if (phi_high >= twopi) phi_high -= twopi;

            double phi_factor = n_in_ring / twopi;

            ip_lo = (size_t)(phi_low * phi_factor + 0.5 * shift);
            ip_hi = (size_t)(phi_high * phi_factor + 0.5 * shift+1);
        }


        if (ip_hi < ip_lo) ip_hi += n_in_ring;

        if (ip_hi - ip_lo > n_in_ring) {
            ip_lo = 1;
            ip_hi = n_in_ring;
        }

        for (size_t iphi = ip_lo  ; iphi <= ip_hi  ; iphi++) {
            size_t ip = iphi;
            if (ip > n_in_ring) ip -= n_in_ring;

            size_t ipix = ring_and_phi_index_to_pixel_index(nside, iring, ip);

            // Compute angular distance to the center
            double theta = acos(z);
            double phi = (twopi * ((double) ip - 0.5 * ((double) shift)) / n_in_ring);
            if (phi >= twopi) phi -= twopi;


            double vec[3];
            ang2vec(theta, phi, vec);

            double dist = angdist(vec0, vec);

            // printf("ipix: %ld, z: %f, theta: %f, phi: %f, dist: %f\n", ipix, z, theta, phi, dist);

            if (dist <= radius) {
                distpix[count] = dist;
                listpix[count++] = ipix;
            }
        }
    }

    return count;
}


/*
int main() {
    size_t nside = 8;  // Example nside
    double theta0 = PI / 4;  // Center at 45 degrees
    double phi0 = PI / 4;    // Center at 45 degrees
    double radius = PI / 8;  // Radius of 22.5 degrees

    double vec0[3];
    ang2vec(theta0, phi0, vec0);

    size_t *listpix = malloc(4 * nside * nside * sizeof(size_t));
    size_t npix;

    npix = query_disc_c(nside, vec0, radius, listpix);

    printf("\nNumber of pixels found: %ld\n", npix);
    for (size_t i = 0; i < npix; i++) {
        printf("%ld ", listpix[i]);
    }
    printf("\n");

    free(listpix);
    return 0;
}
*/
