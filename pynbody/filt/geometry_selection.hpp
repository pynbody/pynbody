template<typename T>
inline void wrapfn(T & x, T & y, T & z, const T & wrap, const T & wrap_by_two) {
    if (x > wrap_by_two) x -= wrap;
    if (y > wrap_by_two) y -= wrap;
    if (z > wrap_by_two) z -= wrap;
    if (x < -wrap_by_two) x += wrap;
    if (y < -wrap_by_two) y += wrap;
    if (z < -wrap_by_two) z += wrap;
}

template<typename T>
inline void perform_selection(T* position_array, char* output_array,
                              T x0, T y0, T z0, T max_radius,
                              Py_ssize_t num_particles) {
    T max_radius_by_2 = max_radius/2;
    for(Py_ssize_t i = 0; i < num_particles; i++) {
        T x = position_array[i*3];
        T y = position_array[i*3+1];
        T z = position_array[i*3+2];
        T dx = x - x0;
        T dy = y - y0;
        T dz = z - z0;
        wrapfn(dx, dy, dz, max_radius, max_radius_by_2);
        T r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < max_radius*max_radius) {
            output_array[i] = 1;
        }
    }
}