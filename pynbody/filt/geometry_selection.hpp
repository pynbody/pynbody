#include <tuple>

template<typename T>
class DifferenceWithWrap {
    private:
    T wrap, wrap_by_two;

    public:
    DifferenceWithWrap(T wrap) : wrap(wrap), wrap_by_two(wrap/2) { }

    inline std::tuple<T, T, T> calculate_offset(T x1, T y1, T z1, T x2, T y2, T z2) const {
        T dx = x2 - x1;
        T dy = y2 - y1;
        T dz = z2 - z1;
        /* While the wrapping below might seem a bit convoluted, it is actually
         * the fastest way to do it. The modulo operator is slow. However, this does mean
         * that offsets more than a single wrap away are not handled correctly.
         * See also the note in filter_test.py where this is tested.
         */
        if (dx> wrap_by_two) dx -= wrap;
        if (dy> wrap_by_two) dy -= wrap;
        if (dz> wrap_by_two) dz -= wrap;
        if (dx<-wrap_by_two) dx += wrap;
        if (dy<-wrap_by_two) dy += wrap;
        if (dz<-wrap_by_two) dz += wrap;

        return std::make_tuple(dx, dy, dz);
    }
};

template<typename T>
class DifferenceWithoutWrap {
    public:
    DifferenceWithoutWrap(T wrap) {  } // we still take wrap in the constructor, for uniformity

    inline std::tuple<T, T, T> calculate_offset(T x1, T y1, T z1, T x2, T y2, T z2) const {
        T dx = x2 - x1;
        T dy = y2 - y1;
        T dz = z2 - z1;

        return std::make_tuple(dx, dy, dz);
    }
};

template<typename T, typename WrapPolicy>
class SphereSelector: public WrapPolicy {
    T x0, y0, z0, max_radius2;

    public:
    SphereSelector(T x0, T y0, T z0, T max_radius, T wrap) :
        x0(x0), y0(y0), z0(z0), max_radius2(max_radius * max_radius),
        WrapPolicy(wrap) {}

    inline bool operator()(T x, T y, T z) const {
        T dx, dy, dz;
        std::tie(dx, dy, dz) = this->calculate_offset(x0, y0, z0, x, y, z);

        T r2 = dx*dx + dy*dy + dz*dz;
        return r2 < max_radius2;
    }
};

template<typename T, typename WrapPolicy>
class CubeSelector: public WrapPolicy {
    T xc, yc, zc, xsize, ysize, zsize;
    public:
    CubeSelector(T x0, T y0, T z0, T x1, T y1, T z1, T wrap): xc((x0+x1)/2), yc((y0+y1)/2), zc((z0+z1)/2),
        xsize((x1-x0)/2), ysize((y1-y0)/2), zsize((z1-z0)/2), WrapPolicy(wrap) {}

    inline bool operator()(T x, T y, T z) const {
        T dx, dy, dz;
        std::tie(dx, dy, dz) = this->calculate_offset(xc, yc, zc, x, y, z);
        return (abs(dx) < xsize && abs(dy) < ysize && abs(dz) < zsize);
    }
};

template<typename T, typename Selector>
void perform_selection(T* position_array, char* output_array, Py_ssize_t num_particles,
                       const Selector & selector, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for(Py_ssize_t i = 0; i < num_particles; i++) {
        output_array[i] = selector(position_array[i*3], position_array[i*3+1], position_array[i*3+2]);
    }
}

template<typename T>
inline void sphere_selection(T* position_array, char* output_array,
                              T x0, T y0, T z0, T max_radius,
                              T wrap,
                              Py_ssize_t num_particles, int num_threads) {
    if(wrap>0)
        perform_selection(position_array, output_array, num_particles,
                          SphereSelector<T, DifferenceWithWrap<T>>(x0, y0, z0, max_radius, wrap), num_threads);
    else
        perform_selection(position_array, output_array, num_particles,
                          SphereSelector<T, DifferenceWithoutWrap<T>>(x0, y0, z0, max_radius, 0), num_threads);
}

template<typename T>
inline void cube_selection(T* position_array, char* output_array,
                              T x0, T y0, T z0, T x1, T y1, T z1,
                              T wrap,
                              Py_ssize_t num_particles, int num_threads) {
    if(wrap>0)
        perform_selection(position_array, output_array, num_particles,
                          CubeSelector<T, DifferenceWithWrap<T>>(x0, y0, z0, x1, y1, z1, wrap), num_threads);
    else
        perform_selection(position_array, output_array, num_particles,
                          CubeSelector<T, DifferenceWithoutWrap<T>>(x0, y0, z0, x1, y1, z1, 0), num_threads);
}
