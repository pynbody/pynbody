namespace kernels
{

  template <typename T>
  class Kernel;

  template <typename T>
  class CubicSplineKernel;

  template <typename T>
  class WendlandC2Kernel;

  template <typename T>
  class Kernel
  {
  public:
    Kernel() {}
    virtual ~Kernel() {}
    virtual T operator()(T r_over_h_squared) const = 0;
    virtual T gradient(T r_over_h_squared, T r_squared) const = 0;
    static std::shared_ptr<Kernel<T>> create(int kernel_id, int n_smooth)
    {
      // if you create a custom kernel, it needs to be added here
      switch (kernel_id)
      {
      case 0:
        return std::make_shared<CubicSplineKernel<T>>();
      case 1:
        return std::make_shared<WendlandC2Kernel<T>>(n_smooth);
      default:
        throw std::runtime_error("Unknown kernel id");
      }
    }
  };

  template <typename T>
  class CubicSplineKernel : public Kernel<T>
  {
  public:
    CubicSplineKernel() {}
    virtual ~CubicSplineKernel() {}
    virtual T operator()(T r_over_h_squared) const override
    {
      T r_over_h = sqrt(r_over_h_squared);
      T rs = 2.0 - r_over_h;
      if NPY_UNLIKELY (rs < 0)
        rs = 0;
      else if (r_over_h_squared < 1.0)
        rs = (1.0 - 0.75 * rs * r_over_h_squared);
      else
        rs = 0.25 * rs * rs * rs;
      return rs;
    }
    virtual T gradient(T r_over_h_squared, T r_squared) const override
    {
      T r_over_h = sqrt(r_over_h_squared);
      T r = sqrt(r_squared);
      T rs;
      if (r_over_h < 1e-10)
        return 0.0;
      else if (r_over_h < 1.0)
        rs = -3.0 * r_over_h + 2.25 * r_over_h_squared;
      else
        rs = -0.75 * (2 - r_over_h) * (2 - r_over_h);

      return rs / r;
    }
  };

  template <typename T>
  class WendlandC2Kernel : public Kernel<T>
  {
  private:
    int nSmooth;
  public:
    WendlandC2Kernel() = delete;
    WendlandC2Kernel(int nSmooth) : nSmooth(nSmooth) {}
    virtual ~WendlandC2Kernel() {}
    virtual T operator()(T r_over_h_squared) const override
    {
      T rs;

      if NPY_UNLIKELY(r_over_h_squared > 4.0)
        rs = 0;
      else if NPY_UNLIKELY(r_over_h_squared == 0.0)
        rs = (21 / 16.) * (1 - 0.0294 * pow(nSmooth * 0.01, -0.977));
      else {
        T au = sqrt(r_over_h_squared * 0.25);
        rs = pow(1 - au, 4);
        rs = (21 / 16.) * rs * (1 + 4 * au);
      }

      return rs;

    }
    virtual T gradient(T r_over_h_squared, T r_squared) const override
    {
      T rs;
      T q = sqrt(r_over_h_squared);
      T r = sqrt(r_squared);

      // Fix to avoid dividing by zero in case r = 0.
      // For this case q = 0 and rs = 0 in any case, so we can savely set r to a
      // tiny value:

      if (r < 1e-24)
        r = 1e-24;

      if (q < 2.0)
        rs = -5.0 * q * (1.0 - 0.5 * q) * (1.0 - 0.5 * q) * (1.0 - 0.5 * q) / r;
      else
        rs = 0.0;

      return rs;
    }
  };
}
