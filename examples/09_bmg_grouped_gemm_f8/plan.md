Overall plan for the implementation of grouped GEMM with f8 data type:
User new APIs referred in the [design doc](./media/docs/cpp/xe_rearchitecture.md)
1. Change sycl-tla/examples/09_bmg_grouped_gemm_f8/09_bmg_grouped_gemm_f8.cpp to use new APIs
2.New APIs/ atoms to use are 
    a. XE_LOAD_2D
    b. XE_STORE_2D
    c.XE_LOAD_2D
    d.XE_LOAD_VNNI_2D
    e.XE_STORE_VNNI_2D
    f.XE_DPAS
3. Rewrite collective MMA using new APIs in include/cutlass/gemm/collective/xe_array_mma_fp8.hpp
a. Use XE_DPAS for MMA
b. Use XE_LOAD_2D, XE_STORE_2D for loading/storing tiles
c. Use XE_LOAD_VNNI_2D, XE_STORE_VNNI_2D for loading/storing tiles in VNNI layout
d. Modify the operator() to take care of f8 to f16 conversion and new APIs
4. Copy traits XE_LOAD_2D is missing for few data types
a.Add new copy traits in include/cute/atom/copy_traits_xe_2d.hpp
5. Compile and test the example



  