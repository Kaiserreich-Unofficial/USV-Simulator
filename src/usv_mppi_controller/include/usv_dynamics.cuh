#ifndef USV_DYNAMIC_CUH
#define USV_DYNAMIC_CUH

#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>

namespace wamv
{
    // 定义状态变量的微分方程
    __host__ __device__ inline float u_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return -.1586 * u - .0939 * u * fabsf(u) + .5859 * v * r + .7949 * Tl + .8332 * Tr + .1714 * r * r;
    }

    __host__ __device__ inline float v_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return -.0851 * v + .0145 * r - .0085 * v * fabsf(v) - .1418 * r * fabsf(r) - .0078 * u * v - 1.0014 * u * r - .0911 * Tl + .0926 * Tr - .0138 * fabsf(v) * r - .0694 * fabsf(r) * v;
    }

    __host__ __device__ inline float r_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return .1787 * v + .0770 * r - .0320 * v * fabsf(v) - .6022 * r * fabsf(r) - .0262 * u * v - .0081 * u * r - .3729 * Tl + .3778 * Tr - .0654 * fabsf(v) * r - .2661 * fabsf(r) * v;
    }

    struct USVDynamicsParams : public DynamicsParams
    {
        enum class StateIndex : int
        {
            POS_X = 0,
            POS_Y,
            POS_PSI,
            VEL_U,
            VEL_V,
            VEL_R,
            NUM_STATES = 6
        };
        enum class ControlIndex : int
        {
            INPUT_LEFT = 0, // S_left
            INPUT_RIGHT,    // S_right
            NUM_CONTROLS = 2
        };
        enum class OutputIndex : int
        {
            POS_X = 0,
            POS_Y,
            POS_PSI,
            VEL_U,
            VEL_V,
            VEL_R,
            NUM_OUTPUTS = 6
        };
        USVDynamicsParams() = default;
        ~USVDynamicsParams() = default;
    };

    using namespace MPPI_internal;

    class USVDynamics : public Dynamics<USVDynamics, USVDynamicsParams>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        USVDynamics(cudaStream_t stream = nullptr);

        std::string getDynamicsModelName() const override
        {
            return "WAM-V USV Dynamics";
        }

        void computeDynamics(const Eigen::Ref<const state_array> &state, const Eigen::Ref<const control_array> &control,
                             Eigen::Ref<state_array> state_der);

        // bool computeGrad(const Eigen::Ref<const state_array> &state, const Eigen::Ref<const control_array> &control,
        //                  Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

        void printState(float *state);
        void printState(const float *state);

        __device__ void computeDynamics(float *state, float *control, float *state_der, float *theta = nullptr);

        state_array stateFromMap(const std::map<std::string, float> &map);

        // 施加控制约束
        __host__ void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control);
        __device__ void enforceConstraints(float *state, float *control);
    };
}

#if __CUDACC__
#include "usv_dynamics.cu"
#endif

#endif // USV_DYNAMIC_CUH
