#ifndef USV_DYNAMIC_CUH
#define USV_DYNAMIC_CUH

#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>

namespace heron
{
    // 定义状态变量的微分方程
    __host__ __device__ inline float u_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return (Tl + Tr + 67.3171 * v * r - 26.43 * u) / 39.9;
    }

    __host__ __device__ inline float v_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return (-39.9 * u * r - 72.64 * v) / 67.3171;
    }

    __host__ __device__ inline float r_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return (0.37 * Tl - 0.37 * Tr - 29.3171 * u + 1.9 * u * v - 22.96 * r) / 10.4655;
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
            return "Heron USV Dynamics";
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

namespace wamv
{
    // 定义状态变量的微分方程
    __host__ __device__ inline float u_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return 0.004739 * Tl + 0.004739 * Tr - 0.2431 * u - 0.004739 * r * (54.44 * r - 211.0 * v);
    }

    __host__ __device__ inline float v_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return 0.0005043 * Tl - 0.0005043 * Tr - 0.1978 * r - 0.1947 * v - 1.0 * r * u - 0.2087 * u * v;
    }

    __host__ __device__ inline float r_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return 0.001955 * Tl - 0.001955 * Tr - 0.7666 * r - 0.01978 * v - 0.8088 * u * v;
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
