#ifndef USV_DYNAMIC_CUH
#define USV_DYNAMIC_CUH

#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>

namespace wamv
{
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
            INPUT_LEFT = 0,
            INPUT_RIGHT,
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
