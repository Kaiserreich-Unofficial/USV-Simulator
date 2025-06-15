// #define __INPUTDOT__
#include <usv_dynamics.cuh>

namespace wamv
{
    __host__ __device__ inline float u_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return Tl / 180 + Tr / 180 + r * v - (u * ((362 * fabsf(u)) / 5 + 513 / 10)) / 180;
    }

    __host__ __device__ inline float v_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return -(2 * v) / 9 - r * u;
    }

    __host__ __device__ inline float r_dot(const float &u, const float &v, const float &r, const float &Tl, const float &Tr)
    {
        return (201 * Tr) / 89200 - (201 * Tl) / 89200 - (200 * r) / 223;
    }

    // 构造函数
    USVDynamics::USVDynamics(cudaStream_t stream)
        : Dynamics<USVDynamics, USVDynamicsParams>(stream)
    {
        this->params_ = USVDynamicsParams();
    }

    // 计算动力学方程
    void USVDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
                                      const Eigen::Ref<const control_array> &control,
                                      Eigen::Ref<state_array> state_der)
    {
        // Extract state variables
        const float psi = state(2); // Heading Angle
        const float u = state(3);   // Surge Velocity
        const float v = state(4);   // Sway Velocity
        const float r = state(5);   // Yaw Rate

        // Extract control inputs
        const float S_left = control(0);  // Left Thruster Input
        const float S_right = control(1); // Right Thruster Input

        // Compute the dynamics
        state_der(0) = cosf(psi) * u - sinf(psi) * v;
        state_der(1) = sinf(psi) * u + cosf(psi) * v;
        state_der(2) = r;
        state_der(3) = u_dot(u, v, r, S_left, S_right);
        state_der(4) = v_dot(u, v, r, S_left, S_right);
        state_der(5) = r_dot(u, v, r, S_left, S_right);
    }

    // 连续动力学方程（CUDA设备）
    __device__ void USVDynamics::computeDynamics(float *state, float *control, float *state_der,
                                                 float *theta_s)
    {
        // Extract state variables
        const float psi = state[2]; // Heading Angle
        const float u = state[3];   // Surge Velocity
        const float v = state[4];   // Sway Velocity
        const float r = state[5];   // Yaw Rate

        // Extract control inputs
        const float S_left = control[0];  // Left Thruster Input
        const float S_right = control[1]; // Right Thruster Input

        // Compute the dynamics
        state_der[0] = __cosf(psi) * u - __sinf(psi) * v;
        state_der[1] = __sinf(psi) * u + __cosf(psi) * v;
        state_der[2] = r;
        state_der[3] = u_dot(u, v, r, S_left, S_right);
        state_der[4] = v_dot(u, v, r, S_left, S_right);
        state_der[5] = r_dot(u, v, r, S_left, S_right);
    }

    // 从输入数据映射到状态
    Dynamics<USVDynamics, USVDynamicsParams>::state_array
    USVDynamics::stateFromMap(const std::map<std::string, float> &map)
    {
        state_array s;
        s(0) = map.at("POS_X");
        s(1) = map.at("POS_Y");
        s(2) = map.at("POS_PSI");
        s(3) = map.at("VEL_U");
        s(4) = map.at("VEL_V");
        s(5) = map.at("VEL_R");
        return s;
    }

    void USVDynamics::printState(float *state)
    {
        printf("X position: %.2f; Y position: %.2f; Heading Angle: %.2f \n", state[0], state[1], state[2]);
    }

    void USVDynamics::printState(const float *state)
    {
        printf("X position: %.2f; Y position: %.2f; Heading Angle: %.2f \n", state[0], state[1], state[2]);
    }

    // 施加控制约束（主机）
    __host__ void USVDynamics::enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
    {
        // 声明 state 未使用
        (void)state;
        control = control.cwiseMin(250).cwiseMax(-100); // 限制控制量在 -100 到 250 之间
    }

    // 施加控制约束（CUDAs）
    __device__ void USVDynamics::enforceConstraints(float *state, float *control)
    {
        // TODO should control_rngs_ be a constant memory parameter
        int i, p_index, step;
        mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
        // parallelize setting the constraints with y dim
        for (i = p_index; i < CONTROL_DIM; i += step)
        {
            control[i] = fminf(fmaxf(-100, control[i]), 250); // 限制控制量在 -100 到 250 之间 (CUDA)
        }
    }
}
