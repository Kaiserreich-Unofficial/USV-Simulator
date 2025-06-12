// #define __INPUTDOT__
#include <usv_dynamics.cuh>

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
        float psi = state(2); // Heading Angle
        float u = state(5);   // Surge Velocity
        float v = state(6);   // Sway Velocity
        float r = state(7);   // Yaw Rate

        // Extract control inputs
        float S_left = control(0);  // Left Thruster Input
        float S_right = control(1); // Right Thruster Input
        Eigen::Matrix3f Jacobian = (Eigen::Matrix3f() << cosf(psi), -sinf(psi), 0, sinf(psi), cosf(psi), 0, 0, 0, 1).finished();
        Eigen::Vector3f nu = Jacobian * Eigen::Vector3f(u, v, r); // Transform the control inputs to the body frame
        // Compute the dynamics
        state_der(0) = nu(0);
        state_der(1) = nu(1);
        state_der(2) = nu(2);
        state_der(3) = -sinf(psi);
        state_der(4) = cosf(psi);
        state_der(5) = u_dot(u, v, r, S_left, S_right);
        state_der(6) = v_dot(u, v, r, S_left, S_right);
        state_der(7) = r_dot(u, v, r, S_left, S_right);
    }

    // 连续动力学方程（CUDA设备）
    __device__ void USVDynamics::computeDynamics(float *state, float *control, float *state_der,
                                                 float *theta_s)
    {
        // Extract state variables
        float psi = state[2]; // Heading Angle
        float u = state[5];   // Surge Velocity
        float v = state[6];   // Sway Velocity
        float r = state[7];   // Yaw Rate

        // Extract control inputs
        float S_left = control[0];  // Left Thruster Input
        float S_right = control[1]; // Right Thruster Input

        // Compute the dynamics
        state_der[0] = __cosf(psi) * u - __sinf(psi) * v;
        state_der[1] = __sinf(psi) * u + __cosf(psi) * v;
        state_der[2] = r;
        state_der[3] = -__sinf(psi); // CPSI_DOT
        state_der[4] = __cosf(psi); // SPSI_DOT
        state_der[5] = u_dot(u, v, r, S_left, S_right);
        state_der[6] = v_dot(u, v, r, S_left, S_right);
        state_der[7] = r_dot(u, v, r, S_left, S_right);
    }

    // 从输入数据映射到状态
    Dynamics<USVDynamics, USVDynamicsParams>::state_array
    USVDynamics::stateFromMap(const std::map<std::string, float> &map)
    {
        state_array s;
        s(0) = map.at("POS_X");
        s(1) = map.at("POS_Y");
        s(2) = map.at("POS_PSI");
        s(3) = map.at("POS_CPSI");
        s(4) = map.at("POS_SPSI");
        s(5) = map.at("VEL_U");
        s(6) = map.at("VEL_V");
        s(7) = map.at("VEL_R");
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
        control = control.cwiseMin(0.5).cwiseMax(-0.5); // 限制控制量在 -0.5 到 0.5 之间
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
            control[i] = fminf(fmaxf(-0.5, control[i]), 0.5);
        }
    }
}

