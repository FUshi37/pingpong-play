from __future__ import annotations

import itertools
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from recovered_rmp import (
    ConstantAccelerationKalman,
    JointLimits,
    JointRMPPlanner,
    QPProblem,
    RMPParams,
    build_joint_qp,
    build_task_qp,
    combine_qps,
    compute_joint_reference,
    compute_task_reference,
    joint_acceleration_bounds,
    load_movax_rmp_config,
    solve_box_qp,
)
from recovered_rmp.whole_body import WeightedTask
from recovered_rmp.replay import forward_velocity_acceleration


def enumerate_box_qp(problem: QPProblem) -> np.ndarray:
    """Tiny independent oracle used only by the test suite."""

    n = problem.g.size
    best_x = None
    best_cost = np.inf
    for states in itertools.product((-1, 0, 1), repeat=n):
        x = np.zeros(n)
        fixed = np.asarray(states) != 0
        free = ~fixed
        x[np.asarray(states) == -1] = problem.lower[np.asarray(states) == -1]
        x[np.asarray(states) == 1] = problem.upper[np.asarray(states) == 1]
        if np.any(free):
            rhs = -problem.g[free] - problem.H[np.ix_(free, fixed)] @ x[fixed]
            x[free] = np.linalg.solve(problem.H[np.ix_(free, free)], rhs)
        if np.any(x < problem.lower - 1.0e-10) or np.any(x > problem.upper + 1.0e-10):
            continue
        cost = 0.5 * x @ problem.H @ x + problem.g @ x
        if cost < best_cost:
            best_x, best_cost = x, cost
    if best_x is None:
        raise AssertionError("oracle failed to find a feasible active set")
    return best_x


class RecoveredRMPTests(unittest.TestCase):
    def test_forward_velocity_acceleration_matches_log_alignment(self) -> None:
        velocity = np.asarray(
            [[0.0, 1.0], [0.1, 0.8], [0.4, 0.2], [0.8, -0.2]]
        )
        time = np.asarray([10.0, 10.005, 10.010, 10.015])
        sequence = np.asarray([20, 21, 23, 24], dtype=np.int64)
        acceleration, valid = forward_velocity_acceleration(
            velocity, time, sequence
        )
        np.testing.assert_allclose(acceleration[0], [20.0, -40.0])
        np.testing.assert_allclose(acceleration[2], [80.0, -80.0])
        self.assertEqual(valid.tolist(), [True, False, True, False])
        self.assertTrue(np.all(np.isnan(acceleration[1])))
        self.assertTrue(np.all(np.isnan(acceleration[3])))

    def test_joint_reference_and_qp_are_recovered_formula(self) -> None:
        params = RMPParams(jnt_kp=10.0, jnt_kd=2.0, joint_vel_ffd=0.5, joint_acc_ffd=0.25, acc_weight=0.1)
        q = np.array([0.1, -0.2])
        qd = np.array([0.3, -0.1])
        target = np.array([0.4, -0.1])
        qd_target = np.array([0.2, 0.4])
        qdd_target = np.array([1.0, -2.0])
        reference = compute_joint_reference(q, qd, target, qd_target, qdd_target, [100.0, 100.0], params)
        expected = 10.0 * (target - q) + 2.0 * (0.5 * qd_target - qd) + 0.25 * qdd_target
        np.testing.assert_allclose(reference, expected)
        problem = build_joint_qp(reference, params, [-100.0, -100.0], [100.0, 100.0])
        np.testing.assert_allclose(solve_box_qp(problem), expected / 1.1)

    def test_task_qp_matrix_and_vector(self) -> None:
        params = RMPParams(vel_weight=3.0, acc_weight=0.2, pos_limit_weight=0.4)
        J = np.array([[1.0, 2.0], [-0.5, 1.0]])
        Jdqd = np.array([0.2, -0.1])
        reference = np.array([3.0, -2.0])
        q = np.array([0.3, -0.2])
        qd = np.array([0.4, -0.5])
        q_mid = np.array([0.1, 0.1])
        dt = 0.01
        problem = build_task_qp(J, Jdqd, reference, q, qd, q_mid, params, dt, [-10, -10], [10, 10])
        expected_H = J.T @ J + (3.0 * dt * dt + 0.2) * np.eye(2)
        expected_g = -J.T @ (reference - Jdqd) + 3.0 * dt * qd + 4.0 * 0.4 * (q - q_mid)
        np.testing.assert_allclose(problem.H, expected_H)
        np.testing.assert_allclose(problem.g, expected_g)

    def test_task_reference_limits_error_and_acceleration(self) -> None:
        params = RMPParams(kp=100.0, kd=0.0, max_linear_vel=1.0, max_angular_vel=2.0, max_linear_acc=3.0, max_angular_acc=4.0)
        reference, error, gain = compute_task_reference(
            [100, 0, 0, 0, 100, 0], np.zeros(6), np.zeros(6), np.zeros(6), params, 0.001
        )
        self.assertAlmostEqual(np.linalg.norm(error[:3]), 0.2)
        self.assertAlmostEqual(np.linalg.norm(error[3:]), 0.4)
        self.assertAlmostEqual(np.linalg.norm(reference[:3]), 3.0)
        self.assertAlmostEqual(np.linalg.norm(reference[3:]), 4.0)
        self.assertEqual(gain, 0.0)

    def test_active_set_solver_matches_enumeration(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(25):
            matrix = rng.normal(size=(3, 3))
            H = matrix.T @ matrix + 0.2 * np.eye(3)
            g = rng.normal(size=3)
            lower = rng.uniform(-1.5, -0.1, size=3)
            upper = rng.uniform(0.1, 1.5, size=3)
            problem = QPProblem(H, g, lower, upper)
            np.testing.assert_allclose(solve_box_qp(problem), enumerate_box_qp(problem), atol=1.0e-8)

    def test_predictive_bounds_brake_near_position_limit(self) -> None:
        limits = JointLimits(np.array([-1.0]), np.array([1.0]), np.array([5.0]), np.array([10.0]))
        lower, upper = joint_acceleration_bounds([0.999], [0.2], limits, 0.01, angle_gap=0.0)
        self.assertLess(upper[0], 0.0)
        self.assertGreaterEqual(lower[0], -10.0)
        self.assertLessEqual(upper[0], 10.0)

    def test_yaml_loader_recovers_right_arm_si_values(self) -> None:
        root = Path(__file__).resolve().parents[2]
        path = root / "movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml"
        if not path.exists():
            self.skipTest("copied Movax configuration is not present")
        params, limits = load_movax_rmp_config(path, "RightArm")
        self.assertEqual(params.kp, 324.0)
        self.assertAlmostEqual(params.max_angular_vel, np.deg2rad(400.0))
        self.assertAlmostEqual(limits.q_min[0], np.deg2rad(-120.0))
        self.assertAlmostEqual(limits.qdd_max[3], np.deg2rad(3000.0))

    def test_constant_acceleration_filter_tracks_ramp(self) -> None:
        dt = 0.01
        estimator = ConstantAccelerationKalman(1, dt, 0.1, 1.0e-5)
        for step in range(400):
            estimator.update([2.0 * step * dt])
        self.assertAlmostEqual(estimator.velocity[0], 2.0, places=2)

    def test_estimator_deceleration_uses_acceleration_limit(self) -> None:
        estimator = ConstantAccelerationKalman(1, 0.001, 50.0, 0.0001)
        estimator.reset([0.0], velocity=[0.05])
        estimator.decelerate([1.0])
        np.testing.assert_allclose(estimator.acceleration, [-1.0])
        np.testing.assert_allclose(estimator.velocity, [0.049])

    def test_stateful_joint_rmp_returns_consistent_q_qd_qdd(self) -> None:
        limits = JointLimits(
            np.array([-2.0]),
            np.array([2.0]),
            np.array([10.0]),
            np.array([100.0]),
            np.array([10000.0]),
        )
        params = RMPParams(
            jnt_kp=20.0,
            jnt_kd=4.0,
            filter_length=1,
            target_filter_length=1,
            angle_gap=(0.0,),
        )
        planner = JointRMPPlanner(limits, params, 0.001)
        planner.reset([0.0], target_position=[0.0])
        previous_q = planner.q.copy()
        previous_qd = planner.qd.copy()
        command = planner.step([0.2])
        np.testing.assert_allclose(
            command.position,
            previous_q + (previous_qd + 0.5 * planner.dt * command.acceleration) * planner.dt,
        )
        np.testing.assert_allclose(
            command.velocity,
            previous_qd + planner.dt * command.acceleration,
        )
        self.assertTrue(np.all(np.isfinite(command.acceleration)))

    @unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX is not installed")
    def test_jax_joint_rmp_matches_numpy_and_returns_servo_history(self) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)
        from recovered_rmp.jax_backend import (
            init_joint_rmp_state_jax,
            joint_rmp_step_jax,
        )

        limits = JointLimits(
            np.array([-2.0, -2.0]),
            np.array([2.0, 2.0]),
            np.array([4.0, 4.0]),
            np.array([30.0, 30.0]),
            np.array([3000.0, 3000.0]),
        )
        params = RMPParams(
            jnt_kp=80.0,
            jnt_kd=12.0,
            target_filter_length=10,
            angle_gap=(0.0, 0.0),
        )
        initial = np.array([0.1, -0.2])
        numpy_planner = JointRMPPlanner(limits, params, 0.001)
        numpy_planner.reset(initial, target_position=initial)
        jax_state = init_joint_rmp_state_jax(
            initial[None, :], target_filter_length=10
        )
        for sample in range(30):
            target = initial + 0.15 * np.sin(0.11 * sample + np.arange(2))
            for servo_index in range(5):
                numpy_command = numpy_planner.step(
                    target,
                    target_updated=servo_index == 0,
                )
            (
                jax_state,
                q_history,
                qd_history,
                qdd_history,
            ) = joint_rmp_step_jax(
                jax_state,
                target[None, :],
                limits.q_min,
                limits.q_max,
                limits.qd_max,
                limits.qdd_max,
                limits.qddd_max,
                params,
                0.001,
                servo_steps=5,
                return_history=True,
            )
        self.assertEqual(np.asarray(q_history).shape, (5, 1, 2))
        self.assertEqual(np.asarray(qd_history).shape, (5, 1, 2))
        self.assertEqual(np.asarray(qdd_history).shape, (5, 1, 2))
        np.testing.assert_allclose(
            np.asarray(jax_state.command_q)[0], numpy_command.position
        )
        np.testing.assert_allclose(
            np.asarray(jax_state.command_qd)[0], numpy_command.velocity
        )
        np.testing.assert_allclose(
            np.asarray(jax_state.command_qdd)[0], numpy_command.acceleration
        )

    def test_new_targets_are_rate_limited_by_elapsed_servo_ticks(self) -> None:
        limits = JointLimits(
            np.array([-2.0]),
            np.array([2.0]),
            np.array([1.0]),
            np.array([100.0]),
            np.array([10000.0]),
        )
        params = RMPParams(
            filter_length=1,
            target_filter_length=1,
            angle_gap=(0.0,),
        )
        planner = JointRMPPlanner(limits, params, 0.001)
        planner.reset([0.0], target_position=[0.0])
        for servo_index in range(5):
            planner.step([0.0], target_updated=servo_index == 0)
        planner.step([1.0], target_updated=True)
        np.testing.assert_allclose(planner.accepted_target, [0.005])

    def test_jax_target_update_delay_matches_numpy_event_timing(self) -> None:
        try:
            from recovered_rmp.jax_backend import (
                init_joint_rmp_state_jax,
                joint_rmp_step_jax,
            )
        except ImportError:
            self.skipTest("JAX is not installed")
        limits = JointLimits(
            np.array([-2.0]),
            np.array([2.0]),
            np.array([1.0]),
            np.array([100.0]),
            np.array([10000.0]),
        )
        params = RMPParams(
            filter_length=3,
            target_filter_length=2,
            angle_gap=(0.0,),
        )
        planner = JointRMPPlanner(limits, params, 0.001)
        planner.reset([0.0], target_position=[0.0])
        state = init_joint_rmp_state_jax(
            np.zeros((1, 1)),
            target_filter_length=2,
            output_filter_length=3,
        )
        target = np.asarray([0.2])
        for servo_index in range(5):
            command = planner.step(
                target,
                target_updated=servo_index == 3,
            )
        state = joint_rmp_step_jax(
            state,
            target[None, :],
            limits.q_min,
            limits.q_max,
            limits.qd_max,
            limits.qdd_max,
            limits.qddd_max,
            params,
            0.001,
            servo_steps=5,
            target_update_delay_steps=3,
        )
        np.testing.assert_allclose(np.asarray(state.command_q)[0], command.position)
        np.testing.assert_allclose(np.asarray(state.command_qd)[0], command.velocity)
        np.testing.assert_allclose(
            np.asarray(state.command_qdd)[0], command.acceleration
        )

    def test_whole_body_composition_adds_metrics_and_intersects_bounds(self) -> None:
        one = QPProblem(np.eye(2), np.array([-1.0, 0.0]), np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
        two = QPProblem(2.0 * np.eye(2), np.array([0.0, -2.0]), np.array([-1.0, -3.0]), np.array([3.0, 1.0]))
        combined = combine_qps([WeightedTask(one, 2.0), WeightedTask(two, 0.5)])
        np.testing.assert_allclose(combined.H, 3.0 * np.eye(2))
        np.testing.assert_allclose(combined.g, [-2.0, -1.0])
        np.testing.assert_allclose(combined.lower, [-1.0, -2.0])
        np.testing.assert_allclose(combined.upper, [2.0, 1.0])

    def test_mujoco_adapter_returns_spatial_jacobian_and_jdot_qd(self) -> None:
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        from recovered_rmp.mujoco_adapter import single_dof_joint_indices, site_kinematics

        xml = """
        <mujoco><worldbody><body name="link">
          <joint name="j0" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
          <site name="tip" pos="1 0 0"/>
        </body></worldbody></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        angle, speed = 0.2, 0.5
        data.qpos[0] = angle
        data.qvel[0] = speed
        mujoco.mj_forward(model, data)
        _, dofs = single_dof_joint_indices(model, ["j0"])
        kin = site_kinematics(model, data, "tip", dofs)
        expected_jacobian = np.array([-np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.0])
        expected_jdot_qd = np.array(
            [-np.cos(angle) * speed**2, -np.sin(angle) * speed**2, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(kin.jacobian[:, 0], expected_jacobian)
        np.testing.assert_allclose(kin.twist, expected_jacobian * speed)
        np.testing.assert_allclose(kin.jacobian_dot_qd, expected_jdot_qd)

    def test_repository_right_arm_model_runs_end_to_end(self) -> None:
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        root = Path(__file__).resolve().parents[2]
        model_path = root / "src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_pd.xml"
        if not model_path.exists():
            self.skipTest("repository MuJoCo model is not present")
        from recovered_rmp.examples.right_arm_mujoco_step import right_arm_rmp_step

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")
        qdd, action = right_arm_rmp_step(
            model,
            data,
            data.site_xpos[site_id].copy(),
            data.site_xmat[site_id].reshape(3, 3).copy(),
            float(model.opt.timestep),
        )
        self.assertEqual(qdd.shape, (7,))
        self.assertTrue(np.all(np.isfinite(qdd)))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


if __name__ == "__main__":
    unittest.main()
