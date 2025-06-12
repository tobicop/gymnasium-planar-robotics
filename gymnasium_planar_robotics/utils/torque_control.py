import numpy as np
import mujoco
from mujoco import MjModel, MjData
from gymnasium_planar_robotics.utils import mujoco_utils, rotations_utils

class MoverTorqueController:
    """Controller for applying Cartesian force/torque (wrench) control to a mover body.
    This class generates actuator XML strings for MuJoCo, manages actuator names, and
    computes and applies joint torques based on a desired wrench using the Jacobian.
    Each mover should have its own instance of this controller.

    :param model: mjModel of the MuJoCo environment
    :param mover_joint_name: the name of the joint of the mover in the MuJoCo model
    :param force_limit: optional force limit to ensure minimum and maximum acceleration [N]
    :param velocity_limit: optional limit for translational velocity [m/s]
    :param torque_limit: optional torque limit for rotational actuation [Nm]
    :param ang_velocity_z_limit: optional angular velocity limit (only z-rotation for now) [rad/s]
    :param angle_xy_limit: optional angle limit for x- and y- rotation [rad]
    """

    def __init__(
        self,
        model: MjModel,
        mover_joint_name: str,
        force_limit: float | None = None,
        velocity_limit: float | None = None,
        torque_limit: float | None = None,
        ang_velocity_z_limit: float | None = None,
        angle_xy_limit: float | None = None,
    ) -> None:
        self.desired_wrench = np.zeros(6)  # for direct torque input (fx, fy, fz, tx, ty, tz)
        
        self.mover_joint_name = mover_joint_name
        self.mover_body_id = model.joint(self.mover_joint_name).bodyid[0]
        self.mover_dofadr = model.body(self.mover_body_id).dofadr[0]
        self.mover_dofnum = model.body(self.mover_body_id).dofnum[0]
        self.mover_idx = int(self.mover_dofadr / self.mover_dofnum)

        self.force_limit = force_limit
        self.velocity_limit = velocity_limit
        self.torque_limit = torque_limit
        self.ang_velocity_z_limit = ang_velocity_z_limit
        self.angle_xy_limit = angle_xy_limit
        
    def generate_actuator_xml_string(self, idx_mover: int):
        """Generate an actuator xml string which can be added to the MuJoCo model xml string. This method must be called manually by
        the user after the controller has been initialized.

        :param idx_mover: the index of the mover (can be found in the body name of the mover, e.g. for a mover with index 0: 'mover_0')
        :return: the actuator xml string
        """
        self.actuator_names = []
        actuator_xml_str = ''
        names = ['x', 'y', 'z', 'a', 'b', 'c']

        for idx in range(0, 6):
            str_gear = '0 0 0 0 0 0'
            str_gear = str_gear[: 2 * idx] + '1' + str_gear[2 * idx + 1 :]
            self.actuator_names.append(f'mover_actuator_{names[idx]}_{idx_mover}')
            actuator_xml_str += (
                f'\n\t\t<general name="{self.actuator_names[-1]}" joint="{self.mover_joint_name}" gear="{str_gear}" '
                + 'dyntype="none" gaintype="fixed" biastype="none"/>'       # mover mass is handled in basic_envs
            )

        return actuator_xml_str

    def update(self, model: MjModel, data: MjData):
        """Compute and apply joint torques to achieve the desired Cartesian wrench (fx, fy, fz, tx, ty, tz),
        subject to force, torque or velocity limits defined by the user.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        """

        # compute Jacobians
        jacp = np.zeros((3, model.nv))  # tanslational part of the Jacobian
        jacr = np.zeros((3, model.nv))  # rotational part of the Jacobian
        mujoco.mj_jacBody(model, data, jacp, jacr, self.mover_body_id)
        jac = np.vstack((jacp, jacr))
        jac = jac[:, self.mover_dofadr : self.mover_dofadr + self.mover_dofnum]

        # direct wrench to joint torque via Jacobian
        ctrl = (jac.T @ self.desired_wrench.reshape((6, 1))).flatten()

        # apply velocity limit (translation) (small overshoot due to simulation effects)
        if self.velocity_limit is not None:
            fz_vel = data.qvel[self.mover_dofadr : self.mover_dofadr + 3]
            fz_vel_norm = np.linalg.norm(fz_vel)
            
            if fz_vel_norm > self.velocity_limit and fz_vel_norm > 0:
                # only allow force that reduces velocity magnitude
                vel_dir = fz_vel / fz_vel_norm
                # project force onto velocity direction
                force_along_v = np.dot(ctrl[:3], vel_dir)
                # if force would accelerate further, remove that component
                if force_along_v > 0:
                    ctrl[:3] -= force_along_v * vel_dir        

        # apply force (hence acceleration) limit (translation)
        if self.force_limit is not None:
            force_vec = ctrl[:3]
            force_norm = np.linalg.norm(force_vec)
            if force_norm > self.force_limit and force_norm > 0:
                ctrl[:3] = force_vec * (self.force_limit / force_norm)

        # apply angle limit with virtual damping (x- and y-rotation) (small overshoot and oscillation)
        #TODO: set values depending on max acceleration as well to improve performance and flexibility
        if self.angle_xy_limit is not None:
            # higher -> more aggressive, less overshoot, more oscillation on same torque
            spring_gain_factor = 25

            # overwrite dofnum and dofadr as qpos have 7 DoF (x_p,y_p,z_p,w_o,x_o,y_o,z_o)
            dofadr = self.mover_idx * 7

            # Euler angles
            tx_rad, ty_rad = rotations_utils.quat2euler(data.qpos[dofadr + 3:dofadr + 7])[:2]
            # angular velocities
            tx_vel, ty_vel = data.qvel[self.mover_dofadr + 3], data.qvel[self.mover_dofadr + 4]

            # set limits to "default" values if not defined
            torque_limit = 0.453 if self.torque_limit is None else self.torque_limit
            ang_velocity_limit = 20*np.pi if self.ang_velocity_z_limit is None else self.ang_velocity_z_limit

            # set gain values s.t. damping starts at margin and uses maximum (opposite) torque when reaching limit
            damping_gain = torque_limit / ang_velocity_limit
            spring_gain = spring_gain_factor * (torque_limit / self.angle_xy_limit)

            # limit angles (tx/roll, ty/pitch) with virtual spring + damping
            if abs(tx_rad) > self.angle_xy_limit:
                ctrl[3] = -spring_gain * (tx_rad - np.sign(tx_rad) * self.angle_xy_limit) - damping_gain * tx_vel
            if abs(ty_rad) > self.angle_xy_limit:
                ctrl[4] = -spring_gain * (ty_rad - np.sign(ty_rad) * self.angle_xy_limit) - damping_gain * ty_vel

        # apply angular velocity limit (z-rotation only) (small overshoot due to hard clipping)
        #TODO: apply limit to all rotation axes?
        if self.ang_velocity_z_limit is not None:
            tz_vel = data.qvel[self.mover_dofadr + 5]
            if tz_vel > self.ang_velocity_z_limit and tz_vel > 0:
                ctrl[5] = 0.0 

        # apply torque limit (rotation)
        if self.torque_limit is not None:
            torque_vec = ctrl[3:6]
            torque_norm = np.linalg.norm(torque_vec)
            if torque_norm > self.torque_limit and torque_norm > 0:
                ctrl[3:6] = torque_vec * (self.torque_limit / torque_norm)

        for idx in range(self.mover_dofnum):
            mujoco_utils.set_actuator_ctrl(model, data, actuator_name=self.actuator_names[idx], value=ctrl[idx])

    def set_desired_force(self, force: np.ndarray):
        """Set the desired force/torque (Cartesian wrench) for torque control mode.

        :param force: a numpy array of shape (6,) with fx, fy, fz, tx, ty, tz
        """
        assert force.shape == (6,)
        self.desired_wrench = force

    def sample_translation_2d(self) -> np.ndarray:
        """Sample a random 6D force/torque vector [fx, fy, fz, tx, ty, tz] such that:
        - The vector [fx, fy] has a uniformly random direction in 2D.
        - The magnitude is uniformly sampled in [0, force_limit) (which depends on a_max by default).
        - The result is a 6D vector (np.ndarray) with all other values being zero.
        """
        # uniform direction in 2D
        angle = np.random.uniform(0, 2 * np.pi)
        # uniform magnitude in [0, a_max)
        mag = np.random.uniform(0, self.force_limit)
        
        # scale components
        fx = mag * np.cos(angle)
        fy = mag * np.sin(angle)

        force = np.zeros(6)
        force[0] = fx
        force[1] = fy
        return force
