import numpy as np
import mujoco
from mujoco import MjModel, MjData
from gymnasium_planar_robotics.utils import mujoco_utils

class MoverTorqueController:
    """Controller for applying Cartesian force/torque (wrench) control to a mover body.
    This class generates actuator XML strings for MuJoCo, manages actuator names, and
    computes and applies joint torques based on a desired wrench using the Jacobian.

    :param model: mjModel of the MuJoCo environment
    :param mover_joint_name: the name of the joint of the mover in the MuJoCo model
    :param force_limit: optional force limit to ensure minimum and maximum acceleration
    """

    def __init__(
        self,
        model: MjModel,
        mover_joint_name: str,
        force_limit: float | None = None,
    ) -> None:
        self.desired_wrench = np.zeros(6)  # for direct torque input (fx, fy, fz, tx, ty, tz)
        
        self.mover_joint_name = mover_joint_name
        self.mover_body_id = model.joint(self.mover_joint_name).bodyid[0]
        self.mover_dofadr = model.body(self.mover_body_id).dofadr[0]
        self.mover_dofnum = model.body(self.mover_body_id).dofnum[0]

        self.force_limit = force_limit
        
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
        """Compute and apply joint torques to achieve the desired Cartesian wrench. If a force limit was
        defined by the user, it will be applied to the translational acceleration here.

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

        # if passed initially, apply acceleration limits to force vector (first 3 components)  # TODO: limit velocity? how?
        if self.force_limit is not None:
            force_vec = ctrl[:3]
            norm = np.linalg.norm(force_vec)
            if norm > self.force_limit and norm > 0:
                ctrl[:3] = force_vec * (self.force_limit / norm)

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
