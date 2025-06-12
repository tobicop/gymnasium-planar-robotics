##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################
# Taken from benchmark_planning_env.py for testing       #
# Goals were removed to test manual control              #
##########################################################

"""This is a significantly shortened version of ``BenchmarkPlanningEnv`` used for testing movement, dynamics, collisions and
the manual keyboard controller. The environment is a simple planning environment without goals, proper reward computation,
observation etc.

Movers can be moved from random (x,y) or user-defined start positions. In this environment, positions, velocities,
accelerations, and jerks have the units m, m/s, m/s² and m/s³, respectively.

Action Space
------------

The action space is continuous. The type of action depends on the selected actuator type:

- If ``actuator_type=JERK``, an action

.. math::
    a_j := [j_{1x}, j_{1y}, ..., j_{nx}, j_{ny}]

represents the desired jerks for each mover in x and y direction of the base frame (unit: m/s³), where

.. math::
    j_{1x}, j_{1y}, ..., j_{nx}, j_{ny} \in [-j_{max},j_{max}]

``j_max`` is the maximum possible jerk (see environment parameters) and n denotes the number of movers.

Accordingly, if ``learn_jerk=False``, an action

.. math::
    a_a := [a_{1x}, a_{1y}, ..., a_{nx}, a_{ny}]

represents the accelerations for each mover in x and y direction of the base frame (unit: m/s²), where

.. math::
    a_{1x}, a_{1y}, ..., a_{nx}, a_{ny} \in [-a_{max},a_{max}]

``a_max`` is the maximum possible acceleration (see environment parameters) and n denotes the number of movers.

- If ``actuator_type=VELOCITY``, an action

.. math::
    a_v := [v_{1x}, v_{1y}, ..., v_{nx}, v_{ny}]

represents the velocities for each mover in x and y direction of the base frame (unit: m/s), where

.. math::
    v_{1x}, v_{1y}, ..., v_{nx}, v_{ny} \in [-v_{max},v_{max}]

``v_max`` is the maximum possible velocity (see environment parameters) and n denotes the number of movers.

Immediate Rewards
-----------------

The agent receives a reward of 50 if all movers reach their goals without collisions. In case of a collision either with another mover
or with a wall, the agent receives a reward of -50. For each timestep in which at least one mover has not reached its goal and in which
there is no collision, the environment emits the following immediate reward:
number of movers that have not reached their goals * (-1)

Episode Termination
-------------------

Each episode has a time limit of 50 environment steps. If the time limit is reached, the episode is truncated. Thus, each episode has
50 environment steps, except that all movers have reached their goals or there has been a collision. In these cases, the episode
terminates immediately, regardless of the time limit.

Version History
---------------

- v0: initial version of the environment

Parameters
----------

"""

import numpy as np
import gymnasium as gym
from gymnasium import logger
import mujoco
import enum
from gymnasium_planar_robotics import BasicPlanarRoboticsSingleAgentEnv
from gymnasium_planar_robotics.utils import mujoco_utils, torque_control
from gymnasium_planar_robotics import Matplotlib2DViewer


class CustomTestingEnv(BasicPlanarRoboticsSingleAgentEnv):
    """A simple planning environment.

    :param layout_tiles: a numpy array of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base
        frame is in the upper left corner.
    :param num_movers: the number of movers to add
    :param show_2D_plot: whether to show a 2D matplotlib plot (useful for debugging)
    :param mover_colors_2D_plot: a list of matplotlib colors, one for each mover (only used if ``show_2D_plot=True``), defaults to
        None. None is only accepted if ``show_2D_plot = False``.
    :param tile_params: a dictionary that can be used to specify the mass and size of a tile using the keys 'mass' or 'size',
        defaults to None. Since one planar motor system usually only contains tiles of one type, i.e. with the same mass and size,
        the mass is a single float value and the size must be specified as a numpy array of shape (3,). If set to None or only one
        key is specified, both mass and size or the missing value are set to the following default values:

        - mass: 5.6 [kg]
        - size: [0.24/2, 0.24/2, 0.0352/2] (x,y,z) [m] (note: half-size)
    :param mover_params: a dictionary that can be used to specify the mass and size of each mover using the keys 'mass' or 'size',
        defaults to None. To use the same mass and size for each mover, the mass can be specified as a single float value and the
        size as a numpy array of shape (3,). However, the movers can also be of different types, i.e. different masses and sizes.
        In this case, the mass and size should be specified as numpy arrays of shapes (num_movers,) and (num_movers,3),
        respectively. If set to None or only one key is specified, both mass and size or the missing value are set to the following
        default values:

        - mass: 1.24 [kg]
        - size: [0.155/2, 0.155/2, 0.012/2] (x,y,z) [m] (note: half-size)
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.003 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle: whether to call 'render' after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if 'render_mode != None'.
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: a dictionary that can be used to specify the following collision parameters, defaults to None:

        - collision shape (key: 'shape'): can be 'box' or 'circle', defaults to 'circle'
        - size of the collision shape (key: 'size'), defaults to 0.11 [m]:

            - collision shape 'circle':
                a single float value which corresponds to the radius of the circle, or a numpy array of shape (num_movers,) to specify
                individual values for each mover
            - collision shape 'box':
                a numpy array of shape (2,) to specify x and y half-size of the box, or a numpy array of shape (num_movers, 2) to
                specify individual sizes for each mover

        - additional size offset (key: 'offset'), defaults to 0.0 [m]: an additional safety offset that is added to the size of the
            collision shape. Think of this offset as increasing the size of a mover by a safety margin.
        - additional wall offset (key: 'offset_wall'), defaults to 0.0 [m]: an additional safety offset that is added to the size
            of the collision shape to detect wall collisions. Think of this offset as moving the wall, i.e. the edge of a tile
            without an adjacent tile, closer to the center of the tile.

    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk, defaults to 100.0 [m/s³]
    :param actuator_type: the actuator type to be used (position, velocity, acceleration, jerk) [CustomTestingEnv.ActuatorType]
    :param threshold_pos: the position threshold used to determine whether a mover has reached its goal position, defaults
        to 0.1 [m]
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    class ActuatorType(enum.Enum):
        POSITION = 0
        VELOCITY = 1
        ACCELERATION = 2
        JERK = 3
        TORQUE = 10

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        #TODO: always use 2D plot?
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, any] | None = None,
        mover_params: dict[str, any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, any] | None = None,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        actuator_type: "ActuatorType" = ActuatorType.JERK,   # avoid issues with forward references
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.actuator_type = actuator_type
        self.torque_controllers = []
        self.torque_control_mode = actuator_type == self.ActuatorType.TORQUE
        self.torque_joint_mask = np.array([1, 1, 0, 0, 0, 1], dtype=bool)   # fz, tx, ty disabled in this env

        # initialize for first execution of _custom_xml_string_callback(), will be set properly by BasicPlanarRoboticsEnv afterwards
        self.cycle_time = 1/1000

        # cam config
        default_cam_config = {
            'distance': 1.1,
            'azimuth': 90.0,
            'elevation': -65.0,
            'lookat': np.array([0.44, 0.18, 0.067]),
        }

        super().__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            render_every_cycle=render_every_cycle,
            num_cycles=num_cycles,
            collision_params=collision_params,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        # ensures acceleration limit (F = ma), gets clipped anyway by default
        max_mover_mass = self.mover_mass if isinstance(self.mover_mass, float) else max(self.mover_mass)
        force_acceleration_limit = abs(max_mover_mass * a_max)

        # map maximum velocity, acceleration and jerk to actuator type
        self.motion_constraints = {
            self.ActuatorType.POSITION: 1,      # used as direction (per axis) for manual position control
            self.ActuatorType.VELOCITY: v_max,
            self.ActuatorType.ACCELERATION: a_max,
            self.ActuatorType.JERK: j_max,
            self.ActuatorType.TORQUE: force_acceleration_limit  # used for defining action_space
        }

        # position threshold in m
        self.threshold_pos = threshold_pos
        # reward in case of success
        self.reward_success = 50
        # whether to show a 2D matplotlib plot
        self.show_2D_plot = show_2D_plot
        if self.show_2D_plot and mover_colors_2D_plot is None:
            raise ValueError('Please specify the colors of the movers for the 2D plot.')

        # minimum and maximum possible mover (x,y)-positions
        safety_margin = self.c_size + self.c_size_offset_wall + self.c_size_offset
        self.min_xy_pos = np.zeros(2) + safety_margin
        self.max_xy_pos = (
            np.array([np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)])
            - safety_margin
        )

        # action space (for now only used for kinematics, not for dynamics)
        if self.actuator_type == self.ActuatorType.POSITION:
            # use absolute position limits (possible mover positions)
            as_low = np.tile(self.min_xy_pos, self.num_movers)
            as_high = np.tile(self.max_xy_pos, self.num_movers)
        else:
            # use defined motion constraints
            as_low = -self.motion_constraints[self.actuator_type]
            as_high = self.motion_constraints[self.actuator_type]
        self.action_space = gym.spaces.Box(low=as_low, high=as_high, shape=(self.num_movers * 2,), dtype='float64')

        # minimum distance between any two goals
        if self.c_shape == 'circle':
            self.min_goal_dist = 2 * (self.c_size + self.c_size_offset)
        else:
            # self.c_shape == 'box'
            self.min_goal_dist = 2 * np.linalg.norm(self.c_size + self.c_size_offset, ord=2)
        
        # initialize torque controllers for all movers if enabled
        if self.torque_control_mode:
            for idx in range(self.num_movers):
                mover_mass = self.mover_mass if isinstance(self.mover_mass, float) else self.mover_mass[idx]
                mover_size = 2 * (self.mover_size if self.mover_size.shape == (3,) else self.mover_size[idx])
                mover_xy_sq_sum = mover_size[0]**2 + mover_size[1]**2      # mover_length² + mover_width²

                # ensure translational acceleration limit (F = ma)
                force_limit = abs(mover_mass * a_max)

                # ensure tangential acceleration limit <= a_max (for z-rotation, using circumscribed radius of mover)
                moment_of_inertia = (mover_mass * mover_xy_sq_sum) / 12     # approximating mover by rectangular cuboid
                radius = np.sqrt(mover_xy_sq_sum) / 2                       # half of mover's diagonal
                torque_limit = abs((a_max * moment_of_inertia) / radius)    # small overshoot due to mover body approximation

                # ensure rotational velocity limit to match z-rotation limit of 10 Hz
                ang_velocity_limit = 10 * 2 * np.pi    # [rad/s]

                # ensure x- and y-rotation limit of ±1.5° (avoid touching ground in MuJoCo, XPlanar supports ±5°)
                angle_xy_limit = np.deg2rad(1.5)

                self.torque_controllers.append(
                    torque_control.MoverTorqueController(
                        model=self.model,
                        mover_joint_name=self.mover_joint_names[idx],
                        force_limit=force_limit,
                        velocity_limit=v_max,
                        torque_limit=torque_limit,
                        ang_velocity_z_limit=ang_velocity_limit,
                        angle_xy_limit=angle_xy_limit
                    )
                )
        self.reload_model()     # needed for some proper initialization (see pushing_env)

        # remember actuator names
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_x'
        )
        self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_y'
        )

        # 2D plot
        if self.show_2D_plot:
            self.matplotlib_2D_viewer = Matplotlib2DViewer(
                layout_tiles=layout_tiles,
                num_movers=self.num_movers,
                mover_sizes=self.mover_size,
                mover_colors=mover_colors_2D_plot,
                tile_size=self.tile_size,
                x_pos_tiles=self.x_pos_tiles,
                y_pos_tiles=self.y_pos_tiles,
                c_shape=self.c_shape,
                c_size=self.c_size,
                c_size_offset=self.c_size_offset,
                arrow_scale=0.2,
                figure_size=(9, 9),
                torque_mode=self.torque_control_mode,
            )

            # set/overwrite axis control value for manual control to the current max control value
            self.matplotlib_2D_viewer.manual_controller.set_axis_control_value(self.motion_constraints[self.actuator_type])
    
    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """For each mover, this callback adds the appropriate actuator XML strings to the ``custom_model_xml_strings`` dictionary,
        depending on the selected actuator type (jerk, acceleration, velocity, or position).

        - For ``ActuatorType.JERK``: Adds MuJoCo general actuators with integrator dynamics for jerk control.
        - For ``ActuatorType.ACCELERATION``: Adds MuJoCo general actuators for direct acceleration control.
        - For ``ActuatorType.VELOCITY``: Adds MuJoCo velocity actuators for velocity control (further adjustments necessary).
        - For ``ActuatorType.POSITION``: Adds MuJoCo position actuators for position control (not fully implemented).

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified the current ``custom_model_xml_strings``-dict
        """
        mover_actuator_xml_str = '\n\n\t<actuator>' + '\n\t\t<!-- mover actuators -->'
        
        for idx_mover in range(0, self.num_movers):
            joint_name = f'mover_joint_{idx_mover}'

            if self.torque_control_mode and self.torque_controllers:
                # torque actuator
                mover_actuator_xml_str += self.torque_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover)
            else:
                # kinematics actuators
                mover_mass = self.mover_mass if isinstance(self.mover_mass, float) else self.mover_mass[idx_mover]
                match self.actuator_type:
                    case self.ActuatorType.JERK:
                        mover_actuator_xml_str += (
                            f'\n\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="integrator" '
                            + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>'
                            + f'\n\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                            + f'dyntype="integrator" gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>'
                        )
                    case self.ActuatorType.ACCELERATION:
                        mover_actuator_xml_str += (
                            f'\n\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none" '
                            + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>'
                            + f'\n\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" dyntype="none" '
                            + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>'
                        )
                    case self.ActuatorType.VELOCITY | self.ActuatorType.POSITION:
                        # position control uses velocity actuators + separate (custom) controller
                        kv_gain = mover_mass / self.cycle_time     # kv = mover_mass * 1/timestep
                        mover_actuator_xml_str += (
                            f'\n\t\t<velocity name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" kv="{kv_gain}"/>'
                            + f'\n\t\t<velocity name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" kv="{kv_gain}"/>'
                        )

        mover_actuator_xml_str += '\n\t</actuator>'

        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
        custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
        if custom_outworldbody_xml_str is not None:
            custom_outworldbody_xml_str += mover_actuator_xml_str
        else:
            custom_outworldbody_xml_str = mover_actuator_xml_str
        custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate a new model xml string with new start and goal positions and reload the model. In this environment, it is necessary
        to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
        """
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

    def _reset_callback(self, options: dict[str, any] | None = None) -> None:
        """Reset the start and goal positions of all movers and reload the model. It is also checked whether the start positions are
        collision-free (mover and wall collisions) and whether the new goals can be reached without mover or wall collisions.

        :param options: A dictionary containing optional parameters for resetting the environment. The following keys are supported:
            - 'mover_pos': a numpy array of shape (num_movers, 2) specifying the start positions of the movers.
            - 'goal_pos': a numpy array of shape (num_movers, 2) specifying the goal positions of the movers.
        :raises ValueError: If user-defined positions are invalid (i.e., collisions are detected).
        """
        # init new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        def init_collision_check(start_qpos):
            """Check if the start positions are collision-free, between movers and walls as well as between movers themselves.

            :param start_qpos: a numpy array of shape (num_movers, 7) containing the start positions of the movers
            :return: True if the positions are collision-free, False otherwise
            """
            # check wall collision
            pos_is_valid = self.qpos_is_valid(qpos=start_qpos, c_size=self.c_size, add_safety_offset=True)
            # check mover collision
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=True, mover_qpos=start_qpos
            )
            return not mover_collision and pos_is_valid.all()

        if options and 'mover_pos' in options:
            # check and set user-defined start positions
            assert(options['mover_pos'].shape == (self.num_movers, 2))
            start_qpos[:, :2] = options['mover_pos']
            if not init_collision_check(start_qpos):
                raise ValueError("User-defined mover position invalid! Collision detected.")
        else:
            # randomly sample start positions
            # ensure that the start positions are chosen such that there no wall or mover collisions
            counter = 0
            while True:
                counter += 1
                if counter > 0 and counter % 100 == 0:
                    logger.warn(
                        'Trying to find a collision-free configuration of start positions for all movers. '
                        + f'No valid configuration found within {counter} trails. Consider choosing fewer movers or more tiles.'
                    )

                start_qpos[:, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(self.num_movers, 2))
                if init_collision_check(start_qpos):
                    break

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2])

    def sample_action(self) -> np.ndarray:
        """Sample a valid action for the current actuator type:
        - If torque control mode is enabled, returns a sampled 2D translation force from the torque controller.
        - Otherwise, samples an action from the environment's action space (jerk, acceleration, or velocity).

        :return: A numpy array representing a valid action for the current actuator type, per mover.
        """
        if self.torque_control_mode:
            return np.stack([tc.sample_translation_2d() for tc in self.torque_controllers], axis=0)
        # else: standard kinematics (jerk, acceleration, velocity)
        return self.action_space.sample()
    
    def torque_control_step(self, force: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, any]]:
        """Set the desired force/torque (Cartesian wrench) for torque control mode.

        :param force: a numpy array of shape (num_movers, 6) with fx, fy, fz, tx, ty, tz (fz, tx, ty disabled in this env)
        """
        assert self.torque_control_mode
        assert force.shape == (self.num_movers, 6)

        # fz, tx, ty disabled in this env
        for idx, tc in enumerate(self.torque_controllers):
            tc.set_desired_force(force[idx] * self.torque_joint_mask)
        
        # pass empty action (zeros) to execute step() while applying only the specified force wrench
        empty_action = np.zeros((self.num_movers * 2), dtype=np.float64)
        return super().step(empty_action)
    
    # override function for torque controller support
    def step(self, action: int | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, any]]:
        """Take a step in the environment using the provided action.

        If torque control mode is enabled, the action is interpreted as a force/torque vector and handled by
        `torque_control_step()`. Otherwise, the action is passed to the parent class's `step()` method.

        :param action: The action to take. In torque control mode, this should be a numpy array of shape (6,) representing
                       the desired force/torque (fx, fy, fz, tx, ty, tz). Otherwise, it should match the action format for
                       the selected actuator type.
        :return: A tuple containing (observation, reward, terminated, truncated, info) as defined by the environment.
        """
        if self.torque_control_mode:
            return self.torque_control_step(action)
        # else
        return super().step(action)
    
    def _mujoco_step_callback(self, action: np.ndarray) -> None:
        """Apply the next action, i.e. it sets the jerk, acceleration or velocity, ensuring the minimum and maximum values
        (for one cycle). The force limits of the torque controller are handled there.

        :param action: a numpy array of shape (num_movers * 2,), which specifies the next action (jerk or acceleration)
        """
        if self.torque_control_mode:
            # only update the torque controllers, skip the rest
            for tc in self.torque_controllers:
                tc.update(model=self.model, data=self.data)
            return
    
        action = action.reshape((self.num_movers, 2))

        for idx_mover in range(0, self.num_movers):
            mover_name = self.mover_names[idx_mover]
            current_vel = self.get_mover_qvel(mover_name=mover_name, add_noise=True)[:2]

            match self.actuator_type:
                case self.ActuatorType.JERK:
                    acc = self.get_mover_qacc(mover_name=mover_name, add_noise=False)[:2]
                    next_acc_tmp, next_jerk = self.ensure_max_dyn_val(
                        current_values=acc, max_value=self.motion_constraints[self.ActuatorType.ACCELERATION], next_derivs=action[idx_mover, :]
                    )
                    _, next_acc = self.ensure_max_dyn_val(
                        current_values=current_vel, max_value=self.motion_constraints[self.ActuatorType.VELOCITY], next_derivs=next_acc_tmp)
                    if (next_acc_tmp != next_acc).any():
                        next_jerk = (next_acc - acc) / self.cycle_time
                    ctrl = next_jerk.copy()
                case self.ActuatorType.ACCELERATION:
                    _, next_acc = self.ensure_max_dyn_val(
                        current_values=current_vel, max_value=self.motion_constraints[self.ActuatorType.VELOCITY], next_derivs=action[idx_mover, :])
                    ctrl = next_acc.copy()
                case self.ActuatorType.VELOCITY:
                    ctrl = self.compute_next_velocity(
                        current_vel=current_vel,
                        desired_vel=action[idx_mover],
                        v_max=self.motion_constraints[self.ActuatorType.VELOCITY],
                        a_max=self.motion_constraints[self.ActuatorType.ACCELERATION],
                    )
                case self.ActuatorType.POSITION:
                    v_max = self.motion_constraints[self.ActuatorType.VELOCITY]
                    a_max = self.motion_constraints[self.ActuatorType.ACCELERATION]

                    # simple P-controller, moving to absolute position
                    kp = a_max      # proportional gain; a_max ensures maximum allowed acc. and fastest possible response without overshooting
                    current_pos = self.get_mover_qpos(mover_name=mover_name)[:2]
                    error = action[idx_mover] - current_pos
                    desired_vel = kp * error

                    ctrl = self.compute_next_velocity(current_vel=current_vel, desired_vel=desired_vel, v_max=v_max, a_max=a_max)
                case _:
                    raise ValueError(f"ActuatorType {self.actuator_type.name} not implemented")

            mujoco_utils.set_actuator_ctrl(
                model=self.model, data=self.data, actuator_name=self.mover_actuator_x_names[idx_mover], value=ctrl[0, 0]
            )

            mujoco_utils.set_actuator_ctrl(
                model=self.model, data=self.data, actuator_name=self.mover_actuator_y_names[idx_mover], value=ctrl[0, 1]
            )

    def _render_callback(self) -> None:
        """Update the Matplotlib2DViewer if ``show_2D_plot=True``."""
        if self.show_2D_plot:
            mover_qpos = self.get_mover_qpos_arr(mover_names=self.mover_names, add_noise=False)
            mover_qvel = self.get_mover_qvel_arr(mover_names=self.mover_names, add_noise=False)
            self.matplotlib_2D_viewer.render(mover_qpos=mover_qpos, mover_qvel=mover_qvel, mover_goals=None)

    def compute_terminated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. A state is terminal when there is a collision between two movers or between a
        mover and a wall.

        :param achieved_goal: unused in this test environment
        :param desired_goal: unused in this test environment
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,). An entry is True if the state is terminal, False otherwise
            - if batch_size = 1:
                True if the state is terminal, False otherwise
        """
        reward = self.compute_reward(info=info)
        terminated = np.bitwise_or(reward == self.reward_success, reward == -self.reward_success)
        return terminated

    def compute_truncated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. The truncation condition (a time limit in this environment) is
        automatically checked by the Gymnasium TimeLimit Wrapper, which is why this method always returns False.

        :param achieved_goal: unused in this test environment
        :param desired_goal: unused in this test environment
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: False
        """
        return False

    def compute_reward(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward only from wall and mover collisions.

        :param achieved_goal: unused in this test environment
        :param desired_goal: unused in this test environment
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value containing the immediate rewards
        """

        _, mover_collisions, wall_collisions = self._preprocess_info_dict(info=info)

        mask_collision = np.bitwise_or(mover_collisions, wall_collisions)
        mask_no_collision = np.bitwise_not(mask_collision)

        reward = -self.reward_success * mask_collision.astype(np.float64)
        reward += -1.0 * self.num_movers * mask_no_collision.astype(np.float64)
        return reward[0]

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a generic numpy array instead of dict as goals are unused in this test environment
        """
        return np.ndarray

    def _get_info(self, mover_collision: bool, wall_collision: bool) -> dict[str, any]:
        """Return a dictionary that contains auxiliary information.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :return: the info dictionary with keys 'is_success', 'mover_collision' and 'wall_collision'
        """
        is_success = not mover_collision and not wall_collision
        assert not isinstance(is_success, np.ndarray)
        assert not isinstance(mover_collision, np.ndarray)
        assert not isinstance(wall_collision, np.ndarray)
        info = {'is_success': is_success, 'mover_collision': mover_collision, 'wall_collision': wall_collision}
        return info

    def close(self) -> None:
        """Close the environment."""
        super().close()
        if self.show_2D_plot:
            self.matplotlib_2D_viewer.close()

    def ensure_max_dyn_val(
        self, current_values: np.ndarray, max_value: float, next_derivs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure the minimum and maximum dynamic values.

        :param current_values: the current velocity or acceleration specified as a numpy array of shape (2,) or
            (num_checks,2)
        :param max_value: the maximum velocity or acceleration (float)
        :param next_derivs: the next derivative (acceleration or jerk) used for one integrator step specified as a numpy array of
            shape (2,) or (num_checks,2)
        :return: the next velocity or acceleration and the next derivative (acceleration or jerk) corresponding to the next action
            that must be applied to ensure the minimum and maximum dynamics (each of shape (num_checks,2))
        """
        if len(current_values.shape) == 1:
            current_values = current_values.reshape((1, -1))
        if len(next_derivs.shape) == 1:
            next_derivs = next_derivs.reshape((1, -1))

        next_values = np.zeros((current_values.shape[0], 2))
        next_derivs_new = np.zeros((current_values.shape[0], 2))

        next_values_tmp = self.cycle_time * next_derivs + current_values

        norm_next_values_tmp = np.linalg.norm(next_values_tmp, ord=2, axis=1)
        mask_norm = norm_next_values_tmp >= max_value

        next_values[np.bitwise_not(mask_norm), :] = next_values_tmp[np.bitwise_not(mask_norm), :]
        next_derivs_new[np.bitwise_not(mask_norm), :] = next_derivs[np.bitwise_not(mask_norm), :]

        if mask_norm.any():
            next_values[mask_norm] = max_value * np.divide(
                next_values_tmp[mask_norm], np.tile(norm_next_values_tmp[mask_norm], reps=(1, 2))
            )
            next_derivs_new[mask_norm] = (next_values[mask_norm] - current_values[mask_norm]) / self.cycle_time

        return next_values, next_derivs_new

    def compute_next_velocity(self, current_vel: np.ndarray, desired_vel: np.ndarray, v_max: float, a_max: float) -> np.ndarray:
        """Compute the next velocity for a mover given the current velocity, desired velocity, and motion constraints.
        This method calculates the acceleration required to reach the desired velocity, clips the acceleration to the defined maximum,
        updates the velocity accordingly and then clips the resulting velocity to the defined maximum velocity.

        :param current_vel: The current velocity of the mover as a numpy array of shape (2,)
        :param desired_vel: The desired velocity of the mover as a numpy array of shape (2,)
        :param v_max: The maximum allowed velocity (float)
        :param a_max: The maximum allowed acceleration (float)
        :return: The next velocity as a numpy array of shape (1, 2)
        """
        # compute acceleration to reach desired velocity
        desired_acc = (desired_vel - current_vel) / self.cycle_time

        # clip acceleration (mainly happens when current_vel is close to v_max)
        acc_norm = np.linalg.norm(desired_acc)
        if acc_norm > a_max and acc_norm > 0:
            desired_acc *= a_max / acc_norm

        # update with acceleration within limits
        next_vel = current_vel + desired_acc * self.cycle_time

        # clip velocity
        vel_norm = np.linalg.norm(next_vel)
        if vel_norm > v_max and vel_norm > 0:
            next_vel *= v_max / vel_norm

        return next_vel.reshape((1, -1))

    def _preprocess_info_dict(self, info: np.ndarray | dict[str, any]) -> tuple[int, np.ndarray, np.ndarray]:
        """Extract information about mover collisions, wall collisions and the batch size from the info dictionary.

        :param info: the info dictionary or an array of info dictionary to be preprocessed. All dictionaries must contain the keys
            'mover_collision' and 'wall_collision'.
        :return: the batch_size (int), a numpy array of shape (batch_size,) containing the mover collision values (bool),
            a numpy array of shape (batch_size,) containing the wall collision values (bool)
        """
        if isinstance(info, np.ndarray):
            batch_size = info.shape[0]
            mover_collisions = np.zeros(batch_size).astype(bool)
            wall_collisions = np.zeros(batch_size).astype(bool)

            for i in range(0, batch_size):
                mover_collisions[i] = info[i]['mover_collision']
                wall_collisions[i] = info[i]['wall_collision']
        else:
            assert isinstance(info, dict)
            batch_size = 1
            mover_collisions = np.array([info['mover_collision']])
            wall_collisions = np.array([info['wall_collision']])

        return batch_size, mover_collisions, wall_collisions
