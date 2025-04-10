##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
from gymnasium_planar_robotics.utils import rotations_utils


def check_line_segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check whether two line segments p and q intersect by considering the orientation of ordered points, as explained here:
    https://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf.
    The two line segments are each defined by two points in the (x,y)-plane (p1 and p2; q1 and q2).
    This function is vectorized and can perform multiple checks. In the case of multiple tests, the line segments given by the points
    ``p1[i,:]``, ``p2[i,:]``, ``q1[i,:]``, ``q2[i,:]`` are tested.

    :param p1: a numpy array of shape (num_checks,2) specifying the first points belonging to p
    :param p2: a numpy array of shape (num_checks,2) specifying the second points belonging to p
    :param q1: a numpy array of shape (num_checks,2) specifying the first points belonging to q
    :param q2: a numpy array of shape (num_checks,2) specifying the second points belonging to q
    :return: a numpy array of shape (num_checks,), where an entry is True if the two line segments intersect, False otherwise
    """
    num_checks = p1.shape[0]
    assert p1.shape == (num_checks, 2)
    assert p2.shape == (num_checks, 2)
    assert q1.shape == (num_checks, 2)
    assert q2.shape == (num_checks, 2)

    ls_intersect = np.zeros(num_checks).astype(bool)

    mask_points_equal = (
        (np.sum((np.abs(p1 - q1) < 1e-7), axis=1) == 2)
        + (np.sum((np.abs(p1 - q2) < 1e-7), axis=1) == 2)
        + (np.sum((np.abs(p2 - q1) < 1e-7), axis=1) == 2)
        + (np.sum((np.abs(p2 - q2) < 1e-7), axis=1) == 2)
    )

    min_xy_p = np.minimum(p1, p2)
    min_xy_q = np.minimum(q1, q2)
    max_xy_p = np.maximum(p1, p2)
    max_xy_q = np.maximum(q1, q2)

    mask_pq = max_xy_p < min_xy_q
    mask_qp = max_xy_q < min_xy_p
    mask_minmax = mask_pq * (1 - (np.abs(max_xy_p - min_xy_q) < 1e-7)) + mask_qp * (1 - (np.abs(max_xy_q - min_xy_p) < 1e-7))
    mask_minmax = np.sum(mask_minmax, axis=1) >= 1

    p11 = np.pad(p1, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    p21 = np.pad(p2, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    q11 = np.pad(q1, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    q21 = np.pad(q2, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    mat_p11p21q11 = np.swapaxes(np.swapaxes(np.array([p11, p21, q11]), 0, 1), 1, 2)
    mat_p11p21q21 = np.swapaxes(np.swapaxes(np.array([p11, p21, q21]), 0, 1), 1, 2)
    mat_q11q21p11 = np.swapaxes(np.swapaxes(np.array([q11, q21, p11]), 0, 1), 1, 2)
    mat_q11q21p21 = np.swapaxes(np.swapaxes(np.array([q11, q21, p21]), 0, 1), 1, 2)

    det_p11p21q11 = np.linalg.det(mat_p11p21q11)
    det_p11p21q21 = np.linalg.det(mat_p11p21q21)
    det_q11q21p11 = np.linalg.det(mat_q11q21p11)
    det_q11q21p21 = np.linalg.det(mat_q11q21p21)

    mask_orientation = ((np.sign(det_p11p21q11 * det_p11p21q21) <= 0) + (np.abs(det_p11p21q11 * det_p11p21q21) < 1e-7)) * (
        (np.sign(det_q11q21p11 * det_q11q21p21) <= 0) + (np.abs(det_q11q21p11 * det_q11q21p21) < 1e-7)
    )

    ls_intersect[mask_orientation] = True
    ls_intersect[mask_minmax] = False
    ls_intersect[mask_points_equal] = True
    return ls_intersect


def get_2D_rect_vertices(qpos: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Get the (x,y) coordinates of the vertices of rectangles w.r.t. the base frame. This function is vectorized and can calculate the
    vertices of multiple rectangles.

    :param qpos: qpos (position and orientation) of the rectangles specified as a numpy array of shape (num_rectangles,7)
        (x_p,y_p,z_p,w_o,x_o,y_o,z_o)
    :param size: length and width (half-size) of the rectangles specified as a numpy array of shape (num_rectangles,2)
    :return: the (x,y) coordinates of the vertices (numpy array of shape (num_rectangles,2,4))
    """
    num_rectangles = qpos.shape[0]
    assert qpos.shape == (num_rectangles, 7)
    assert size.shape == (num_rectangles, 2)

    quats = qpos[:, -4:].copy()
    # ensure that quaternions are normalized
    nomalized_quats = rotations_utils.unit_vector(quats, axis=1)
    rot_mats = rotations_utils.quat2mat(nomalized_quats)
    assert rot_mats.shape == (num_rectangles, 3, 3)
    # vertices w.r.t. local frame of each rectangle
    vertices_l = np.array(
        [
            [-size[:, 0], -size[:, 0], size[:, 0], size[:, 0]],
            [-size[:, 1], size[:, 1], size[:, 1], -size[:, 1]],
            np.zeros((4, num_rectangles)),
        ]
    )
    vertices_l = np.swapaxes(vertices_l, 0, 2)
    vertices_l = np.swapaxes(vertices_l, 1, 2)

    # vertices w.r.t. base frame
    vertices_b = (rot_mats @ vertices_l)[:, :2, :] + np.repeat(qpos[:, :2].reshape((num_rectangles, 2, -1)), 4, axis=2)

    return vertices_b


def check_rectangles_intersect(qpos_r1: np.ndarray, qpos_r2: np.ndarray, size_r1: np.ndarray, size_r2: np.ndarray) -> bool:
    """Check whether two rectangles of any orientation intersect. This function is vectorized and can perform multiple checks, i.e.
    it is checked whether the rectangles with qpos ``qpos_r1[i,:]`` and ``qpos_r2[i,:]`` and sizes ``size_r1[i,:]`` and
    ``size_r2[i,:]`` intersect.

    :param qpos_r1: qpos (position and orientation) of the first rectangles specified as a numpy array of shape (num_checks,7)
        (x_p,y_p,z_p,w_o,x_o,y_o,z_o)
    :param qpos_r2: qpos (position and orientation) of the second rectangles specified as a numpy array of shape (num_checks,7)
        (x_p,y_p,z_p,w_o,x_o,y_o,z_o)
    :param size_r1: length and width (half-size) of the first rectangles specified as a numpy array of shape (num_checks,2)
    :param size_r2: length and width (half-size) of the second rectangles specified as a numpy array of shape (num_checks,2)
    :return: a numpy array of shape (num_checks,), where an entry is True if the rectangles intersect, False otherwise
    """
    num_checks = qpos_r1.shape[0]
    assert qpos_r1.shape == (num_checks, 7)
    assert qpos_r2.shape == (num_checks, 7)
    assert size_r1.shape == (num_checks, 2)
    assert size_r2.shape == (num_checks, 2)

    # vertices rectangles 1 w.r.t. base frame (shape: (num_checks, 2, 4))
    vertices_r1_b = get_2D_rect_vertices(qpos=qpos_r1, size=size_r1)
    # vertices rectangles 2 w.r.t. base frame (shape: (num_checks, 2, 4))
    vertices_r2_b = get_2D_rect_vertices(qpos=qpos_r2, size=size_r2)

    # line segments
    p1 = np.swapaxes(np.repeat(vertices_r1_b, 4, axis=2), 1, 2).reshape((num_checks * 16, 2))
    p2 = np.swapaxes(np.repeat(np.roll(vertices_r1_b, shift=-1, axis=2), 4, axis=2), 1, 2).reshape((num_checks * 16, 2))
    q1 = np.swapaxes(np.tile(vertices_r2_b, reps=(1, 1, 4)), 1, 2).reshape((num_checks * 16, 2))
    q2 = np.swapaxes(np.tile(np.roll(vertices_r2_b, shift=-1, axis=2), reps=(1, 1, 4)), 1, 2).reshape((num_checks * 16, 2))

    res = check_line_segments_intersect(p1=p1, p2=p2, q1=q1, q2=q2)
    return np.sum(res.reshape((num_checks, 16)), axis=1) >= 1

def calculate_mover_distances(
        mover_qpos: np.ndarray,
        c_size_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the distances between movers and prepare collision size arrays.

    :return: A tuple containing:
        - mover_dist: The distances between movers (flattened list of unique pairs i < j).
        - mover_i_qpos: The qpos of the first mover in each pair.
        - mover_j_qpos: The qpos of the second mover in each pair.
        - c_size_arr_i: The collision size arrays for the first mover in each pair.
        - c_size_arr_j: The collision size arrays for the second mover in each pair.
        - mover_dist_table: Full (N x N) symmetric distance matrix with np.inf on diagonal.
    """
    num_movers = mover_qpos.shape[0]
    assert mover_qpos.shape == (num_movers, 7)

    num_checks = np.sum(np.arange(start=1, stop=num_movers, step=1))
    mover_i_qpos = np.zeros((num_checks, 7))
    mover_j_qpos = np.zeros((num_checks, 7))
    c_size_arr_i = np.zeros((num_checks, c_size_arr.shape[1]))
    c_size_arr_j = np.zeros((num_checks, c_size_arr.shape[1]))

    # initialize full distance matrix
    mover_dist_matrix = np.full((num_movers, num_movers), np.inf)

    start_idx = 0
    for i in range(0, num_movers - 1):
        offset_idx = num_movers - (i + 1)
        stop_idx = start_idx + offset_idx
        mover_i_qpos[start_idx:stop_idx, :] = np.repeat(mover_qpos[i : i + 1, :], offset_idx, axis=0)
        mover_j_qpos[start_idx:stop_idx, :] = mover_qpos[i + 1 :, :]
        c_size_arr_i[start_idx:stop_idx, :] = np.repeat(c_size_arr[i : i + 1, :], offset_idx, axis=0)
        c_size_arr_j[start_idx:stop_idx, :] = c_size_arr[i + 1 :, :]

        # Compute distances and update distance table
        distances = np.linalg.norm(mover_qpos[i, :2] - mover_qpos[i + 1 :, :2], axis=1)
        mover_dist_matrix[i, i + 1 :] = distances
        mover_dist_matrix[i + 1 :, i] = distances  # Symmetric

        start_idx = stop_idx

    # flattened distances for i < j
    mover_dist = mover_dist_matrix[np.triu_indices(num_movers, k=1)]

    return mover_dist, mover_i_qpos, mover_j_qpos, c_size_arr_i, c_size_arr_j, mover_dist_matrix

def get_close_mover_pairs(
        mover_qpos: np.ndarray,
        c_size_arr: np.ndarray,
        threshold: float        # without c_size taken into account
    ):
    dists = calculate_mover_distances(mover_qpos, c_size_arr)[5]

    # get all (i, j) pairs where distance < threshold and i < j
    pairs = np.argwhere((dists < threshold) & (np.triu(np.ones_like(dists), k=1) == 1))
    return [tuple(int(i) for i in pair) for pair in pairs], dists
