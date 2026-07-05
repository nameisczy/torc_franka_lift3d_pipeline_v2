import numpy as np
import so3
from typing import Tuple, Union


def vee(mat: np.ndarray) -> np.ndarray:
    """
    matrix shape: ...x4x4
    ([[w_hat, v],
      [0,     0]])_V = [v, w]
    """
    rot = so3.vee(mat)
    pos = mat[..., :3, 3]
    res = np.concatenate([pos, rot], axis=-1)
    return res


def hat(vec: np.ndarray) -> np.ndarray:
    """
    vector shape: ...x6
    [v,w])_hat = ([[w_hat, v],
                   [0,     0]])
    """
    rot = so3.hat(vec[...,3:])
    pos = vec[..., :3]
    res = np.concatenate([np.concatenate([rot, pos[..., np.newaxis]], axis=-1),
                          np.zeros(vec.shape[:-1] + (1, 4))], axis=-2)
    return res


def screw(axis: np.ndarray, pt: np.ndarray, angle: Union[float, np.ndarray], distance: Union[float, np.ndarray]) -> np.ndarray:
    """
    axis: rotation axis, shape: ...,3
    pt: a point on the axis, shape: ...,3
    angle: rotation angle, shape: ...
    distance: ditance for the screw motion, shape: ...
    result: [pt cross axis + linear_v, axis]
    """
    if isinstance(angle, float):
        res = np.cross(pt, axis*angle) + axis*distance
        return np.concatenate([res, axis*angle])
    res = np.cross(pt, axis*angle[...,np.newaxis]) + axis*distance[...,np.newaxis]
    return np.concatenate([res, axis*angle[...,np.newaxis]], axis=-1)

def se3_to_unit_se3(se3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    se3: shape: ...x4x4
    return:
        - unit_se3: shape: ...x4x4
        - theta: shape: ...
    convert se3 to [v,w]. Check if w is zero. If so, then unit_w=0, and normalize v.
    Otherwise, normalize w.
    """
    xi = vee(se3)
    w = xi[..., 3:]
    v = xi[..., :3]
    w_length = np.linalg.norm(w, axis=-1, keepdims=False)
    v_length = np.linalg.norm(v, axis=-1, keepdims=False)
    length = np.array(w_length)
    # unit_w = w / w_length[..., np.newaxis]
    # unit_v = v / v_length[..., np.newaxis]
    # unit_w[w_length==0] = 0
    # unit_v[w_length==0] = v[w_length==0] / v_length[w_length==0][..., np.newaxis]
    length[w_length==0] = v_length[w_length==0]
    # unit_xi = np.concatenate([unit_v, unit_w], axis=-1)
    # xi_length = w_length
    # xi_length[w_length==0] = v_length[w_length==0]
    unit_se3 = se3 / length[..., np.newaxis, np.newaxis]
    return unit_se3, length

def mat_exp(unit_se3: np.ndarray, theta: Union[float, np.ndarray]) -> np.ndarray:
    """
    unit_se3: shape: ...x4x4, when w = 0, it is [I, v*theta; 0, 1], otherwise ||w||=1
    theta: shape: ...
    vee(se3) = [v,w] ([linear, angular])
    when w = 0, it is [I, v*theta; 0, 1]
    when w != 0, it is [R, (I - R)(w cross v) + w w^T v*theta; 0, 1]
    where w is of unit length
    """
    # * when unit_se3 of shape 4x4 and theta is a float
    if isinstance(theta, float):
        unit_xi = vee(unit_se3)
        unit_w = unit_xi[3:]
        unit_v = unit_xi[:3]
        w_length = np.linalg.norm(unit_w)
        if w_length == 0:
            return np.array([[1, 0, 0, unit_v[0]*theta],
                             [0, 1, 0, unit_v[1]*theta],
                             [0, 0, 1, unit_v[2]*theta],
                             [0, 0, 0, 1]])
        else:
            R = so3.mat_exp(unit_se3[:3, :3], theta)
            I = np.eye(3)
            res = np.array([[R[0, 0], R[0, 1], R[0, 2], 0],
                             [R[1, 0], R[1, 1], R[1, 2], 0],
                             [R[2, 0], R[2, 1], R[2, 2], 0],
                             [0, 0, 0, 1]])
            res[:3,3] = (I-R).dot(np.cross(unit_w, unit_v)) + np.outer(unit_w, unit_w).dot(unit_v)*theta
            return res
    # * when unit_se3 of shape ...x4x4 and theta is an array
    # obtain where w = 0
    unit_xi = vee(unit_se3)
    unit_w = unit_xi[..., 3:]
    unit_v = unit_xi[..., :3]
    w_length = np.linalg.norm(unit_w, axis=-1, keepdims=False)
    res = np.zeros((unit_se3.shape[:-2] + (4, 4)))
    res[w_length==0, :3, :3] = np.eye(3)
    res[w_length==0, :3, 3] = unit_v[w_length==0] * theta[w_length==0, np.newaxis]  # since we haven't normalized v, just use the original v
    R = so3.mat_exp(unit_se3[w_length!=0, :3, :3], theta[w_length!=0])
    I = np.eye(3).reshape([1]*(len(R.shape)-2) + [3,3])
    res[w_length!=0, :3, :3] = R
    unit_w_selected = unit_w[w_length!=0]
    unit_v_selected = unit_v[w_length!=0]
    theta_selected = theta[w_length!=0,np.newaxis]
    unit_w_selected_reshaped_1 = unit_w_selected[..., np.newaxis]
    unit_w_selected_reshaped_2 = unit_w_selected[..., np.newaxis, :]

    term1 = I-R  # ..x3x3
    term2 = np.cross(unit_w_selected, unit_v_selected)[...,np.newaxis] # ..x3x1
    term3 = np.matmul(unit_w_selected_reshaped_1, unit_w_selected_reshaped_2) # ..x3x3
    term4 = unit_v_selected*theta_selected
    term4 = term4[..., np.newaxis] # ..x3x1
    term5 = np.matmul(term3, term4) # ..x3x1
    res[w_length!=0, :3, 3] = (np.matmul(term1, term2) + term5)[...,0]
    res[...,3,3] = 1  # transformation matrix has 1 at the end
    # res[w_length!=0, :3, 3] = np.matmul(I-R, np.cross(unit_w_selected, unit_v_selected)[...,np.newaxis]) + \
    #                           np.matmul(np.matmul(unit_w_selected_reshaped_1, unit_w_selected_reshaped_2),unit_v_selected*theta_selected)  
    return res

def adjoint(transform: np.ndarray) -> np.ndarray:
    """
    transform: shape: ...x4x4
    return: adjoint matrix, shape: ...x6x6
    """
    rot = transform[..., :3, :3]
    pos = transform[..., :3, 3]
    res = np.zeros(transform.shape[:-2] + (6, 6))
    res[..., :3, :3] = rot
    res[..., 3:, 3:] = rot
    res[..., :3, 3:] = np.matmul(so3.hat(pos), rot)
    return res