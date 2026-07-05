"""
reference:
https://github.com/nurlanov-zh/so3_log_map
compute d log(R)_V / dR, shape: 3x9 (or 3x3x3)
"""
import numpy as np
# import sophus_pybind
from typing import Tuple, Union

def vee(mat: np.ndarray) -> np.ndarray:
    """
    matrix shape: ...x3x3
    """
    res = np.array([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]]) # 3x...
    return np.moveaxis(res, 0, -1)

def hat(vec: np.ndarray) -> np.ndarray:
    """
    vector shape: ...x3
    """
    res = np.array([[np.zeros((vec.shape[:-1])), -vec[..., 2], vec[..., 1]],
                    [vec[..., 2], np.zeros((vec.shape[:-1])), -vec[..., 0]],
                    [-vec[..., 1], vec[..., 0], np.zeros((vec.shape[:-1]))]]) # 3x...
    return np.moveaxis(res, [0,1], [-2,-1])

def so3_to_unit_so3(so3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    so3: shape: ...x3x3
    return:
        - unit_so3: shape: ...x3x3
        - theta: shape: ...
    """
    w = vee(so3)
    w_length = np.linalg.norm(w, axis=-1, keepdims=False)
    unit_so3 = so3 / w_length[..., np.newaxis, np.newaxis]
    unit_so3[w_length==0] = hat(np.array([0.,0,1]))
    return unit_so3, w_length

def mat_exp(unit_so3: np.ndarray, theta: Union[float, np.ndarray]) -> np.ndarray:
    """
    so3: shape: ...x3x3
    theta: shape: ...
    NOTE: theta has to be an ndarray, potentially with size 0 (similar to float)
    TODO: offload the normalization to the arguments
    """
    # get the unit so3
    I = np.eye(3).reshape([1]*(len(unit_so3.shape)-2) + [3,3])
    if not isinstance(theta, float):    
        theta = theta[...,np.newaxis,np.newaxis]
    res = I + unit_so3 * np.sin(theta) + np.matmul(unit_so3, unit_so3) * (1 - np.cos(theta))
    return res

def matrix2quaternion(R):    
    # quaternion [q_0, q_{1:3}] = [c, vec]
    q = np.zeros(4)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        # case 1
        t = np.sqrt(1 + t)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[1+i] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[1+j] = (R[j, i] + R[i, j]) * t
        q[1+k] = (R[k, i] + R[i, k]) * t
    return q

def quaternion2vec(q):
    squared_n = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    w = q[0]
    theta = 0
    if squared_n < np.finfo(float).eps * np.finfo(float).eps:
        squared_w = w*w
        two_atan_nbyw_by_n = 2 / w - 2.0 / 3 * squared_n / (w * squared_w) 
        theta = 2 * squared_n / w
    else:
        n = np.sqrt(squared_n)
        if (abs(w) < np.finfo(float).eps):
            if w >= 0:
                two_atan_nbyw_by_n = np.pi / n
            else:
                two_atan_nbyw_by_n = -np.pi / n
        else:
            two_atan_nbyw_by_n = 2 * np.arctan(n / w) / n;
        theta = two_atan_nbyw_by_n * n
            
    tangent = two_atan_nbyw_by_n * q[1:]
    return tangent


def log_quaternion(R):    
    # q = sophus_pybind.SO3.matrix2quaternion(R)
    # v = sophus_pybind.SO3.quaternion2vec(q)
    q = matrix2quaternion(R)
    v = quaternion2vec(q)
    return v


def Dquaternion_DR(R):
    """Computes d quaternion(R) / d R , 4 x 9 Jacobian."""
    J_quat = np.zeros((4, 3, 3))
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        # case 1
        isqrt_t = 1 / np.sqrt(1 + t)
        J_quat[0, :, :] = 0.25 * isqrt_t * np.eye(3)
        J_quat[1, :, :] = -0.25 * isqrt_t / (1 + t) * (R[2, 1] - R[1, 2]) * np.eye(3)
        J_quat[1, 2, 1] = 0.5 * isqrt_t
        J_quat[1, 1, 2] = -0.5 * isqrt_t
        J_quat[2, :, :] = -0.25 * isqrt_t / (1 + t) * (R[0, 2] - R[2, 0]) * np.eye(3)
        J_quat[2, 0, 2] = 0.5 * isqrt_t
        J_quat[2, 2, 0] = -0.5 * isqrt_t
        J_quat[3, :, :] = -0.25 * isqrt_t / (1 + t) * (R[1, 0] - R[0, 1]) * np.eye(3)
        J_quat[3, 1, 0] = 0.5 * isqrt_t
        J_quat[3, 0, 1] = -0.5 * isqrt_t
    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        r = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        i_r = 1 / r
        i_r_cube = 1 / ((R[i, i] - R[j, j] - R[k, k] + 1) * r)
        r_eye = np.eye(3)
        r_eye[j, j] = -1
        r_eye[k, k] = -1
        J_quat[1 + i, :, :] = 0.25 * i_r * r_eye
        
        J_quat[0, :, :] = -0.25 * (R[k, j] - R[j, k]) * i_r_cube * r_eye
        J_quat[0, k, j] = 0.5 * i_r
        J_quat[0, j, k] = -0.5 * i_r
        J_quat[1+j, :, :] = -0.25 * (R[j, i] + R[i, j]) * i_r_cube * r_eye
        J_quat[1+j, j, i] = 0.5 * i_r
        J_quat[1+j, i, j] = 0.5 * i_r
        J_quat[1+k, :, :] = -0.25 * (R[k, i] + R[i, k]) * i_r_cube * r_eye
        J_quat[1+k, k, i] = 0.5 * i_r
        J_quat[1+k, i, k] = 0.5 * i_r
    return J_quat.reshape((4, 9))


def Dlog_Dquaternion(q):
    """Computes d log(q) / d q , 3 x 4 Jacobian."""
    J_vec = np.zeros((3, 4))
    squared_n = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    n = np.sqrt(squared_n)
    w = q[0]
    squared_w = w*w
    if squared_n < np.finfo(float).eps * np.finfo(float).eps:
        # Theta close to 0
        two_atan_nbyw_by_n = 2 / w - 2.0 / 3 * squared_n / (w * squared_w)
        d_q0 = -2 / squared_w + 2 * squared_n / (squared_w * squared_w)
        d_q1 = -4.0 / 3 * q[1] / (w * squared_w)
        d_q2 = -4.0 / 3 * q[2] / (w * squared_w)
        d_q3 = -4.0 / 3 * q[3] / (w * squared_w)
    else:
        if (abs(w) < np.finfo(float).eps):
            # Theta close to pi
            d_q0 = -2 / (squared_w + squared_n)
            if w >= 0:
                # From left
                # 2 * arccos(w), w -> 0
                # arcos'(w) = -1 / (sqrt(1 - w*w)) = |w=0| = -1
                
                two_atan_nbyw_by_n = np.pi / n
                d_q1 = -q[1] * np.pi / (squared_n * n)
                d_q2 = -q[2] * np.pi / (squared_n * n)
                d_q3 = -q[3] * np.pi / (squared_n * n)
            else:
                # From right
                two_atan_nbyw_by_n = -np.pi / n
                d_q1 = q[1] * np.pi / (squared_n * n)
                d_q2 = q[2] * np.pi / (squared_n * n)
                d_q3 = q[3] * np.pi / (squared_n * n)
        else:
            # Regular case
            two_atan_nbyw_by_n = 2 * np.arctan(n / w) / n
            d_q0 = -2 / (squared_w + squared_n)
            c0 = (2 / (squared_n * (w + squared_n/w)) - 2 * np.arctan(n / w) / (squared_n * n))
            d_q1 = c0 * q[1]
            d_q2 = c0 * q[2]
            d_q3 = c0 * q[3]
        
    J_vec[:, 0] = d_q0 * q[1:]
    J_vec[:, 1] = d_q1 * q[1:]
    J_vec[0, 1] += two_atan_nbyw_by_n
    J_vec[:, 2] = d_q2 * q[1:]
    J_vec[1, 2] += two_atan_nbyw_by_n
    J_vec[:, 3] = d_q3 * q[1:]
    J_vec[2, 3] += two_atan_nbyw_by_n
    return J_vec


def Dlog_Dquaternion2(q):
    """Computes d log(q) / d q , 3 x 4 Jacobian."""
    J_vec = np.zeros((3, 4))
    squared_n = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    n = np.sqrt(squared_n)
    w = q[0]
    squared_w = w*w
    sign = 1
    if w < 0:
        sign = -1
        w = -w
    if squared_n < np.finfo(float).eps * np.finfo(float).eps:
        # n (~Theta) close to 0
        two_atan_nbyw_by_n = 2 / w - 2.0 / 3 * squared_n / (w * squared_w)
        d_q0 = -2 / squared_w + 2 * squared_n / (squared_w * squared_w)
        d_q1 = -4.0 / 3 * q[1] / (w * squared_w)
        d_q2 = -4.0 / 3 * q[2] / (w * squared_w)
        d_q3 = -4.0 / 3 * q[3] / (w * squared_w)
    else:
        # Regular case
        two_atan_nbyw_by_n = 4 * np.arctan(n / (w + np.sqrt(squared_w + squared_n))) / n
        d_q0 = -2 / (squared_w + squared_n)
        c0 = (2 * w - two_atan_nbyw_by_n) / squared_n
        d_q1 = c0 * q[1]
        d_q2 = c0 * q[2]
        d_q3 = c0 * q[3]
        
    J_vec[:, 0] = sign * d_q0 * q[1:]
    J_vec[:, 1] = d_q1 * q[1:]
    J_vec[0, 1] += two_atan_nbyw_by_n
    J_vec[:, 2] = d_q2 * q[1:]
    J_vec[1, 2] += two_atan_nbyw_by_n
    J_vec[:, 3] = d_q3 * q[1:]
    J_vec[2, 3] += two_atan_nbyw_by_n
    J_vec = J_vec * sign
    return J_vec


def Dx_log_x_quaternion(R):
    """Computes d log(R)_V / d R , 3 x 9 Jacobian."""
    J_quat = Dquaternion_DR(R)
    # q = sophus_pybind.SO3.matrix2quaternion(R)
    q = matrix2quaternion(R)
    J_vec = Dlog_Dquaternion2(q)
    # assert np.allclose(Dlog_Dquaternion(q), Dlog_Dquaternion2(q))
    return J_vec @ J_quat