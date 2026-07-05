import so3
import se3
import numpy as np
import transformations as tf

def so3_test1():
    # generate 100 random axis-angle vectors
    n_sample = 100
    ws = []
    ws = np.random.random((n_sample, 3))
    ws = ws.reshape((n_sample//5, 5, 3))
    # * test hat
    mat = so3.hat(ws)
    # check for each matrix if it is the cross product matrix
    print('vec shape: ', ws.shape)
    print('mat shape: ', mat.shape)
    for i in range(n_sample//5):
        for j in range(5):
            # check manually if each matrix is the cross product matrix
            w = ws[i, j]
            w_hat = mat[i, j]
            w_hat_manual = np.array([[0, -w[2], w[1]],
                                     [w[2], 0, -w[0]],
                                     [-w[1], w[0], 0]])
            assert np.allclose(w_hat, w_hat_manual)
    # * test vee
    vec = so3.vee(mat)
    print('vec shape: ', vec.shape)
    # check for each vector if it is the cross product vector
    for i in range(n_sample//5):
        for j in range(5):
            # check manually if each matrix is the cross product matrix
            w = vec[i, j]
            w_hat = mat[i, j]
            w_hat_manual = np.array([[0, -w[2], w[1]],
                                     [w[2], 0, -w[0]],
                                     [-w[1], w[0], 0]])
            assert np.allclose(w_hat, w_hat_manual)


def se3_test1():
    # generate 100 random axis-angle vectors
    n_sample = 100
    Vs = []
    Vs = np.random.random((n_sample, 6))
    Vs = Vs.reshape((n_sample//5, 5, 6))
    # * test hat
    mat = se3.hat(Vs)
    # check for each matrix if it is the cross product matrix
    print('vec shape: ', Vs.shape)
    print('mat shape: ', mat.shape)
    for i in range(n_sample//5):
        for j in range(5):
            # check manually if each matrix is the cross product matrix
            V = Vs[i, j]
            V_hat = mat[i, j]
            w = V[3:]
            w_hat = so3.hat(w)
            v = V[:3]
            hat_manual = np.zeros((4,4))
            hat_manual[:3,:3] = np.array([[0, -w[2], w[1]],
                                     [w[2], 0, -w[0]],
                                     [-w[1], w[0], 0]])
            hat_manual[:3,3] = v
            assert np.allclose(V_hat, hat_manual)
    # * test vee
    vec = se3.vee(mat)
    print('vec shape: ', vec.shape)
    # check for each vector if it is the cross product vector
    for i in range(n_sample//5):
        for j in range(5):
            # check manually if each matrix is the cross product matrix
            V = vec[i, j]
            V_hat = mat[i, j]
            w = V[3:]
            w_hat = so3.hat(w)
            v = V[:3]
            hat_manual = np.zeros((4,4))
            hat_manual[:3,:3] = np.array([[0, -w[2], w[1]],
                                     [w[2], 0, -w[0]],
                                     [-w[1], w[0], 0]])
            hat_manual[:3,3] = v
            assert np.allclose(V_hat, hat_manual)

def so3_exp_test1():
    # randomly generate 100 axis-angle vectors
    w = np.random.random((5, 20, 3))
    # 10 of them are of small values
    w[:2,10:,:] = w[:2,10:,:]*1e-5
    w[4,10:,:] = 0
    # generate so3
    so3_mat = so3.hat(w)
    # compute the unit so3
    unit_so3, theta = so3.so3_to_unit_so3(so3_mat)
    print("min theta: ", theta.min())
    print('unit_so3: ', unit_so3[0,10,:])
    print(theta[0,10])
    # check if unit_so3*theta = so3_mat
    assert np.allclose(unit_so3*theta[...,np.newaxis,np.newaxis], so3_mat)
    # compute the exponential
    exp_mat = so3.mat_exp(unit_so3, theta)  # 5x20x3x3
    # check if the exponential is correct
    for i in range(5):
        for j in range(20):
            # transformation: rotate around axis for angle
            size = np.linalg.norm(w[i,j])
            axis = w[i,j] / size
            if size == 0:
                axis = np.array([0,0,1])
            # axis = w[i,j] / size
            angle = size
            assert np.allclose(exp_mat[i,j], tf.rotation_matrix(angle, axis, [0,0,0])[:3,:3])


def so3_exp_test2():
    # randomly generate 1 axis-angle vectors
    # w = np.random.random((3))
    w = np.zeros((3))
    # generate so3
    so3_mat = so3.hat(w)
    # compute the unit so3
    unit_so3, theta = so3.so3_to_unit_so3(so3_mat)
    # check if unit_so3*theta = so3_mat
    assert np.allclose(unit_so3*theta, so3_mat)
    # compute the exponential
    exp_mat = so3.mat_exp(unit_so3, theta)  # 5x20x3x3
    # check if the exponential is correct
    # transformation: rotate around axis for angle
    size = np.linalg.norm(w)
    axis = w / size
    if size == 0:
        axis = np.array([0,0,1])
    # axis = w[i,j] / size
    angle = size
    assert np.allclose(exp_mat, tf.rotation_matrix(angle, axis, [0,0,0])[:3,:3])


def se3_exp_test1():
    # randomly generate 100 screw vectors
    axis = np.random.random((5, 20, 3))
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    point = np.random.random((5, 20, 3))
    angle = np.random.random((5, 20))
    distance = np.random.random((5, 20))
    w = axis * angle[...,np.newaxis]
    v = np.cross(point, angle[...,np.newaxis]*axis) + distance[...,np.newaxis] * axis
    xi = np.concatenate([v, w], axis=-1)
    print(xi.shape)
    # generate se3
    se3_mat = se3.hat(xi)
    # compute the unit se3
    unit_se3, theta = se3.se3_to_unit_se3(se3_mat)
    # check if unit_se3*theta = se3_mat
    assert np.allclose(unit_se3*theta[...,np.newaxis,np.newaxis], se3_mat)
    # check for theta != 0, if unit_se3[:3,:3] is a unit so3
    unit_so3, so3_theta = so3.so3_to_unit_so3(unit_se3[theta!=0,:3,:3])
    assert np.allclose(unit_so3, unit_se3[theta!=0,:3,:3])
    # compute the exponential
    exp_mat = se3.mat_exp(unit_se3, theta)  # 5x20x4x4
    # check if the exponential is correct

    # check orientation
    for i in range(5):
        for j in range(20):
            T = tf.rotation_matrix(angle[i,j], axis[i,j], None)
            R = np.array(T[:3,:3])
            assert np.allclose(exp_mat[i,j,:3,:3], R)
            # TODO: update the computation of the rotation matrix

    for i in range(5):
        for j in range(20):
            T1 = tf.rotation_matrix(angle[i,j], axis[i,j], point[i,j])
            T1[:3,3] += distance[i,j] * axis[i,j]
            T = tf.rotation_matrix(angle[i,j], axis[i,j], None)
            R = np.array(T[:3,:3])
            T[:3,3] = (np.eye(3)-R).dot(point[i,j]) + distance[i,j] * axis[i,j]
            # print('T: ', T)
            # print('exp_mat[i,j]: ', exp_mat[i,j])
            # print('T1 and T difference: ', np.allclose(T1, T))  # T1 = T
            assert np.allclose(exp_mat[i,j], T)
            # TODO: update the computation of the rotation matrix


def se3_exp_test2():
    # randomly generate 100 screw vectors
    axis = np.random.random((3))
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    point = np.random.random((3))
    angle = np.random.random()
    distance = np.random.random()
    w = axis * angle
    v = np.cross(point, angle*axis) + distance * axis
    xi = np.concatenate([v, w], axis=-1)
    print(xi.shape)
    # generate se3
    se3_mat = se3.hat(xi)
    # compute the unit se3
    unit_se3, theta = se3.se3_to_unit_se3(se3_mat)
    print('theta: ', theta)
    print('unit_se3: ', unit_se3)
    # check if unit_se3*theta = se3_mat
    assert np.allclose(unit_se3*theta, se3_mat)
    # check for angle != 0, if unit_se3[:3,:3] is a unit so3
    if angle != 0:
        unit_so3, so3_theta = so3.so3_to_unit_so3(unit_se3[:3,:3])
        assert np.allclose(unit_so3, unit_se3[:3,:3])
    # compute the exponential
    theta2 = float(theta)
    print('before calling se3.matexp')
    exp_mat = se3.mat_exp(unit_se3, theta2)  # 5x20x4x4
    print('after calling se3.matexp')
    # check if the exponential is correct
    # check orientation
    T = tf.rotation_matrix(angle, axis, None)
    R = np.array(T[:3,:3])
    assert np.allclose(exp_mat[:3,:3], R)
    # TODO: update the computation of the rotation matrix

    T1 = tf.rotation_matrix(angle, axis, point)
    T1[:3,3] += distance * axis
    T = tf.rotation_matrix(angle, axis, None)
    R = np.array(T[:3,:3])
    T[:3,3] = (np.eye(3)-R).dot(point) + distance * axis
    # print('T: ', T)
    # print('exp_mat[i,j]: ', exp_mat[i,j])
    # print('T1 and T difference: ', np.allclose(T1, T))  # T1 = T
    assert np.allclose(exp_mat, T)
    # TODO: update the computation of the rotation matrix


def adjoint_test():
    # sample 100 twist
    twist = np.random.random((5, 20, 6))
    pos = np.random.random((5,20,3))
    # generate 100 transformations
    transforms = []
    for i in range(100):
        transform = tf.random_rotation_matrix()
        transforms.append(transform)
    transforms = np.array(transforms).reshape((5,20,4,4))
    transforms[...,:3,3] = pos
    # compute the adjoint
    adjoint = se3.adjoint(transforms)
    # check if the adjoint is correct
    for i in range(5):
        for j in range(20):
            T = transforms[i,j]
            twist_hat = se3.hat(twist[i,j])
            twist2_hat = T@twist_hat@np.linalg.inv(T)
            twist2 = se3.vee(twist2_hat)
            twist2_from_adjoint = adjoint[i,j]@twist[i,j]
            assert np.allclose(twist2, twist2_from_adjoint)

def screw_test():
    # randomly generate 100 screw vectors
    axis = np.random.random((5, 20, 3))
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    point = np.random.random((5, 20, 3))
    angle = np.random.random((5, 20))
    distance = np.random.random((5, 20))
    w = axis * angle[...,np.newaxis]
    v = np.cross(point, angle[...,np.newaxis]*axis) + distance[...,np.newaxis] * axis
    xi = np.concatenate([v, w], axis=-1)
    xi_computed = se3.screw(axis, point, angle, distance)
    assert np.allclose(xi, xi_computed)


def naive_finite_diff_jacobian(func, x, eps=1e-8):
    # given a function and the point for differentiation, compute the naive finite diff
    # func returns array of shape (M,)  #(M can be a list)
    # x is a numpy array of shape (n,)
    # result of shape (M,n)
    # jacobian = np.zeros(list(func(x).shape)+[len(x)])
    # for i in range(len(x)):
    #     x_plus = x.copy()
    #     x_plus[i] += eps
    #     x_minus = x.copy()
    #     x_minus[i] -= eps
    #     jacobian[...,i] = (func(x_plus) - func(x_minus)) / (2*eps)
    func_shape = list(func(x).shape)
    x_shape = list(x.shape)
    jacobian = np.zeros(list(func(x).shape)+list(x.shape)).reshape((np.prod(func_shape), np.prod(x_shape)))  # first flatten the func shape
    for i in range(np.prod(x_shape)):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        x_minus = x.copy()
        x_minus.flat[i] -= eps
        
        deriv = (func(x_plus) - func(x_minus)) / (2*eps)
        jacobian[:,i] = deriv.flatten()
    jacobian = jacobian.reshape(func_shape+x_shape)
    return jacobian

def test_naive_finite_diff_jacobian():
    # test the naive finite diff jacobian
    def func(x):
        return np.array([x[0]*x[1], x[0]**2])
    # x = np.array([2.0,3])
    x = np.random.random((2))
    jacobian = naive_finite_diff_jacobian(func, x)
    print(jacobian)
    jacobian_analytical = np.array([[x[1], x[0]], [2*x[0], 0]])
    assert np.allclose(jacobian, jacobian_analytical)
# TODO: test the log function compare with the axis angle method

def so3_log_test1():
    R = tf.random_rotation_matrix()
    angle, direct, _ = tf.rotation_from_matrix(R)
    w = angle * np.array(direct)
    w_so3 = so3.log_quaternion(R)
    print('w: ', w)
    print('w_so3: ', w_so3)
    assert np.allclose(w, w_so3)


def so3_jacobian_test1():
    # test the function Dx_log_x_quaternion
    # randomly sample a rotation matrix
    R = tf.random_rotation_matrix()[:3,:3]
    jac = so3.Dx_log_x_quaternion(R).reshape((3,3,3))  # d [log(R)]_V / dR
    # compute the finite diff
    def vee_log(R):
        # obtain the twist
        T = np.eye(4)
        T[:3,:3] = R
        # NOTE: when R is invalid (not a rotation matrix), then this has problems
        angle, direct, point = tf.rotation_from_matrix(T)
        return np.array(direct)*angle
    finite_diff = naive_finite_diff_jacobian(vee_log, R)
    print('jac: ')
    print(jac)
    print('finite_diff: ')
    print(finite_diff)
    print('diff: ')
    print(np.linalg.norm(jac-finite_diff))

def so3_jacobian_test2():
    """
    perturb R to be closest valid rotation, R'=perturb(R)
    then according to Taylor expansion:
    log_V(R') = log_V(R) + d log_V(R)/dR * (R'-R)
    verify this
    """
    # randomly sample a rotation matrix
    R = tf.random_rotation_matrix()[:3,:3]
    # perturb R
    w = np.random.random((3))
    w = w / np.linalg.norm(w)
    angle = 1e-5*np.pi/180
    purt_R = tf.rotation_matrix(angle, w)[:3,:3]
    R_prime = purt_R @ R
    # compute the log_V for R
    T = np.eye(4)
    T[:3,:3] = R
    angle, direct, _ = tf.rotation_from_matrix(T)
    w = angle * np.array(direct)

    # compute the log_V for R_prime
    T_prime = np.eye(4)
    T_prime[:3,:3] = R_prime
    angle_prime, direct_prime, _ = tf.rotation_from_matrix(T_prime)
    w_prime = angle_prime * np.array(direct_prime)

    # compute the difference
    delta_w = w_prime - w
    # compute the first-order approximation
    jac = so3.Dx_log_x_quaternion(R).reshape((3,3,3))  # d [log(R)]_V / dR
    delta_w_approx = np.tensordot(jac, R_prime-R, axes=2)
    # compare the difference
    print('delta_w: ', delta_w)
    print('delta_w_approx: ', delta_w_approx)
    print('diff: ', np.linalg.norm(delta_w-delta_w_approx))
    print('diff: ', np.linalg.norm(delta_w-delta_w_approx)/np.linalg.norm(delta_w))
    print('diff/theta: ', np.linalg.norm(delta_w-delta_w_approx)/angle)

    # # seems to be close
    # R_diff = R_prime - R
    # R_diff_inv = np.linalg.inv(R_diff)
    # print('delta_w dot R_diff_inv: ', delta_w.dot(R_diff_inv))
    # print('jac: ')
    # print(jac)

if __name__ == "__main__":
    so3_test1()
    print('so3 test 1 passed')
    se3_test1()
    print('se3 test 1 passed')
    so3_exp_test1()
    print('so3 exp test 1 passed')
    so3_exp_test2()
    print('so3 exp test 2 passed')
    se3_exp_test1()
    print('se3 exp test 1 passed')
    se3_exp_test2()
    print('se3 exp test 2 passed')
    adjoint_test()
    print('adjoint test passed')
    screw_test()
    print('screw test passed')
    test_naive_finite_diff_jacobian()
    print('naive finite diff jacobian test passed')
    so3_log_test1()
    print('so3 log test 1 passed')
    so3_jacobian_test1()
    print('so3 jacobian test 1 passed')
    so3_jacobian_test2()
    print('so3 jacobian test 2 passed')