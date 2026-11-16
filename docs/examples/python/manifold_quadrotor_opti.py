#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
"""
Quadrotor trajectory optimization with an orientation on SO(3)

The state (position, velocity, orientation, angular velocity) lives on the
manifold R^3 x R^3 x SO(3) x R^3, with the orientation stored as a unit
quaternion. Declaring the state with opti.variable(M, N+1) lets sqpmethod
optimize in the tangent space: quaternions stay normalized without any
unit-norm constraint. Dynamics and boundary conditions are written in the
tangent space with M.minus.
"""
import casadi as ca
import numpy as np

R3 = ca.Manifold.Euclidean(3)
SO3 = ca.Manifold.SO3()
M = ca.Manifold.product([R3, R3, SO3, R3])   # p, v, q, omega: nx = 13, ndx = 12

N = 20       # control intervals
dt = 0.1     # time step [s]
mass = 1.0
grav = ca.DM([0, 0, -9.81])


def step(x, u):
    """Semi-implicit Euler step; the orientation is propagated with the retraction"""
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    thrust, torque = u[0], u[1:4]
    acc = SO3.act(q, ca.vertcat(0, 0, thrust/mass)) + grav
    return ca.vertcat(p + v*dt, v + acc*dt, SO3.retract(q, w*dt), w + torque*dt)


opti = ca.Opti()
X = opti.variable(M, N+1)   # one manifold element per column
U = opti.variable(4, N)     # thrust and body torques (Euclidean)

x_init = ca.vertcat(ca.DM.zeros(6), SO3.exp(ca.DM([0.3, 0, 0])), ca.DM.zeros(3))
x_goal = ca.vertcat(ca.DM([1, 1, 1]), ca.DM.zeros(3), SO3.exp(ca.DM([0, 0, 1.0])), ca.DM.zeros(3))

opti.subject_to(M.minus(X[:, 0], x_init) == 0)
for k in range(N):
    opti.subject_to(M.minus(X[:, k+1], step(X[:, k], U[:, k])) == 0)
opti.subject_to(opti.bounded(0, U[0, :], 20))
opti.subject_to(opti.bounded(-2, U[1:, :], 2))

opti.minimize(ca.sumsqr(U[1:, :])*dt + 0.01*ca.sumsqr(U[0, :] - 9.81*mass)*dt
              + 100*ca.sumsqr(M.minus(x_goal, X[:, N])))
opti.set_initial(U[0, :], 9.81*mass)

opti.solver("sqpmethod", {"qpsol": "qrqp", "print_time": False,
                          "qpsol_options": {"print_iter": False, "print_header": False,
                                            "error_on_fail": False}})
sol = opti.solve()

Xv = sol.value(X)
print("iterations:", sol.stats()["iter_count"])
print("final position:", Xv[0:3, -1])
print("orientation error to goal [rad]:",
      float(ca.norm_2(SO3.minus(ca.DM(Xv[6:10, -1]), x_goal[6:10]))))
print("quaternion norms in [%.15f, %.15f]" % (np.linalg.norm(Xv[6:10, :], axis=0).min(),
                                               np.linalg.norm(Xv[6:10, :], axis=0).max()))
