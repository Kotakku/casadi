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
Quadrotor trajectory optimization on R^3 x R^3 x SO(3) x R^3 with fatrop

Same problem as manifold_quadrotor_opti.py, formulated for fatrop: variables
and constraints are declared in stage order (x_0, u_0, x_1, ...; dynamics of
stage k, then the path constraints of stage k). fatrop works in the tangent
space (12 states per stage instead of 13) and updates the iterate with the
retraction, so quaternions stay normalized.
"""
import sys
import casadi as ca
import numpy as np

if not ca.has_nlpsol("fatrop"):
    print("fatrop is not available")
    sys.exit(0)

R3 = ca.Manifold.Euclidean(3)
SO3 = ca.Manifold.SO3()
M = ca.Manifold.product([R3, R3, SO3, R3])   # p, v, q, omega

N = 20
dt = 0.1


def step(x, u):
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    acc = SO3.act(q, ca.vertcat(0, 0, u[0])) + ca.DM([0, 0, -9.81])
    return ca.vertcat(p + v*dt, v + acc*dt, SO3.retract(q, w*dt), w + u[1:4]*dt)


x_init = ca.vertcat(ca.DM.zeros(6), SO3.exp(ca.DM([0.3, 0, 0])), ca.DM.zeros(3))
x_goal = ca.vertcat(ca.DM([1, 1, 1]), ca.DM.zeros(3), SO3.exp(ca.DM([0, 0, 1.0])), ca.DM.zeros(3))

opti = ca.Opti()
X, U = [], []
for k in range(N+1):
    X.append(opti.variable(M))
    if k < N:
        U.append(opti.variable(4))

J = 0
for k in range(N):
    opti.subject_to(M.minus(X[k+1], step(X[k], U[k])) == 0)   # dynamics of stage k
    if k == 0:
        opti.subject_to(M.minus(X[0], x_init) == 0)          # path constraints of stage 0
    opti.subject_to(opti.bounded(0, U[k][0], 20))
    opti.subject_to(opti.bounded(-2, U[k][1:], 2))
    opti.set_initial(U[k][0], 9.81)
    J += ca.sumsqr(U[k][1:])*dt + 0.01*ca.sumsqr(U[k][0] - 9.81)*dt
opti.minimize(J + 100*ca.sumsqr(M.minus(x_goal, X[N])))

opti.solver("fatrop", {"structure_detection": "auto", "expand": True,
                       "fatrop": {"print_level": 0}})
sol = opti.solve()

stats = sol.stats()
q = np.hstack([np.array(sol.value(x)).reshape(-1, 1)[6:10] for x in X])
print("iterations:", stats["iter_count"])
print("tangent state dimensions per stage:", stats["nx"])
print("final position:", sol.value(X[N])[0:3])
print("quaternion norms in [%.15f, %.15f]" % (np.linalg.norm(q, axis=0).min(),
                                               np.linalg.norm(q, axis=0).max()))
