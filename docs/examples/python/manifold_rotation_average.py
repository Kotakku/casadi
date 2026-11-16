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
Rotation averaging on SO(3) with casadi.Manifold

Given noisy measurements q_i of an orientation (unit quaternions), find the
orientation q that minimizes sum_i |minus(q_i, q)|^2. The optimization runs
in the 3-dimensional tangent space: every Gauss-Newton step dq is computed
there and applied with q <- retract(q, dq), so q stays a unit quaternion
without any normalization constraint.
"""
import casadi as ca
import numpy as np

np.random.seed(0)
SO3 = ca.Manifold.SO3()

# Noisy measurements around a true orientation
q_true = SO3.exp(ca.DM([0.3, -0.5, 1.2]))
N = 20
noise = 0.1 * np.random.randn(3, N)
Q = SO3.retract(q_true, ca.DM(noise))   # 4-by-N, one quaternion per column

# Residuals as a function of a tangent perturbation dq of the estimate q
q = ca.SX.sym("q", 4)
dq = ca.SX.sym("dq", 3)
r = ca.vec(SO3.minus(ca.DM(Q), SO3.retract(q, dq)))   # 3*N residuals
J = ca.jacobian(r, dq)
gauss_newton = ca.Function("gauss_newton", [q, dq], [r, J])

# Gauss-Newton iterations, starting from the identity
q_est = SO3.identity()
for it in range(10):
    r_k, J_k = gauss_newton(q_est, ca.DM.zeros(3))
    step = -ca.solve(ca.mtimes(J_k.T, J_k), ca.mtimes(J_k.T, r_k))
    q_est = SO3.retract(q_est, step)
    print("iter %d: cost %.6e, |step| %.2e" % (it, float(ca.sumsqr(r_k)), float(ca.norm_2(step))))
    if float(ca.norm_2(step)) < 1e-12:
        break

print("estimate:          ", q_est.T)
print("unit norm:         ", float(ca.norm_2(q_est)))
print("error to truth [rad]:", float(ca.norm_2(SO3.minus(q_est, q_true))))
print("closed-form check: mean of minus(q_i, q_est) =",
      ca.sum2(SO3.minus(ca.DM(Q), q_est)).T / N)
