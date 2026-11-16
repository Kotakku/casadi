#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             KU Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
import casadi as ca
import numpy as np
import os
import unittest
from helpers import casadiTestCase

inf = float("inf")


def attitude_problem():
  # type: () -> tuple[ca.Manifold, dict, ca.DM]
  """min |minus(q_target, q)|^2 over unit quaternions q"""
  SO3 = ca.Manifold.SO3()
  q = ca.MX.sym("q", 4)
  q_target = SO3.exp(ca.DM([0.3, -0.2, 0.8]))
  return SO3, {"x": q, "f": ca.sumsqr(SO3.minus(q_target, q))}, q_target


def pose_problem():
  # type: () -> tuple[ca.Manifold, dict, ca.DM]
  """Euclidean position with a bound and a constraint, plus an orientation"""
  SO3 = ca.Manifold.SO3()
  M = ca.Manifold.product([ca.Manifold.Euclidean(2), SO3])
  x = ca.MX.sym("x", 6)
  q_target = SO3.exp(ca.DM([0.1, 0.5, -0.4]))
  f = ca.sumsqr(x[:2] - 3) + ca.sumsqr(SO3.minus(q_target, x[2:]))
  return M, {"x": x, "f": f, "g": x[0] + x[1]}, q_target


class ManifoldNlpsolCommonTests(casadiTestCase):
  """Option x_manifold in the Nlpsol base class"""

  def setUp(self):
    if not ca.has_nlpsol("ipopt"):
      self.skipTest("ipopt not available")
    self.opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}

  def solver(self, nlp, M, **opts):
    # type: (dict, ca.Manifold, ...) -> ca.Function
    o = dict(self.opts)
    o.update({"x_manifold": M, "manifold_fallback": "embed"})
    o.update(opts)
    return ca.nlpsol("S", "ipopt", nlp, o)

  def test_option_type(self):
    self.assertEqual(ca.nlpsol_option_type("ipopt", "x_manifold"), "OT_MANIFOLD")
    self.assertEqual(ca.nlpsol_option_type("ipopt", "manifold_fallback"), "OT_STRING")

  def test_no_native_support(self):
    M, nlp, _ = attitude_problem()
    with self.assertRaisesRegex(Exception, "no native support"):
      ca.nlpsol("S", "ipopt", nlp, {"x_manifold": M})
    with self.assertRaisesRegex(Exception, "no native support"):
      ca.nlpsol("S", "ipopt", nlp, {"x_manifold": M, "manifold_fallback": "error"})
    with self.assertRaisesRegex(Exception, "manifold_fallback"):
      ca.nlpsol("S", "ipopt", nlp, {"x_manifold": M, "manifold_fallback": "foo"})

  def test_dimension_mismatch(self):
    _, nlp, _ = attitude_problem()
    for fb in ["error", "embed"]:
      with self.assertRaisesRegex(Exception, "nx = 7"):
        ca.nlpsol("S", "ipopt", nlp, {"x_manifold": ca.Manifold.SE3(), "manifold_fallback": fb})

  def test_embed(self):
    M, nlp, q_target = attitude_problem()
    S = self.solver(nlp, M)
    # the embedding constraint is hidden from the user
    self.assertEqual(S.size_in("lbg"), (0, 1))
    self.assertEqual(S.size_out("lam_g"), (0, 1))
    r = S(x0=[1, 0.1, 0, 0])
    self.assertTrue(S.stats()["success"])
    self.checkarray(M.minus(r["x"], q_target), ca.DM.zeros(3, 1), digits=6)
    self.checkarray(M.constraint(r["x"]), ca.DM(0), digits=8)
    self.checkarray(r["lam_x"], ca.DM.zeros(4, 1))

  def test_embed_product(self):
    M, nlp, q_target = pose_problem()
    S = self.solver(nlp, M)
    self.assertEqual(S.size_in("lbg"), (1, 1))
    r = S(x0=[0, 0, 1, 0, 0, 0], ubx=[2.5, inf, inf, inf, inf, inf], lbg=-10, ubg=5)
    self.checkarray(r["x"][:2], ca.DM([2.5, 2.5]), digits=4)
    self.checkarray(r["g"], ca.DM(5), digits=4)
    self.assertTrue(float(r["lam_g"]) > 0)
    self.assertTrue(float(r["lam_x"][0]) > 0)
    self.checkarray(r["lam_x"][2:], ca.DM.zeros(4, 1))
    self.checkarray(M.component(1).minus(r["x"][2:], q_target), ca.DM.zeros(3, 1), digits=6)

  def test_bounds_on_manifold(self):
    M, nlp, _ = pose_problem()
    S = self.solver(nlp, M)
    # bounds on Euclidean entries are fine
    S(x0=[0, 0, 1, 0, 0, 0], lbx=[-1, -1, -inf, -inf, -inf, -inf])
    with self.assertRaisesRegex(Exception, "only allowed on the Euclidean entries"):
      S(x0=[0, 0, 1, 0, 0, 0], ubx=[inf, inf, 2, inf, inf, inf])

  def test_initial_guess_projection(self):
    M, nlp, q_target = attitude_problem()
    S = self.solver(nlp, M)
    # x0 off the manifold is projected
    r = S(x0=[2, 0.2, 0, 0])
    self.checkarray(M.minus(r["x"], q_target), ca.DM.zeros(3, 1), digits=6)
    # x0 that cannot be projected
    with self.assertRaisesRegex(Exception, "cannot be projected"):
      S(x0=[0, 0, 0, 0])

  def test_equality_option(self):
    M, nlp, _ = pose_problem()
    S = self.solver(nlp, M, equality=[False])
    S(x0=[0, 0, 1, 0, 0, 0], lbg=-10, ubg=5)
    with self.assertRaisesRegex(Exception, "equality"):
      self.solver(nlp, M, equality=[False, True])

  def test_incompatible_options(self):
    M, nlp, _ = pose_problem()
    with self.assertRaisesRegex(Exception, "detect_simple_bounds"):
      self.solver(nlp, M, detect_simple_bounds=True)
    S = self.solver(nlp, M)
    with self.assertRaisesRegex(Exception, "sensitivities"):
      S.forward(1)

  def test_codegen(self):
    M, nlp, _ = pose_problem()
    S = self.solver(nlp, M)
    args = dict(x0=[0, 0, 1, 0, 0, 0], ubx=[2.5, inf, inf, inf, inf, inf], lbg=-10, ubg=5)
    self.check_codegen(S, args, std="c99", extralibs=["ipopt"], digits=8)

  def test_serialize(self):
    M, nlp, q_target = pose_problem()
    S = self.solver(nlp, M)
    S2 = ca.Function.deserialize(S.serialize())
    self.assertEqual(S2.size_in("lbg"), (1, 1))
    args = dict(x0=[0, 0, 1, 0, 0, 0], ubx=[2.5, inf, inf, inf, inf, inf], lbg=-10, ubg=5)
    r, r2 = S(**args), S2(**args)
    for k in ["x", "g", "lam_g", "lam_x"]:
      self.checkarray(r[k], r2[k], digits=8)
    with self.assertRaisesRegex(Exception, "only allowed on the Euclidean entries"):
      S2(x0=[0, 0, 1, 0, 0, 0], ubx=[inf, inf, 2, inf, inf, inf])


def attitude_ocp(N=10, dt=0.1, native=True):
  # type: (int, float, bool) -> tuple[dict, ca.Manifold, dict]
  """Rest-to-rest-free attitude maneuver: state (q, omega) in SO3 x R3, torque in R3"""
  SO3 = ca.Manifold.SO3()
  R3 = ca.Manifold.Euclidean(3)
  Mx = ca.Manifold.product([SO3, R3])
  q_init = SO3.exp(ca.DM([0.5, 0, 0.3]))
  q_goal = SO3.exp(ca.DM([1.0, -2.0, 0.5]))
  X = [ca.SX.sym("x%d" % k, 7) for k in range(N+1)]
  U = [ca.SX.sym("u%d" % k, 3) for k in range(N)]
  w, mans, g = [], [], []
  J = 0
  for k in range(N+1):
    w.append(X[k])
    mans.append(Mx)
    if k < N:
      w.append(U[k])
      mans.append(R3)
  g.append(SO3.minus(X[0][:4], q_init))
  g.append(X[0][4:] - ca.DM([0, 2, 0]))
  for k in range(N):
    q, om = X[k][:4], X[k][4:]
    g.append(SO3.minus(X[k+1][:4], SO3.retract(q, om*dt)))
    g.append(X[k+1][4:] - (om + U[k]*dt))
    J += ca.sumsqr(U[k])*dt
  J += 100*ca.sumsqr(SO3.minus(q_goal, X[N][:4])) + 10*ca.sumsqr(X[N][4:])
  if not native:
    g += [ca.sumsqr(X[k][:4]) - 1 for k in range(N+1)]
  x0 = ca.vertcat(*[ca.vertcat(q_init, ca.DM.zeros(3)) if i % 2 == 0 else ca.DM.zeros(3)
                    for i in range(2*N+1)])
  lbx = ca.vertcat(*[ca.DM([-inf]*7) if i % 2 == 0 else ca.DM([-3]*3) for i in range(2*N+1)])
  args = {"x0": x0, "lbx": lbx, "ubx": -lbx, "lbg": 0, "ubg": 0}
  return {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}, ca.Manifold.product(mans), args


SQP_OPTS = {"qpsol": "qrqp", "print_header": False, "print_iteration": False,
            "print_time": False, "print_status": False, "max_iter": 200,
            "qpsol_options": {"print_iter": False, "print_header": False, "error_on_fail": False}}


class ManifoldSqpmethodTests(casadiTestCase):
  """Native support for x_manifold in sqpmethod"""

  def solver(self, nlp, M, **opts):
    # type: (dict, ca.Manifold, ...) -> ca.Function
    o = dict(SQP_OPTS)
    o["x_manifold"] = M
    o.update(opts)
    return ca.nlpsol("S", "sqpmethod", nlp, o)

  def test_attitude(self):
    M, nlp, q_target = attitude_problem()
    for X in [ca.SX, ca.MX]:
      q = X.sym("q", 4)
      nlp_x = {"x": q, "f": ca.sumsqr(M.minus(q_target, q))}
      for hess in ["exact", "limited-memory"]:
        S = self.solver(nlp_x, M, hessian_approximation=hess)
        r = S(x0=[1, 0, 0, 0])
        self.assertTrue(S.stats()["success"])
        self.checkarray(M.minus(r["x"], q_target), ca.DM.zeros(3, 1), digits=8)
        # the iterate stays on the manifold
        self.checkarray(ca.norm_2(r["x"]), ca.DM(1), digits=12)

  def test_bounds_and_constraints(self):
    M, nlp, q_target = pose_problem()
    S = self.solver(nlp, M)
    r = S(x0=[0, 0, 1, 0, 0, 0], ubx=[2.5, inf, inf, inf, inf, inf], lbg=-10, ubg=5)
    self.assertTrue(S.stats()["success"])
    self.checkarray(r["x"][:2], ca.DM([2.5, 2.5]), digits=8)
    self.checkarray(r["g"], ca.DM(5), digits=8)
    self.checkarray(r["lam_x"][2:], ca.DM.zeros(4, 1))
    self.checkarray(M.component(1).minus(r["x"][2:], q_target), ca.DM.zeros(3, 1), digits=8)
    with self.assertRaisesRegex(Exception, "only allowed on the Euclidean entries"):
      S(x0=[0, 0, 1, 0, 0, 0], ubx=[inf, inf, 2, inf, inf, inf])

  def test_ocp(self):
    nlp, M, args = attitude_ocp()
    S = self.solver(nlp, M, tol_pr=1e-10, tol_du=1e-10)
    r = S(**args)
    self.assertTrue(S.stats()["success"])
    x = r["x"]
    # all quaternions are unit quaternions
    for k in range(11):
      self.checkarray(ca.norm_2(x[10*k:10*k+4]), ca.DM(1), digits=10)
    if ca.has_nlpsol("ipopt"):
      Si = ca.nlpsol("S", "ipopt", nlp, {"x_manifold": M, "manifold_fallback": "embed",
                                         "ipopt.print_level": 0, "print_time": False,
                                         "ipopt.sb": "yes", "ipopt.tol": 1e-10})
      ri = Si(**args)
      self.checkarray(r["x"], ri["x"], digits=5)
      self.checkarray(r["f"], ri["f"], digits=4)
    # Same optimum as sqpmethod on the formulation with explicit unit-norm constraints
    nlp_e, _, _ = attitude_ocp(native=False)
    S_e = ca.nlpsol("S", "sqpmethod", nlp_e, dict(SQP_OPTS, tol_pr=1e-10, tol_du=1e-10))
    r_e = S_e(**args)
    if S_e.stats()["success"]:
      self.checkarray(r["f"], r_e["f"], digits=4)

  def test_options(self):
    nlp, M, args = attitude_ocp(N=6)
    ref = None
    for opts in [{}, {"max_iter_ls": 0}, {"second_order_corrections": True},
                 {"elastic_mode": True}, {"max_iter_ls": 0, "second_order_corrections": True}]:
      S = self.solver(nlp, M, **opts)
      r = S(**args)
      self.assertTrue(S.stats()["success"], str(opts))
      if ref is None:
        ref = r["x"]
      self.checkarray(r["x"], ref, str(opts), digits=5)

  def test_serialize(self):
    nlp, M, args = attitude_ocp(N=6)
    S = self.solver(nlp, M)
    S2 = ca.Function.deserialize(S.serialize())
    r, r2 = S(**args), S2(**args)
    self.assertTrue(S2.stats()["success"])
    self.checkarray(r["x"], r2["x"], digits=12)

  def test_nested_product(self):
    # Euclidean entries inside nested products keep their bounds
    SO3 = ca.Manifold.SO3()
    M = ca.Manifold.product([ca.Manifold.product([SO3, ca.Manifold.Euclidean(2)]),
                             ca.Manifold.Euclidean(1)])
    x = ca.SX.sym("x", 7)
    q_target = SO3.exp(ca.DM([0.2, 0.1, -0.3]))
    f = ca.sumsqr(SO3.minus(q_target, x[:4])) + ca.sumsqr(x[4:] - ca.DM([1, 2, 3]))
    S = self.solver({"x": x, "f": f}, M)
    r = S(x0=[1, 0, 0, 0, 0, 0, 0], lbx=[-inf]*4 + [-inf, -inf, -inf], ubx=[inf]*4 + [0.5, inf, 2])
    self.checkarray(r["x"][4:], ca.DM([0.5, 2, 2]), digits=8)
    self.checkarray(SO3.minus(r["x"][:4], q_target), ca.DM.zeros(3, 1), digits=8)

  def test_sphere(self):
    from manifold import sphere_functions
    retract, minus, project = sphere_functions()
    S2 = ca.Manifold("S2", retract, minus, {"project": project})
    x = ca.SX.sym("x", 3)
    a = ca.DM([1, -2, 2])
    S = self.solver({"x": x, "f": -ca.dot(a, x)}, S2)
    r = S(x0=[0.1, 0, 1])
    self.assertTrue(S.stats()["success"])
    self.checkarray(r["x"], a/3, digits=8)

  def test_partially_euclidean(self):
    x = ca.SX.sym("x", 2)
    y = ca.SX.sym("y", 2)
    dx = ca.SX.sym("dx", 2)
    M = ca.Manifold("half", ca.Function("r", [x, dx], [x + dx]),
                    ca.Function("m", [y, x], [y - x]), {"euclidean_mask": [True, False]})
    with self.assertRaisesRegex(Exception, "neither Euclidean nor free of Euclidean"):
      self.solver({"x": x, "f": ca.sumsqr(x)}, M)


def quadrotor(opti, N=10, dt=0.1):
  # type: (ca.Opti, int, float) -> tuple[ca.MX, ca.MX, ca.Manifold]
  """Quadrotor with state (p, v, q, omega) in R3 x R3 x SO3 x R3 and input (thrust, torque)"""
  R3 = ca.Manifold.Euclidean(3)
  SO3 = ca.Manifold.SO3()
  Mx = ca.Manifold.product([R3, R3, SO3, R3])
  X = opti.variable(Mx, N+1)
  U = opti.variable(4, N)

  def F(x, u):
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    acc = SO3.act(q, ca.vertcat(0, 0, u[0])) + ca.DM([0, 0, -9.81])
    return ca.vertcat(p + v*dt, v + acc*dt, SO3.retract(q, w*dt), w + u[1:4]*dt)

  x_init = ca.vertcat(ca.DM.zeros(6), SO3.exp(ca.DM([0.3, 0, 0])), ca.DM.zeros(3))
  x_goal = ca.vertcat(ca.DM([1, 1, 1]), ca.DM.zeros(3), SO3.exp(ca.DM([0, 0, 1.0])),
                      ca.DM.zeros(3))
  opti.subject_to(Mx.minus(X[:, 0], x_init) == 0)
  for k in range(N):
    opti.subject_to(Mx.minus(X[:, k+1], F(X[:, k], U[:, k])) == 0)
  opti.subject_to(opti.bounded(0, U[0, :], 20))
  opti.subject_to(opti.bounded(-2, U[1:, :], 2))
  opti.minimize(ca.sumsqr(U[1:, :])*dt + 0.01*ca.sumsqr(U[0, :] - 9.81)*dt
                + 100*ca.sumsqr(Mx.minus(x_goal, X[:, N])))
  opti.set_initial(U[0, :], 9.81)
  return X, U, Mx


class ManifoldOptiTests(casadiTestCase):
  """Decision variables on manifolds in Opti"""

  def test_variable(self):
    opti = ca.Opti()
    SO3 = ca.Manifold.SO3()
    q = opti.variable(SO3)
    Q = opti.variable(SO3, 3)
    self.assertEqual(q.shape, (4, 1))
    self.assertEqual(Q.shape, (4, 3))
    # default initial guess is the identity element
    self.checkarray(opti.debug.value(Q, opti.initial()), ca.repmat(SO3.identity(), 1, 3))

  def test_quadrotor(self):
    opti = ca.Opti()
    X, U, Mx = quadrotor(opti)
    opti.solver("sqpmethod", dict(SQP_OPTS, tol_pr=1e-10, tol_du=1e-10))
    sol = opti.solve()
    self.assertTrue(sol.stats()["success"])
    Xv = ca.DM(sol.value(X))
    for k in range(Xv.shape[1]):
      self.checkarray(ca.norm_2(Xv[6:10, k]), ca.DM(1), digits=10)
    if ca.has_nlpsol("ipopt"):
      opti_i = ca.Opti()
      Xi, _, _ = quadrotor(opti_i)
      opti_i.solver("ipopt", {"manifold_fallback": "embed", "print_time": False,
                              "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.tol": 1e-10})
      sol_i = opti_i.solve()
      self.checkarray(Xv, ca.DM(sol_i.value(Xi)), digits=5)

  def test_to_function(self):
    opti = ca.Opti()
    SO3 = ca.Manifold.SO3()
    q = opti.variable(SO3)
    target = opti.parameter(3)
    opti.minimize(ca.sumsqr(SO3.minus(SO3.exp(target), q)))
    opti.solver("sqpmethod", SQP_OPTS)
    F = opti.to_function("F", [target], [q])
    v = ca.DM([0.4, -0.1, 0.7])
    self.checkarray(F(v), SO3.exp(v), digits=8)

  def test_ambient_equality(self):
    R3 = ca.Manifold.Euclidean(3)
    SO3 = ca.Manifold.SO3()
    Mx = ca.Manifold.product([R3, SO3])
    for lhs in [lambda X: X[:, 0], lambda X: X[3:7, 0], lambda X: 2*X[3:7, 1]]:
      opti = ca.Opti()
      X = opti.variable(Mx, 2)
      opti.subject_to(lhs(X) == ca.DM.ones(lhs(X).shape))
      opti.minimize(ca.sumsqr(X))
      opti.solver("sqpmethod", SQP_OPTS)
      with self.assertRaisesRegex(Exception, "M.minus"):
        opti.solve()
    # Euclidean parts and partial quaternion constraints are fine
    opti = ca.Opti()
    X = opti.variable(Mx, 1)
    opti.subject_to(X[0:3] == ca.DM([1, 2, 3]))
    opti.subject_to(X[6] == 0)
    q_target = SO3.exp(ca.DM([0.3, 0.2, 0.1]))
    opti.minimize(ca.sumsqr(SO3.minus(q_target, X[3:7])))
    opti.solver("sqpmethod", SQP_OPTS)
    sol = opti.solve()
    self.checkarray(sol.value(X[0:3]), ca.DM([1, 2, 3]), digits=8)
    self.checkarray(ca.DM(sol.value(X[6])), ca.DM(0), digits=8)

  def test_errors(self):
    SO3 = ca.Manifold.SO3()
    # explicit x_manifold
    opti = ca.Opti()
    q = opti.variable(SO3)
    opti.minimize(ca.sumsqr(SO3.minus(SO3.exp(ca.DM([1, 0, 0])), q)))
    opti.solver("sqpmethod", dict(SQP_OPTS, x_manifold=SO3))
    with self.assertRaisesRegex(Exception, "set by Opti"):
      opti.solve()
    # linear scaling
    opti = ca.Opti()
    q = opti.variable(SO3)
    opti.minimize(ca.sumsqr(SO3.minus(SO3.exp(ca.DM([1, 0, 0])), q)))
    opti.set_linear_scale(q, 2)
    opti.solver("sqpmethod", SQP_OPTS)
    with self.assertRaisesRegex(Exception, "Linear scaling"):
      opti.solve()
    # solver without native support
    if ca.has_nlpsol("ipopt"):
      opti = ca.Opti()
      q = opti.variable(SO3)
      opti.minimize(ca.sumsqr(SO3.minus(SO3.exp(ca.DM([1, 0, 0])), q)))
      opti.solver("ipopt")
      with self.assertRaisesRegex(Exception, "no native support"):
        opti.solve()

  def test_initial_guess(self):
    SO3 = ca.Manifold.SO3()
    opti = ca.Opti()
    q = opti.variable(SO3)
    q_target = SO3.exp(ca.DM([1, 0.5, 0]))
    opti.minimize(ca.sumsqr(SO3.minus(q_target, q)))
    # an initial guess off the manifold is projected by nlpsol
    opti.set_initial(q, [2, 0, 0.2, 0])
    opti.solver("sqpmethod", SQP_OPTS)
    sol = opti.solve()
    self.checkarray(SO3.minus(ca.DM(sol.value(q)), q_target), ca.DM.zeros(3, 1), digits=8)


def quadrotor_staged(opti, N=10, dt=0.1):
  # type: (ca.Opti, int, float) -> tuple[list, list]
  """Quadrotor with variables and constraints in stage order, as fatrop requires"""
  R3 = ca.Manifold.Euclidean(3)
  SO3 = ca.Manifold.SO3()
  Mx = ca.Manifold.product([R3, R3, SO3, R3])

  def F(x, u):
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    acc = SO3.act(q, ca.vertcat(0, 0, u[0])) + ca.DM([0, 0, -9.81])
    return ca.vertcat(p + v*dt, v + acc*dt, SO3.retract(q, w*dt), w + u[1:4]*dt)

  x_init = ca.vertcat(ca.DM.zeros(6), SO3.exp(ca.DM([0.3, 0, 0])), ca.DM.zeros(3))
  x_goal = ca.vertcat(ca.DM([1, 1, 1]), ca.DM.zeros(3), SO3.exp(ca.DM([0, 0, 1.0])),
                      ca.DM.zeros(3))
  X, U = [], []
  for k in range(N+1):
    X.append(opti.variable(Mx))
    if k < N:
      U.append(opti.variable(4))
  J = 0
  for k in range(N):
    opti.subject_to(Mx.minus(X[k+1], F(X[k], U[k])) == 0)
    if k == 0:
      opti.subject_to(Mx.minus(X[0], x_init) == 0)
    opti.subject_to(opti.bounded(0, U[k][0], 20))
    opti.subject_to(opti.bounded(-2, U[k][1:], 2))
    opti.set_initial(U[k][0], 9.81)
    J += ca.sumsqr(U[k][1:])*dt + 0.01*ca.sumsqr(U[k][0] - 9.81)*dt
  opti.minimize(J + 100*ca.sumsqr(Mx.minus(x_goal, X[N])))
  return X, U


FATROP_OPTS = {"expand": True, "print_time": False,
               "fatrop": {"print_level": 0, "tol": 1e-12}}


class ManifoldFatropTests(casadiTestCase):
  """Native support for x_manifold in fatrop"""

  def setUp(self):
    if not ca.has_nlpsol("fatrop"):
      self.skipTest("fatrop not available")

  def test_quadrotor_opti(self):
    N = 10
    opti = ca.Opti()
    X, U = quadrotor_staged(opti, N)
    opti.solver("sqpmethod", dict(SQP_OPTS, tol_pr=1e-10, tol_du=1e-10))
    sol = opti.solve()
    Xs = ca.hcat([ca.DM(sol.value(x)) for x in X])
    for structure in [{"structure_detection": "auto"},
                      {"structure_detection": "manual", "N": N, "nx": [13]*(N+1),
                       "nu": [4]*N + [0], "ng": [16] + [4]*(N-1) + [0]}]:
      opti_f = ca.Opti()
      Xf_sym, _ = quadrotor_staged(opti_f, N)
      opti_f.solver("fatrop", dict(FATROP_OPTS, **structure))
      sol_f = opti_f.solve()
      stats = sol_f.stats()
      self.assertTrue(stats["success"])
      # fatrop works with tangent dimensions
      self.assertEqual(list(stats["nx"]), [12]*(N+1))
      Xf = ca.hcat([ca.DM(sol_f.value(x)) for x in Xf_sym])
      self.checkarray(Xf, Xs, str(structure), digits=6)
      for k in range(N+1):
        self.checkarray(ca.norm_2(Xf[6:10, k]), ca.DM(1), digits=12)

  def test_multipliers(self):
    # bounds on Euclidean controls (lam_x) and normalized dynamics rows (lam_g)
    nlp, M, args = attitude_ocp(N=10)
    S = ca.nlpsol("S", "sqpmethod", nlp, dict(SQP_OPTS, x_manifold=M, tol_pr=1e-12,
                                                tol_du=1e-12))
    r = S(**args)
    F = ca.nlpsol("F", "fatrop", nlp, dict(FATROP_OPTS, x_manifold=M, structure_detection="auto",
                                           equality=[True]*nlp["g"].shape[0]))
    rf = F(**args)
    self.assertTrue(F.stats()["success"])
    self.checkarray(rf["x"], r["x"], digits=6)
    self.checkarray(rf["f"], r["f"], digits=4)
    self.checkarray(rf["lam_g"]/400, r["lam_g"]/400, digits=6)
    self.checkarray(rf["lam_x"], r["lam_x"], digits=6)
    self.checkarray(rf["lam_x"][::10], ca.DM.zeros(11, 1))  # first entry of each quaternion

  def test_structure_errors(self):
    N = 4
    opti = ca.Opti()
    quadrotor_staged(opti, N)
    # stage boundary inside a quaternion
    opti.solver("fatrop", dict(FATROP_OPTS, structure_detection="manual", N=N,
                               nx=[8] + [13]*N, nu=[9] + [4]*(N-1) + [0],
                               ng=[16] + [4]*(N-1) + [0]))
    with self.assertRaisesRegex(Exception, "splits an element"):
      opti.solve()

  def test_euclidean_unchanged(self):
    # Euclidean OCP (no x_manifold) through the same runtime, including code generation
    N = 8
    w, g, lbw, ubw = [], [], [], []
    J = 0
    x = ca.SX.sym("x0", 2)
    w.append(x)
    lbw += [1, 0]
    ubw += [1, 0]
    for k in range(N):
      u = ca.SX.sym("u%d" % k)
      w.append(u)
      lbw += [-1]
      ubw += [1]
      x_next = ca.SX.sym("x%d" % (k+1), 2)
      w.append(x_next)
      lbw += [-inf, -inf]
      ubw += [inf, inf]
      g.append(x_next - ca.vertcat(x[0] + 0.1*x[1], x[1] + 0.1*u))
      J += ca.sumsqr(x) + u**2
      x = x_next
    nlp = {"x": ca.vertcat(*w), "f": J + 10*ca.sumsqr(x), "g": ca.vertcat(*g)}
    args = {"lbx": lbw, "ubx": ubw, "lbg": 0, "ubg": 0}
    F = ca.nlpsol("F", "fatrop", nlp, dict(FATROP_OPTS, structure_detection="auto",
                                           equality=[True]*2*N))
    S = ca.nlpsol("S", "sqpmethod", nlp, dict(SQP_OPTS, tol_pr=1e-12, tol_du=1e-12))
    rf, rs = F(**args), S(**args)
    self.assertTrue(F.stats()["success"])
    for k in ["x", "lam_g", "lam_x"]:
      self.checkarray(rf[k], rs[k], k, digits=6)
    flags = [] if os.name == "nt" else ["-Wno-strict-prototypes"]
    self.check_codegen(F, args, std="c99", extralibs=["fatrop", "blasfeo"], extra_options=flags,
                       digits=8)

  def test_serialize_and_codegen(self):
    nlp, M, args = attitude_ocp(N=6)
    F = ca.nlpsol("F", "fatrop", nlp, dict(FATROP_OPTS, x_manifold=M, structure_detection="auto",
                                           equality=[True]*nlp["g"].shape[0]))
    F2 = ca.Function.deserialize(F.serialize())
    r, r2 = F(**args), F2(**args)
    self.assertTrue(F2.stats()["success"])
    self.checkarray(r["x"], r2["x"], digits=10)
    # Code generation (compiled and compared with --run_slow)
    flags = [] if os.name == "nt" else ["-Wno-strict-prototypes"]
    self.check_codegen(F, args, std="c99", extralibs=["fatrop", "blasfeo"], extra_options=flags,
                       digits=8)
    # sqpmethod has no code generation support with x_manifold
    S = ca.nlpsol("S", "sqpmethod", nlp, dict(SQP_OPTS, x_manifold=M))
    with self.assertRaisesRegex(Exception, "Code generation"):
      S.generate("manifold_sqp_codegen.c")


if __name__ == '__main__':
    unittest.main()
