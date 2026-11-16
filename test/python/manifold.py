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
import unittest
from helpers import casadiTestCase


class ManifoldTests(casadiTestCase):

  def test_euclidean(self):
    R3 = ca.Manifold.Euclidean(3)
    self.assertEqual(R3.nx(), 3)
    self.assertEqual(R3.ndx(), 3)
    self.assertTrue(R3.is_euclidean())
    self.assertEqual(list(R3.euclidean_mask()), [True]*3)
    self.assertEqual(list(R3.euclidean_mask_tangent()), [True]*3)
    self.assertEqual(R3.n_components(), 1)
    self.assertEqual(list(R3.offsets()), [0, 3])
    self.assertEqual(str(R3), "Euclidean(3)")
    self.checkarray(R3.identity(), ca.DM.zeros(3, 1))

    x = ca.DM([1, 2, 3])
    dx = ca.DM([0.5, 0, -1])
    self.checkarray(R3.retract(x, dx), x + dx)
    self.checkarray(R3.minus(x + dx, x), dx)
    self.checkarray(R3.project(x), x)
    self.assertEqual(R3.constraint(x).shape, (0, 1))

  def test_product(self):
    R2 = ca.Manifold.Euclidean(2)
    R1 = ca.Manifold.Euclidean(1)
    M = ca.Manifold.product([R2, R1])
    self.assertEqual(M.nx(), 3)
    self.assertEqual(M.ndx(), 3)
    self.assertTrue(M.is_euclidean())
    self.assertEqual(M.n_components(), 2)
    self.assertEqual(M.component(0).nx(), 2)
    self.assertEqual(M.component(1).nx(), 1)
    self.assertEqual(list(M.offsets()), [0, 2, 3])
    self.assertEqual(list(M.offsets_tangent()), [0, 2, 3])
    self.assertEqual(str(M), "Product(Euclidean(2), Euclidean(1))")
    with self.assertRaises(Exception):
      M.component(2)

    x = ca.DM([1, 2, 3])
    dx = ca.DM([0.5, 0, -1])
    self.checkarray(M.retract(x, dx), ca.DM([1.5, 2, 2]))
    self.checkarray(M.minus(M.retract(x, dx), x), dx)

    # nested products keep their structure
    N = ca.Manifold.product([M, R1])
    self.assertEqual(N.nx(), 4)
    self.assertEqual(N.n_components(), 2)
    self.assertEqual(N.component(0).n_components(), 2)

    # empty product
    E = ca.Manifold.product([])
    self.assertEqual(E.nx(), 0)
    self.assertEqual(E.retract(ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)).shape, (0, 1))

  def test_numpy_input(self):
    R3 = ca.Manifold.Euclidean(3)
    r = R3.retract(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    self.assertTrue(isinstance(r, ca.DM))
    self.checkarray(r, ca.DM([2, 3, 4]))

  def test_symbolic(self):
    M = ca.Manifold.product([ca.Manifold.Euclidean(2), ca.Manifold.Euclidean(1)])
    x0 = ca.DM([1, 2, 3])
    dx0 = ca.DM([0.5, 0, -1])

    xs = ca.SX.sym("x", 3)
    dxs = ca.SX.sym("dx", 3)
    rs = M.retract(xs, dxs)
    self.assertTrue(isinstance(rs, ca.SX))
    fs = ca.Function("fs", [xs, dxs], [rs, ca.jacobian(rs, dxs)])

    xm = ca.MX.sym("x", 3)
    dxm = ca.MX.sym("dx", 3)
    rm = M.retract(xm, dxm)
    self.assertTrue(isinstance(rm, ca.MX))
    fm = ca.Function("fm", [xm, dxm], [rm, ca.jacobian(rm, dxm)])

    for f in [fs, fm]:
      r, J = f(x0, dx0)
      self.checkarray(r, M.retract(x0, dx0))
      self.checkarray(J, ca.DM.eye(3))

  def test_columnwise(self):
    M = ca.Manifold.product([ca.Manifold.Euclidean(2), ca.Manifold.Euclidean(1)])
    X = ca.DM.rand(3, 4)
    DX = ca.DM.rand(3, 4)
    Y = M.retract(X, DX)
    self.assertEqual(Y.shape, (3, 4))
    self.checkarray(Y, X + DX)
    # broadcasting of a single column
    x = ca.DM([1, 2, 3])
    self.checkarray(M.minus(Y, x), Y - ca.repmat(x, 1, 4))
    # symbolic
    Xm = ca.MX.sym("X", 3, 4)
    f = ca.Function("f", [Xm], [M.retract(Xm, ca.MX(DX))])
    self.checkarray(f(X), X + DX)
    # inconsistent number of columns
    with self.assertRaises(Exception):
      M.retract(ca.DM.rand(3, 2), ca.DM.rand(3, 3))

  def test_dimension_errors(self):
    M = ca.Manifold.Euclidean(3)
    with self.assertRaises(Exception):
      M.retract(ca.DM([1, 2]), ca.DM([1, 2, 3]))
    with self.assertRaises(Exception):
      M.minus(ca.DM([1, 2, 3]), ca.DM([1, 2]))

  def test_get_function(self):
    M = ca.Manifold.Euclidean(2)
    self.assertEqual(list(ca.Manifold.function_names()),
                     ["retract", "minus", "defect", "project", "constraint",
                      "jac_retract_dx", "jac_minus_y", "compose", "inverse", "exp", "log", "act", "hat",
                      "vee", "as_matrix", "Adj", "Jr", "Jl", "Jr_inv", "Jl_inv"])
    f = M.get_function("retract")
    self.assertTrue(isinstance(f, ca.Function))
    self.assertEqual(f.name_in(), ["x", "dx"])
    self.assertEqual(f.name_out(), ["x_next"])
    # cached: same underlying Function
    self.assertTrue(M.get_function("retract").__hash__() == f.__hash__())
    with self.assertRaises(Exception):
      M.get_function("no_such_function")


try:
  from scipy.linalg import expm as _expm
except ImportError:
  _expm = None


def lie_groups():
  # type: () -> list[ca.Manifold]
  """Lie groups covered by the generic property tests"""
  return [ca.Manifold.Euclidean(3),
          ca.Manifold.SO2(),
          ca.Manifold.SO3(),
          ca.Manifold.SE2(),
          ca.Manifold.SE3(),
          ca.Manifold.product([ca.Manifold.Euclidean(2), ca.Manifold.SO3()]),
          ca.Manifold.product([ca.Manifold.SE2(), ca.Manifold.SO2(), ca.Manifold.SE3()])]


def ad_jacobians(G):
  # type: (ca.Manifold) -> ca.Function
  """Reference Adj/Jr/Jl/Jr_inv/Jl_inv by algorithmic differentiation"""
  a = ca.SX.sym("a", G.nx())
  v = ca.SX.sym("v", G.ndx())
  e = ca.SX.sym("e", G.ndx())
  z = ca.DM.zeros(G.ndx(), 1)
  Adj = ca.jacobian(G.log(G.compose(G.compose(a, G.exp(e)), G.inverse(a))), e)
  Jr = ca.jacobian(G.log(G.compose(G.inverse(G.exp(v)), G.exp(v + e))), e)
  Jl = ca.jacobian(G.log(G.compose(G.exp(v + e), G.inverse(G.exp(v)))), e)
  Jr_inv = ca.jacobian(G.log(G.compose(G.exp(v), G.exp(e))), e)
  Jl_inv = ca.jacobian(G.log(G.compose(G.exp(e), G.exp(v))), e)
  out = [ca.substitute(J, e, z) for J in [Adj, Jr, Jl, Jr_inv, Jl_inv]]
  return ca.Function("ref", [a, v], out)


class LieGroupTests(casadiTestCase):

  def random_tangent(self, G, scale=1.0):
    # type: (ca.Manifold, float) -> ca.DM
    return ca.DM(np.random.uniform(-scale, scale, (G.ndx(), 1)))

  def random_element(self, G):
    # type: (ca.Manifold) -> ca.DM
    return G.exp(self.random_tangent(G))

  def assert_same_element(self, G, a, b, digits=10):
    # type: (ca.Manifold, ca.DM, ca.DM, int) -> None
    """a and b represent the same group element"""
    self.checkarray(G.minus(a, b), ca.DM.zeros(G.ndx(), 1), digits=digits)

  def test_is_lie_group(self):
    for G in lie_groups():
      self.assertTrue(G.is_lie_group())

  def test_axioms(self):
    np.random.seed(0)
    for G in lie_groups():
      e = G.identity()
      for _ in range(5):
        a, b, c = [self.random_element(G) for _ in range(3)]
        self.assert_same_element(G, G.compose(a, G.inverse(a)), e)
        self.assert_same_element(G, G.compose(G.inverse(a), a), e)
        self.assert_same_element(G, G.compose(a, e), a)
        self.assert_same_element(G, G.compose(G.compose(a, b), c),
                                 G.compose(a, G.compose(b, c)))
        self.assert_same_element(G, G.exp(G.log(a)), a)
        v = self.random_tangent(G)
        self.checkarray(G.log(G.exp(v)), v)

  def test_retract_minus(self):
    np.random.seed(1)
    for G in lie_groups():
      for _ in range(5):
        x, y = self.random_element(G), self.random_element(G)
        dx = self.random_tangent(G)
        self.assert_same_element(G, G.retract(x, ca.DM.zeros(G.ndx(), 1)), x)
        self.checkarray(G.minus(x, x), ca.DM.zeros(G.ndx(), 1))
        self.assert_same_element(G, G.retract(x, G.minus(y, x)), y)
        self.checkarray(G.minus(G.retract(x, dx), x), dx)
        # right perturbation convention
        self.assert_same_element(G, G.retract(x, dx), G.compose(x, G.exp(dx)))

  def test_jacobians_vs_ad(self):
    np.random.seed(2)
    for G in lie_groups():
      ref = ad_jacobians(G)
      for scale in [0.0, 1e-6, 0.1, 1.0, 1.8]:  # |v| < pi (no wrap-around)
        a = self.random_element(G)
        v = self.random_tangent(G, scale)
        Adj, Jr, Jl, Jr_inv, Jl_inv = ref(a, v)
        self.checkarray(G.Adj(a), Adj, str(G) + " Adj")
        self.checkarray(G.Jr(v), Jr, str(G) + " Jr")
        self.checkarray(G.Jl(v), Jl, str(G) + " Jl")
        self.checkarray(G.Jr_inv(v), Jr_inv, str(G) + " Jr_inv")
        self.checkarray(G.Jl_inv(v), Jl_inv, str(G) + " Jl_inv")
        I = ca.DM.eye(G.ndx())
        self.checkarray(ca.mtimes(G.Jr(v), G.Jr_inv(v)), I)
        self.checkarray(ca.mtimes(G.Jl(v), G.Jl_inv(v)), I)
        self.checkarray(G.Jl(v), G.Jr(-v))
        self.checkarray(ca.mtimes(G.Adj(G.exp(v)), G.Jr(v)), G.Jl(v))

  def test_adjoint(self):
    np.random.seed(3)
    for G in lie_groups():
      a = self.random_element(G)
      v = self.random_tangent(G)
      self.assert_same_element(G, G.compose(a, G.exp(v)),
                               G.compose(G.exp(ca.mtimes(G.Adj(a), v)), a))

  def test_matrix_representation(self):
    np.random.seed(4)
    for G in lie_groups():
      if G.n_components() > 1:
        with self.assertRaises(Exception):
          G.as_matrix(G.identity())
        continue
      a, b = self.random_element(G), self.random_element(G)
      v = self.random_tangent(G)
      self.checkarray(G.as_matrix(G.compose(a, b)),
                      ca.mtimes(G.as_matrix(a), G.as_matrix(b)))
      self.checkarray(G.as_matrix(G.inverse(a)), ca.inv(G.as_matrix(a)))
      self.checkarray(G.vee(G.hat(v)), v)
      if _expm is not None:
        self.checkarray(G.as_matrix(G.exp(v)), ca.DM(_expm(np.array(G.hat(v)))))

  def test_act(self):
    np.random.seed(5)
    for G in lie_groups():
      a, b = self.random_element(G), self.random_element(G)
      p = ca.DM(np.random.randn(G.get_function("act").size1_in(1), 1))
      self.checkarray(G.act(G.compose(a, b), p), G.act(a, G.act(b, p)))
      self.checkarray(G.act(G.identity(), p), p)

  def test_finite_derivatives(self):
    """Values, gradients and Hessians stay finite at singular points"""
    for G in lie_groups():
      v = ca.SX.sym("v", G.ndx())
      a = ca.SX.sym("a", G.nx())
      ops_v = [G.exp(v), ca.vec(G.Jr(v)), ca.vec(G.Jl(v)), ca.vec(G.Jr_inv(v)),
               ca.vec(G.Jl_inv(v)), G.log(G.exp(v))]
      ops_a = [G.log(a), G.minus(a, G.identity())]
      w = ca.SX.sym("w", 1)
      for ex, arg in [(e, v) for e in ops_v] + [(e, a) for e in ops_a]:
        f = ca.Function("f", [arg],
                        [ex, ca.jacobian(ex, arg), ca.hessian(ca.dot(ex, ex), arg)[0]])
        for pt in self.singular_points(G, arg.numel() == G.nx()):
          for r in f(pt):
            self.assertTrue(np.all(np.isfinite(np.array(r))), str(G) + " at " + str(pt))

  def singular_points(self, G, ambient):
    # type: (ca.Manifold, bool) -> list[ca.DM]
    pts = []  # type: list[ca.DM]
    for th in [0.0, 1e-12, 1e-6, 0.1 - 1e-9, 0.1 + 1e-9, 0.2 - 1e-9, 0.2 + 1e-9, 1.0,
               np.pi - 1e-6, np.pi]:
      v = ca.DM.zeros(G.ndx(), 1)
      v[G.ndx() - 1] = th
      if G.ndx() > 1:
        v[0] = th / 3
      pts.append(G.exp(v) if ambient else v)
    return pts

  def test_series_continuity(self):
    """No jump in value or derivatives where Taylor series take over"""
    for G in lie_groups():
      v = ca.SX.sym("v", G.ndx())
      ex = ca.veccat(G.exp(v), G.Jr(v), G.Jr_inv(v), G.log(G.exp(v)))
      f = ca.Function("f", [v], [ex, ca.jacobian(ex, v)])
      for thr in [0.1, 0.2]:  # sqrt of the switching thresholds
        d = ca.DM.zeros(G.ndx(), 1)
        lo, hi = d, ca.DM(d)
        lo[G.ndx() - 1] = thr * (1 - 1e-9)
        hi[G.ndx() - 1] = thr * (1 + 1e-9)
        for rl, rh in zip(f(lo), f(hi)):
          self.checkarray(rl, rh, str(G), digits=7)

  def test_symbolic_types(self):
    np.random.seed(6)
    for G in lie_groups():
      a = self.random_element(G)
      v = self.random_tangent(G)
      for X in [ca.SX, ca.MX]:
        sa = X.sym("a", G.nx())
        sv = X.sym("v", G.ndx())
        f = ca.Function("f", [sa, sv], [G.compose(sa, G.exp(sv)), G.log(sa), G.Jr(sv)])
        r = f(a, v)
        self.checkarray(r[0], G.compose(a, G.exp(v)))
        self.checkarray(r[1], G.log(a))
        self.checkarray(r[2], G.Jr(v))

  def test_columnwise(self):
    np.random.seed(7)
    G = ca.Manifold.SO3()
    V = ca.DM(np.random.randn(3, 4))
    A = G.exp(V)
    self.assertEqual(A.shape, (4, 4))
    for k in range(4):
      self.checkarray(A[:, k], G.exp(V[:, k]))
    # matrix-valued results are stacked horizontally
    J = G.Jr(V)
    self.assertEqual(J.shape, (3, 12))
    self.checkarray(J[:, 3:6], G.Jr(V[:, 1]))
    # matrix-valued inputs likewise
    self.checkarray(G.vee(G.hat(V)), V)


class SO3Tests(casadiTestCase):

  def test_conventions(self):
    G = ca.Manifold.SO3()
    self.assertEqual((G.nx(), G.ndx()), (4, 3))
    self.assertEqual(str(G), "SO3")
    self.checkarray(G.identity(), ca.DM([1, 0, 0, 0]))
    self.assertFalse(any(G.euclidean_mask()))
    # 90 degrees about z: q = [cos(pi/4), 0, 0, sin(pi/4)]
    q = G.exp(ca.DM([0, 0, np.pi/2]))
    self.checkarray(q, ca.DM([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]))
    self.checkarray(G.act(q, ca.DM([1, 0, 0])), ca.DM([0, 1, 0]))
    self.checkarray(G.as_matrix(q), ca.DM([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    self.checkarray(G.hat(ca.DM([1, 2, 3])), ca.DM([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]))

  def test_double_cover(self):
    G = ca.Manifold.SO3()
    v = ca.DM([0.3, -0.4, 0.5])
    q = G.exp(v)
    self.checkarray(G.log(-q), v)
    self.checkarray(G.minus(-q, q), ca.DM.zeros(3, 1))

  def test_log_scale_invariant(self):
    G = ca.Manifold.SO3()
    v = ca.DM([0.3, -0.4, 0.5])
    self.checkarray(G.log(3.7*G.exp(v)), v)

  def test_project_constraint(self):
    G = ca.Manifold.SO3()
    q = ca.DM([1, 2, 3, 4])
    self.checkarray(G.project(q), q/ca.norm_2(q))
    self.checkarray(G.constraint(q), ca.DM(29))
    self.checkarray(G.constraint(G.project(q)), ca.DM(0))

  def test_against_scipy(self):
    try:
      from scipy.spatial.transform import Rotation
    except ImportError:
      self.skipTest("scipy not available")
    G = ca.Manifold.SO3()
    np.random.seed(8)
    for _ in range(5):
      v = np.random.randn(3)
      R = Rotation.from_rotvec(v)
      self.checkarray(G.as_matrix(G.exp(ca.DM(v))), ca.DM(R.as_matrix()))
      x, y, z, w = R.as_quat()
      self.assert_quat_equal(G.exp(ca.DM(v)), ca.DM([w, x, y, z]))

  def assert_quat_equal(self, a, b):
    # type: (ca.DM, ca.DM) -> None
    if float(a[0]) * float(b[0]) < 0:
      b = -b
    self.checkarray(a, b)

  def test_not_lie(self):
    # Lie operations of a product are available only if all components are groups
    M = ca.Manifold.product([ca.Manifold.SO3(), ca.Manifold.Euclidean(1)])
    self.assertTrue(M.is_lie_group())


class SO2SE2SE3Tests(casadiTestCase):

  def test_so2(self):
    G = ca.Manifold.SO2()
    self.assertEqual((G.nx(), G.ndx()), (2, 1))
    self.checkarray(G.identity(), ca.DM([1, 0]))
    a = G.exp(ca.DM(0.3))
    self.checkarray(a, ca.DM([np.cos(0.3), np.sin(0.3)]))
    self.checkarray(G.log(G.compose(a, G.exp(ca.DM(0.5)))), ca.DM(0.8))
    self.checkarray(G.minus(G.exp(ca.DM(3.0)), G.exp(ca.DM(-3.0))), ca.DM(6.0 - 2*np.pi))
    self.checkarray(G.act(G.exp(ca.DM(np.pi/2)), ca.DM([1, 0])), ca.DM([0, 1]))
    self.checkarray(G.project(ca.DM([3, 4])), ca.DM([0.6, 0.8]))
    self.checkarray(G.constraint(ca.DM([3, 4])), ca.DM(24))

  def test_se2(self):
    G = ca.Manifold.SE2()
    self.assertEqual((G.nx(), G.ndx()), (4, 3))
    self.checkarray(G.identity(), ca.DM([0, 0, 1, 0]))
    self.assertFalse(any(G.euclidean_mask()))
    # pure translation
    self.checkarray(G.exp(ca.DM([1, 2, 0])), ca.DM([1, 2, 1, 0]))
    # quarter turn about the origin while moving forward: an arc of radius 2/pi
    a = G.exp(ca.DM([1, 0, np.pi/2]))
    self.checkarray(a, ca.DM([2/np.pi, 2/np.pi, 0, 1]))
    # composition = chaining poses
    b = ca.DM([1, 0, 0, 1])  # at (1, 0), facing +y
    self.checkarray(G.compose(b, b), ca.DM([1, 1, -1, 0]))
    self.checkarray(G.act(b, ca.DM([1, 0])), ca.DM([1, 1]))
    self.checkarray(G.project(ca.DM([5, 6, 3, 4])), ca.DM([5, 6, 0.6, 0.8]))

  def test_se3(self):
    G = ca.Manifold.SE3()
    self.assertEqual((G.nx(), G.ndx()), (7, 6))
    self.checkarray(G.identity(), ca.DM([0, 0, 0, 1, 0, 0, 0]))
    self.assertFalse(any(G.euclidean_mask()))
    # pure translation
    self.checkarray(G.exp(ca.DM([1, 2, 3, 0, 0, 0])), ca.DM([1, 2, 3, 1, 0, 0, 0]))
    # screw motion: rotation about z with translation along z is decoupled
    a = G.exp(ca.DM([0, 0, 2, 0, 0, np.pi/2]))
    self.checkarray(a, ca.DM([0, 0, 2, np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]))
    self.checkarray(G.act(a, ca.DM([1, 0, 0])), ca.DM([0, 1, 2]))
    # SE3 couples translation and rotation, unlike R3 x SO3
    P = ca.Manifold.product([ca.Manifold.Euclidean(3), ca.Manifold.SO3()])
    x = G.exp(ca.DM([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    dx = ca.DM([1, 0, 0, 0, 0, 0])
    q = x[3:]
    self.checkarray(G.retract(x, dx)[:3], x[:3] + ca.mtimes(ca.Manifold.SO3().as_matrix(q), dx[:3]))
    self.checkarray(P.retract(x, dx)[:3], x[:3] + dx[:3])
    self.checkarray(G.project(ca.DM([1, 2, 3, 2, 0, 0, 0])), ca.DM([1, 2, 3, 1, 0, 0, 0]))

  def test_product_point_dim(self):
    M = ca.Manifold.product([ca.Manifold.SE2(), ca.Manifold.SO3()])
    f = M.get_function("act")
    self.assertEqual(f.size1_in(1), 5)
    a = M.exp(ca.DM([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    p = ca.DM([1, 2, 3, 4, 5])
    self.checkarray(M.act(a, p), ca.vertcat(M.component(0).act(a[:4], p[:2]),
                                            M.component(1).act(a[4:], p[2:])))


def sphere_functions():
  # type: () -> tuple[ca.Function, ca.Function, ca.Function]
  """Unit sphere S2 (nx=3, ndx=2) with the normalization retraction"""
  x = ca.SX.sym("x", 3)
  y = ca.SX.sym("y", 3)
  dx = ca.SX.sym("dx", 2)
  a, b, c = x[0], x[1], x[2]
  # orthonormal basis of the tangent plane at x (valid for c > -1)
  B = ca.horzcat(ca.vertcat(1 - a*a/(1 + c), -a*b/(1 + c), -a),
                 ca.vertcat(-a*b/(1 + c), 1 - b*b/(1 + c), -b))
  z = x + ca.mtimes(B, dx)
  retract = ca.Function("S2_retract", [x, dx], [z/ca.norm_2(z)])
  minus = ca.Function("S2_minus", [y, x], [ca.mtimes(B.T, y/ca.dot(x, y) - x)])
  project = ca.Function("S2_project", [x], [x/ca.norm_2(x)])
  return retract, minus, project


class CustomManifoldTests(casadiTestCase):

  def test_sphere(self):
    retract, minus, project = sphere_functions()
    c = ca.SX.sym("x", 3)
    constraint = ca.Function("S2_c", [c], [ca.dot(c, c) - 1])
    S2 = ca.Manifold("S2", retract, minus,
                     {"project": project, "constraint": constraint, "identity": [0, 0, 1]})
    self.assertEqual((S2.nx(), S2.ndx()), (3, 2))
    self.assertFalse(S2.is_lie_group())
    self.assertFalse(S2.is_euclidean())
    self.assertEqual(str(S2), "S2")
    self.assertEqual(repr(S2), "Manifold(S2)")
    self.checkarray(S2.identity(), ca.DM([0, 0, 1]))
    x = S2.project(ca.DM([1, 2, 3]))
    self.checkarray(S2.constraint(x), ca.DM(0))
    dx = ca.DM([0.1, -0.2])
    y = S2.retract(x, dx)
    self.checkarray(S2.minus(y, x), dx)
    self.checkarray(S2.constraint(y), ca.DM(0))
    with self.assertRaises(Exception):
      S2.exp(dx)
    with self.assertRaises(Exception):
      S2.get_function("compose")
    # symbolic use and products with built-in manifolds
    M = ca.Manifold.product([ca.Manifold.Euclidean(1), S2])
    self.assertFalse(M.is_lie_group())
    xs = ca.MX.sym("x", 4)
    dxs = ca.MX.sym("dx", 3)
    f = ca.Function("f", [xs, dxs], [M.retract(xs, dxs)])
    self.checkarray(f(ca.vertcat(1, x), ca.vertcat(2, dx)), ca.vertcat(3, y))

  def test_positive_reals(self):
    # R+^n with the multiplicative retraction, built from MX Functions
    x = ca.MX.sym("x", 2)
    y = ca.MX.sym("y", 2)
    dx = ca.MX.sym("dx", 2)
    retract = ca.Function("r", [x, dx], [x*ca.exp(dx)])
    minus = ca.Function("m", [y, x], [ca.log(y/x)])
    P = ca.Manifold("Rplus", retract, minus, {"identity": [1, 1]})
    self.checkarray(P.retract(ca.DM([2, 3]), ca.DM([0, np.log(2)])), ca.DM([2, 6]))
    self.checkarray(P.minus(ca.DM([2, 6]), ca.DM([2, 3])), ca.DM([0, np.log(2)]))
    self.assertEqual(list(P.euclidean_mask()), [False, False])

  def test_options(self):
    x = ca.SX.sym("x", 2)
    y = ca.SX.sym("y", 2)
    dx = ca.SX.sym("dx", 2)
    retract = ca.Function("r", [x, dx], [x + dx])
    minus = ca.Function("m", [y, x], [y - x])
    M = ca.Manifold("flat", retract, minus,
                    {"euclidean_mask": [True, False], "euclidean_mask_tangent": [False, True]})
    self.assertEqual(list(M.euclidean_mask()), [True, False])
    self.assertEqual(list(M.euclidean_mask_tangent()), [False, True])
    with self.assertRaises(Exception):
      ca.Manifold("flat", retract, minus, {"no_such_option": 1})
    with self.assertRaises(Exception):
      ca.Manifold("flat", retract, minus, {"identity": [1, 2, 3]})
    with self.assertRaises(Exception):
      ca.Manifold("flat", retract, minus, {"euclidean_mask": [True]})

  def test_check(self):
    x = ca.SX.sym("x", 2)
    y = ca.SX.sym("y", 2)
    dx = ca.SX.sym("dx", 2)
    retract = ca.Function("r", [x, dx], [x + dx])
    wrong_minus = ca.Function("m", [y, x], [x - y])
    with self.assertRaises(Exception):
      ca.Manifold("bad", retract, wrong_minus, {"identity": [0, 0]})
    # disabled check
    ca.Manifold("bad", retract, wrong_minus, {"identity": [0, 0], "check": False})
    # without identity or project there are no points to check
    ca.Manifold("bad", retract, wrong_minus)

  def test_dimension_mismatch(self):
    x = ca.SX.sym("x", 3)
    y = ca.SX.sym("y", 3)
    dx = ca.SX.sym("dx", 2)
    retract = ca.Function("r", [x, dx], [x])
    # minus must return ndx = 2 rows
    with self.assertRaises(Exception):
      ca.Manifold("bad", retract, ca.Function("m", [y, x], [y - x]))
    # minus must take (y[nx], x[nx])
    with self.assertRaises(Exception):
      ca.Manifold("bad", retract, ca.Function("m", [y, dx], [dx]))


class ManifoldSerializationTests(casadiTestCase):

  def manifolds(self):
    # type: () -> list[ca.Manifold]
    retract, minus, project = sphere_functions()
    S2 = ca.Manifold("S2", retract, minus,
                     {"project": project, "identity": [0, 0, 1],
                      "euclidean_mask_tangent": [True, False]})
    return lie_groups() + [S2, ca.Manifold.product([S2, ca.Manifold.SE3(), S2]),
                           ca.Manifold.product([])]

  def assert_equivalent(self, M, N):
    # type: (ca.Manifold, ca.Manifold) -> None
    self.assertEqual(str(M), str(N))
    self.assertEqual((M.nx(), M.ndx()), (N.nx(), N.ndx()))
    self.assertEqual(list(M.euclidean_mask()), list(N.euclidean_mask()))
    self.assertEqual(list(M.euclidean_mask_tangent()), list(N.euclidean_mask_tangent()))
    self.assertEqual(M.is_lie_group(), N.is_lie_group())
    self.assertEqual(list(M.offsets()), list(N.offsets()))
    self.checkarray(M.identity(), N.identity())
    x = M.project(ca.DM(np.random.randn(M.nx(), 1)))
    dx = ca.DM(np.random.uniform(-0.5, 0.5, (M.ndx(), 1)))
    self.checkarray(M.retract(x, dx), N.retract(x, dx))
    self.checkarray(M.minus(M.retract(x, dx), x), N.minus(M.retract(x, dx), x))

  def test_string(self):
    np.random.seed(9)
    for M in self.manifolds():
      self.assert_equivalent(M, ca.Manifold.deserialize(M.serialize()))

  def test_pickle(self):
    import pickle
    np.random.seed(10)
    for M in self.manifolds():
      N = pickle.loads(pickle.dumps(M))
      self.assertTrue(isinstance(N, ca.Manifold))
      self.assert_equivalent(M, N)

  def test_null(self):
    self.assertTrue(ca.Manifold.deserialize(ca.Manifold().serialize()).is_null())


class ManifoldJacobianTests(casadiTestCase):

  def manifolds(self):
    # type: () -> list[ca.Manifold]
    retract, minus, project = sphere_functions()
    S2 = ca.Manifold("S2", retract, minus, {"project": project})
    return lie_groups() + [S2, ca.Manifold.product([S2, ca.Manifold.SO3()])]

  def test_jac_retract_dx(self):
    np.random.seed(11)
    for M in self.manifolds():
      x = M.project(ca.DM(np.random.randn(M.nx(), 1)))
      dx = ca.SX.sym("dx", M.ndx())
      ref = ca.Function("ref", [dx], [ca.jacobian(M.retract(x, dx), dx)])
      J = M.jac_retract_dx(x)
      self.assertEqual(J.shape, (M.nx(), M.ndx()))
      self.checkarray(J, ref(ca.DM.zeros(M.ndx(), 1)), str(M))

  def test_jac_minus_y(self):
    np.random.seed(12)
    for M in self.manifolds():
      x = M.project(ca.DM(np.random.randn(M.nx(), 1)))
      y = ca.SX.sym("y", M.nx())
      ref = ca.Function("ref", [y], [ca.jacobian(M.minus(y, x), y)])
      J = M.jac_minus_y(x)
      self.assertEqual(J.shape, (M.ndx(), M.nx()))
      self.checkarray(J, ref(x), str(M))
      # minus(retract(x, dx), x) = dx  =>  jac_minus_y(x) jac_retract_dx(x) = I
      self.checkarray(ca.mtimes(J, M.jac_retract_dx(x)), ca.DM.eye(M.ndx()), str(M))

  def test_defect(self):
    """defect(y, x): zero iff y = x, first-order equal to minus around y = x"""
    np.random.seed(14)
    for M in self.manifolds():
      x = M.project(ca.DM(np.random.randn(M.nx(), 1)))
      self.checkarray(M.defect(x, x), ca.DM.zeros(M.ndx(), 1), str(M))
      dx = ca.SX.sym("dx", M.ndx())
      J = ca.Function("J", [dx], [ca.jacobian(M.defect(M.retract(x, dx), x), dx)])
      self.checkarray(J(ca.DM.zeros(M.ndx(), 1)), ca.DM.eye(M.ndx()), str(M))
      # agrees with minus up to second order
      d = ca.DM(np.random.uniform(-1, 1, (M.ndx(), 1)))*1e-4
      y = M.retract(x, d)
      self.assertTrue(float(ca.norm_inf(M.defect(y, x) - M.minus(y, x))) < 1e-6, str(M))
      # columnwise and symbolic
      X = ca.horzcat(x, y)
      self.checkarray(M.defect(X, ca.repmat(x, 1, 2)), ca.horzcat(M.defect(x, x), M.defect(y, x)))
      ys = ca.MX.sym("y", M.nx())
      f = ca.Function("f", [ys], [M.defect(ys, x)])
      self.checkarray(f(y), M.defect(y, x), str(M))
    # double cover: q and -q are the same rotation
    SO3 = ca.Manifold.SO3()
    q = SO3.exp(ca.DM([0.3, -0.2, 0.5]))
    self.checkarray(SO3.defect(-q, q), ca.DM.zeros(3, 1))

  def test_chain_rule(self):
    """Tangent-space gradient of f(retract(x, dx)) at dx = 0 from the ambient gradient"""
    np.random.seed(13)
    G = ca.Manifold.SE3()
    x = G.exp(ca.DM(np.random.randn(6, 1)))
    xs = ca.SX.sym("x", 7)
    f = ca.sumsqr(xs - ca.DM(np.arange(7)))
    grad = ca.Function("g", [xs], [ca.gradient(f, xs)])(x)
    dx = ca.SX.sym("dx", 6)
    ft = ca.substitute(f, xs, G.retract(x, dx))
    grad_t = ca.Function("gt", [dx], [ca.gradient(ft, dx)])(ca.DM.zeros(6, 1))
    self.checkarray(ca.mtimes(G.jac_retract_dx(x).T, grad), grad_t)

  def test_columnwise_jacobian(self):
    G = ca.Manifold.SO3()
    X = G.exp(ca.DM(np.random.randn(3, 3)))
    J = G.jac_retract_dx(X)
    self.assertEqual(J.shape, (4, 9))
    self.checkarray(J[:, 3:6], G.jac_retract_dx(X[:, 1]))


class ManifoldCodegenTests(casadiTestCase):

  def test_codegen(self):
    np.random.seed(14)
    for G in [ca.Manifold.SO3(), ca.Manifold.SE3()]:
      a = G.exp(ca.DM(np.random.randn(G.ndx(), 1)))
      v = ca.DM(np.random.randn(G.ndx(), 1))
      for name, args in [("retract", [a, v]), ("minus", [a, G.identity()]),
                         ("log", [a]), ("Jr_inv", [v]), ("jac_retract_dx", [a])]:
        self.check_codegen(G.get_function(name), inputs=args)


if __name__ == '__main__':
    unittest.main()
