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
                     ["retract", "minus", "project", "constraint"])
    f = M.get_function("retract")
    self.assertTrue(isinstance(f, ca.Function))
    self.assertEqual(f.name_in(), ["x", "dx"])
    self.assertEqual(f.name_out(), ["x_next"])
    # cached: same underlying Function
    self.assertTrue(M.get_function("retract").__hash__() == f.__hash__())
    with self.assertRaises(Exception):
      M.get_function("no_such_function")


if __name__ == '__main__':
    unittest.main()
