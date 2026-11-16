/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#ifndef CASADI_MANIFOLD_HPP
#define CASADI_MANIFOLD_HPP

#include "function.hpp"
#include "printable.hpp"

namespace casadi {

  // Forward declaration of internal class
  class ManifoldInternal;
  class SerializingStream;
  class DeserializingStream;

  /** \brief Smooth manifold (optionally a Lie group) with a retraction

      A Manifold describes a set whose elements are stored in an ambient
      representation of dimension nx (e.g. a unit quaternion, nx = 4), while
      local variations live in a tangent space of dimension ndx (e.g. so(3),
      ndx = 3). It lets optimization variables such as orientations be
      updated in the tangent space without leaving the manifold.

      The two primitive operations are
      - retract(x, dx): move from x along the tangent vector dx (x (+) dx)
      - minus(y, x):    tangent vector dx such that retract(x, dx) = y (y (-) x)

      Built-in manifolds: Euclidean(n), SO2(), SO3(), SE2(), SE3(), Cartesian
      products of manifolds (product) and user-defined manifolds (constructor
      taking retract and minus Functions). All built-in manifolds are Lie
      groups, as are products of Lie groups; for those, compose, inverse, exp,
      log, act, hat, vee, as_matrix, Adj and the right/left Jacobians Jr, Jl
      (and their inverses) are available.

      Conventions shared by all built-in Lie groups:
      - right perturbation: retract(x, dx) = x * Exp(dx), minus(y, x) = Log(x^-1 * y)
      - quaternions are Hamilton quaternions stored [w, x, y, z] (scalar first)
      - the tangent vector of SE(2)/SE(3) is [translation; rotation]

      All operations accept DM (numeric), SX and MX arguments. Arguments with
      K > 1 columns are processed column-wise and an argument with a single
      column is broadcast; matrix-valued results (e.g. Jr) are stacked
      horizontally. Every operation is a casadi::Function (see get_function),
      so it can be differentiated, code-generated and serialized. Near
      singular points (e.g. zero rotation) Taylor series are used, so values
      and derivatives up to second order stay finite.

      \lang{python}
      \doctest
      >>> SO3 = Manifold.SO3()
      >>> SO3.nx(), SO3.ndx(), SO3.is_lie_group()
      (4, 3, True)
      >>> q = SO3.exp(DM([0, 0, pi/2]))
      >>> SO3.act(q, DM([1, 0, 0]))
      DM([0, 1, 0])
      >>> SO3.minus(SO3.retract(q, DM([0.1, 0, 0])), q)
      DM([0.1, 0, 0])
      >>> M = Manifold.product([Manifold.Euclidean(3), SO3])
      >>> M, M.nx(), M.ndx(), M.euclidean_mask()
      (Manifold(Product(Euclidean(3), SO3)), 7, 6, [True, True, True, False, False, False, False])
      >>> x = MX.sym("x", 7); dx = MX.sym("dx", 6)
      >>> f = Function("f", [x, dx], [M.retract(x, dx)])
      >>> f(M.identity(), DM([1, 2, 3, 0, 0, 0]))
      DM([1, 2, 3, 1, 0, 0, 0])
      \enddoctest
      \endlang
  */
  class CASADI_EXPORT Manifold
    : public SharedObject,
      public SWIG_IF_ELSE(PrintableCommon, Printable<Manifold>) {
  public:
    /// Default constructor (null object)
    Manifold();

    /** \brief User-defined manifold

        \param name     Name, used as prefix of the generated Functions
        \param retract  Function (x[nx], dx[ndx]) -> (x_next[nx])
        \param minus    Function (y[nx], x[nx]) -> (dx[ndx]), the local inverse
                        of retract: retract(x, minus(y, x)) = y
        \param opts     Options:
          - project (Function: x[nx] -> x_proj[nx]): projection onto the manifold
          - constraint (Function: x[nx] -> c[m]): embedding constraint c(x) = 0
          - identity (vector of length nx): reference element (default: zeros)
          - euclidean_mask (bool vector of length nx, default: all false)
          - euclidean_mask_tangent (bool vector of length ndx, default: all false)
          - check (bool, default true): check numerically, at the identity
            (if given) and at projected random points (if project is given),
            that retract(x, 0) = x, minus(x, x) = 0 and
            minus(retract(x, dx), x) = dx for small dx

        The Functions must support evaluation with SX arguments (e.g. any
        SX Function, or an MX Function built from standard operations).
        A user-defined manifold is not a Lie group.
    */
    Manifold(const std::string& name, const Function& retract, const Function& minus,
             const Dict& opts=Dict());

    /// Readable name of the class
    static std::string type_name() {return "Manifold";}

    /// Check if a particular cast is allowed
    static bool test_cast(const SharedObjectInternal* ptr);

#ifndef SWIG
    /// Create from node
    static Manifold create(ManifoldInternal* node);

    /// Access functions of the node
    ManifoldInternal* get() const;
    const ManifoldInternal* operator->() const;
    ManifoldInternal* operator->();
#endif // SWIG

    /// \name Built-in manifolds
    ///@{
    /** \brief Euclidean space R^n (nx = ndx = n), retract(x, dx) = x + dx */
    static Manifold Euclidean(casadi_int n);

    /** \brief Cartesian product of manifolds

        The ambient vector is the vertical concatenation of the component
        ambient vectors, likewise for the tangent vector.
    */
    static Manifold product(const std::vector<Manifold>& components);

    /** \brief Rotation group SO(3) as unit quaternions

        Ambient representation [w, x, y, z] (Hamilton convention, scalar first),
        nx = 4, ndx = 3. Note that Eigen and scipy store [x, y, z, w].
    */
    static Manifold SO3();

    /** \brief Rotation group SO(2) as unit complex numbers

        Ambient representation [cos(theta), sin(theta)], nx = 2, ndx = 1.
    */
    static Manifold SO2();

    /** \brief Rigid motions in the plane SE(2)

        Ambient representation [x, y, cos(theta), sin(theta)], nx = 4.
        Tangent [rho_x, rho_y, theta], ndx = 3 (translation first).
    */
    static Manifold SE2();

    /** \brief Rigid motions in space SE(3)

        Ambient representation [p (3); q (4)] with q a unit quaternion
        [w, x, y, z], nx = 7. Tangent [rho (3); theta (3)], ndx = 6
        (translation first). This is the true group SE(3), in which
        translation and rotation are coupled by retract; for independent
        perturbations use product([Euclidean(3), SO3()]).
    */
    static Manifold SE3();
    ///@}

    /// \name Information
    ///@{
    /// Name of the manifold (also used as prefix for the generated Functions)
    std::string name() const;

    /// Dimension of the ambient representation
    casadi_int nx() const;

    /// Dimension of the tangent space
    casadi_int ndx() const;

    /// Is the manifold (globally) Euclidean, i.e. retract(x, dx) = x + dx ?
    bool is_euclidean() const;

    /// Is the manifold a Lie group (are the Lie group operations available)?
    bool is_lie_group() const;

    /** \brief Which entries of the ambient vector are Euclidean (length nx)

        Only these entries may carry simple bounds in an optimization problem.
    */
    std::vector<bool> euclidean_mask() const;

    /// Which entries of the tangent vector are Euclidean (length ndx)
    std::vector<bool> euclidean_mask_tangent() const;

    /// Number of components (1 unless created with product)
    casadi_int n_components() const;

    /// Get a component (the manifold itself unless created with product)
    Manifold component(casadi_int i) const;

    /// Offsets of the components in the ambient vector (length n_components()+1)
    std::vector<casadi_int> offsets() const;

    /// Offsets of the components in the tangent vector (length n_components()+1)
    std::vector<casadi_int> offsets_tangent() const;

    /// Reference element (identity for Lie groups, zero for Euclidean spaces)
    DM identity() const;
    ///@}

    /// \name Manifold operations
    ///@{
    /** \brief Retraction: x (+) dx

        x: nx-by-K, dx: ndx-by-K, result: nx-by-K
    */
    DM retract(const DM& x, const DM& dx) const;
    SX retract(const SX& x, const SX& dx) const;
    MX retract(const MX& x, const MX& dx) const;

    /** \brief Local difference: y (-) x, i.e. dx with retract(x, dx) = y

        y: nx-by-K, x: nx-by-K, result: ndx-by-K
    */
    DM minus(const DM& y, const DM& x) const;
    SX minus(const SX& y, const SX& x) const;
    MX minus(const MX& y, const MX& x) const;

    /** \brief Cheap residual between two points, for equality constraints y = x

        A smooth function of (y, x) of dimension ndx that agrees with minus(y, x)
        to first order around y = x (its Jacobian with respect to y at y = x equals
        jac_minus_y(x)) but avoids the logarithm: for the built-in Lie groups it is the
        vector part of x^-1 * y in a chart without trigonometric functions, e.g.
        2*vec(q_x^* q_y) for SO(3) and sin(theta_y - theta_x) for SO(2). It vanishes
        exactly when y and x are the same element (for SO(3)/SE(3) also when the
        quaternions differ only in sign). Away from y = x it differs from minus; use it
        as a constraint residual (e.g. multiple shooting defects), not as a distance.
        Equal to minus for Euclidean spaces and user-defined manifolds.

        y: nx-by-K, x: nx-by-K, result: ndx-by-K
    */
    DM defect(const DM& y, const DM& x) const;
    SX defect(const SX& y, const SX& x) const;
    MX defect(const MX& y, const MX& x) const;

    /** \brief Projection of an ambient vector onto the manifold

        E.g. normalization of a quaternion. Identity for Euclidean spaces.
    */
    DM project(const DM& x) const;
    SX project(const SX& x) const;
    MX project(const MX& x) const;

    /** \brief Embedding constraint c(x) = 0 that characterizes the manifold

        E.g. |q|^2 - 1 for a unit quaternion. Has zero rows for Euclidean spaces.
    */
    DM constraint(const DM& x) const;
    SX constraint(const SX& x) const;
    MX constraint(const MX& x) const;

    /** \brief Jacobian of retract(x, dx) with respect to dx at dx = 0 (nx-by-ndx)

        Maps tangent perturbations to ambient perturbations; used to pull
        ambient-space derivatives back to the tangent space. */
    DM jac_retract_dx(const DM& x) const;
    SX jac_retract_dx(const SX& x) const;
    MX jac_retract_dx(const MX& x) const;

    /** \brief Jacobian of minus(y, x) with respect to y at y = x (ndx-by-nx) */
    DM jac_minus_y(const DM& x) const;
    SX jac_minus_y(const SX& x) const;
    MX jac_minus_y(const MX& x) const;
    ///@}

    /// \name Lie group operations (only if is_lie_group())
    ///@{
    /** \brief Group composition a * b */
    DM compose(const DM& a, const DM& b) const;
    SX compose(const SX& a, const SX& b) const;
    MX compose(const MX& a, const MX& b) const;

    /** \brief Group inverse a^-1 */
    DM inverse(const DM& a) const;
    SX inverse(const SX& a) const;
    MX inverse(const MX& a) const;

    /** \brief Exponential map Exp: tangent vector (ndx) -> group element (nx) */
    DM exp(const DM& v) const;
    SX exp(const SX& v) const;
    MX exp(const MX& v) const;

    /** \brief Logarithm map Log: group element (nx) -> tangent vector (ndx)

        For SO3, the shortest rotation is returned (q and -q are identified). */
    DM log(const DM& a) const;
    SX log(const SX& a) const;
    MX log(const MX& a) const;

    /** \brief Action of a on a point p (SO3: R p, SE3: R p + t, R^n: p + a) */
    DM act(const DM& a, const DM& p) const;
    SX act(const SX& a, const SX& p) const;
    MX act(const MX& a, const MX& p) const;

    /** \brief Tangent vector to Lie algebra matrix (e.g. skew-symmetric matrix for SO3) */
    DM hat(const DM& v) const;
    SX hat(const SX& v) const;
    MX hat(const MX& v) const;

    /** \brief Lie algebra matrix to tangent vector (inverse of hat) */
    DM vee(const DM& V) const;
    SX vee(const SX& V) const;
    MX vee(const MX& V) const;

    /** \brief Matrix representation of a group element (SO3: 3x3, SE3: 4x4) */
    DM as_matrix(const DM& a) const;
    SX as_matrix(const SX& a) const;
    MX as_matrix(const MX& a) const;

    /** \brief Adjoint matrix: a * Exp(v) = Exp(Adj(a) v) * a */
    DM Adj(const DM& a) const;
    SX Adj(const SX& a) const;
    MX Adj(const MX& a) const;

    /** \brief Right Jacobian: Exp(v + e) = Exp(v) * Exp(Jr(v) e) + O(|e|^2) */
    DM Jr(const DM& v) const;
    SX Jr(const SX& v) const;
    MX Jr(const MX& v) const;

    /** \brief Left Jacobian: Exp(v + e) = Exp(Jl(v) e) * Exp(v) + O(|e|^2) */
    DM Jl(const DM& v) const;
    SX Jl(const SX& v) const;
    MX Jl(const MX& v) const;

    /** \brief Inverse of the right Jacobian */
    DM Jr_inv(const DM& v) const;
    SX Jr_inv(const SX& v) const;
    MX Jr_inv(const MX& v) const;

    /** \brief Inverse of the left Jacobian */
    DM Jl_inv(const DM& v) const;
    SX Jl_inv(const SX& v) const;
    MX Jl_inv(const MX& v) const;
    ///@}


    /** \brief Get the Function implementing an operation

        Available names: see function_names().
        The Functions are created on first use and cached.
    */
    Function get_function(const std::string& name) const;

    /// Names accepted by get_function
    static std::vector<std::string> function_names();

    /// Serialize to a string (see deserialize)
    std::string serialize() const;

    /// Create from a string produced by serialize
    static Manifold deserialize(const std::string& s);

#ifndef SWIG
    /// Serialize an object
    void serialize(SerializingStream& s) const;

    /// Deserialize an object
    static Manifold deserialize(DeserializingStream& s);
#endif // SWIG
  };

} // namespace casadi

#endif // CASADI_MANIFOLD_HPP
