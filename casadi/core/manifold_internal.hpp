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


#ifndef CASADI_MANIFOLD_INTERNAL_HPP
#define CASADI_MANIFOLD_INTERNAL_HPP

#include "manifold.hpp"
#include "shared_object.hpp"

#include <functional>

#ifdef CASADI_WITH_THREAD
#ifdef CASADI_WITH_THREAD_MINGW
#include <mingw.mutex.h>
#else // CASADI_WITH_THREAD_MINGW
#include <mutex>
#endif // CASADI_WITH_THREAD_MINGW
#endif // CASADI_WITH_THREAD

/// \cond INTERNAL
namespace casadi {

  /** \brief Internal node of Manifold

      A concrete manifold implements its operations as SX expressions
      (the *_sx methods) acting on column vectors (or square matrices for
      vee). Those are turned into Functions on first use and cached
      (get_function).

      For Lie groups, the Jacobians (Adj, Jr, Jl and their inverses) default
      to an exact algorithmic differentiation of compose/exp/log, evaluated at
      zero perturbation. Concrete groups may override them with closed forms.
  */
  class CASADI_EXPORT ManifoldInternal : public SharedObjectInternal {
  public:
    ManifoldInternal(const std::string& name, casadi_int nx, casadi_int ndx);
    ~ManifoldInternal() override;

    /// Print a description
    void disp(std::ostream& stream, bool more) const override;

    /// Short description, e.g. "Euclidean(3)"
    virtual std::string repr() const = 0;

    /// \name Information
    ///@{
    const std::string& name() const {return name_;}
    casadi_int nx() const {return nx_;}
    casadi_int ndx() const {return ndx_;}
    virtual bool is_euclidean() const {return false;}
    virtual bool is_lie_group() const {return false;}
    virtual std::vector<bool> euclidean_mask() const;
    virtual std::vector<bool> euclidean_mask_tangent() const;
    virtual casadi_int n_components() const {return 1;}
    virtual Manifold component(casadi_int i) const;
    virtual std::vector<casadi_int> offsets() const;
    virtual std::vector<casadi_int> offsets_tangent() const;
    virtual DM identity() const = 0;
    /// Dimension of the points the group acts on (act)
    virtual casadi_int point_dim() const;
    /// Size of the matrix representation (hat, as_matrix); -1 if none
    virtual casadi_int matrix_dim() const {return -1;}
    ///@}

    /// \name Symbolic primitives: manifold operations
    ///@{
    virtual SX retract_sx(const SX& x, const SX& dx) const = 0;
    virtual SX minus_sx(const SX& y, const SX& x) const = 0;
    virtual SX defect_sx(const SX& y, const SX& x) const { return minus_sx(y, x);}
    virtual SX project_sx(const SX& x) const;
    virtual SX constraint_sx(const SX& x) const;
    SX jac_retract_dx_sx(const SX& x) const;
    SX jac_minus_y_sx(const SX& x) const;
    ///@}

    /// \name Symbolic primitives: Lie group operations
    ///@{
    virtual SX compose_sx(const SX& a, const SX& b) const;
    virtual SX inverse_sx(const SX& a) const;
    virtual SX exp_sx(const SX& v) const;
    virtual SX log_sx(const SX& a) const;
    virtual SX act_sx(const SX& a, const SX& p) const;
    virtual SX hat_sx(const SX& v) const;
    virtual SX vee_sx(const SX& V) const;
    virtual SX as_matrix_sx(const SX& a) const;
    virtual SX Adj_sx(const SX& a) const;
    virtual SX Jr_sx(const SX& v) const;
    virtual SX Jl_sx(const SX& v) const;
    virtual SX Jr_inv_sx(const SX& v) const;
    virtual SX Jl_inv_sx(const SX& v) const;
    ///@}

    /// Get a cached Function implementing an operation
    Function get_function(const std::string& fname) const;

    /// Names accepted by get_function
    static std::vector<std::string> function_names();

    /// Initialization after the node is owned by a Manifold
    virtual void init_custom() {}

    /// Serialize (type tag followed by the body)
    void serialize(SerializingStream& s) const;

    /// Serialize the type-specific data
    virtual void serialize_body(SerializingStream& s) const {}

    /// Deserialize with type disambiguation
    static ManifoldInternal* deserialize(DeserializingStream& s);

    /// Is fname a Lie group operation?
    static bool is_lie_function(const std::string& fname);

  protected:
    /// Create the Function implementing an operation (not cached)
    virtual Function create_function(const std::string& fname) const;

    /// Throw an error for an unsupported Lie group operation
    [[noreturn]] void lie_error(const std::string& op) const;

    /// Name
    std::string name_;

    /// Dimensions
    casadi_int nx_, ndx_;

  private:
    /// Cache of created Functions
    mutable std::map<std::string, Function> cache_;
#ifdef CASADI_WITH_THREAD
    mutable std::mutex mtx_;
#endif // CASADI_WITH_THREAD
  };

  /** \brief Base class for Lie groups

      retract(x, dx) = compose(x, exp(dx)), minus(y, x) = log(compose(inverse(x), y))
  */
  class CASADI_EXPORT LieGroupInternal : public ManifoldInternal {
  public:
    using ManifoldInternal::ManifoldInternal;
    bool is_lie_group() const override {return true;}
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
    SX defect_sx(const SX& y, const SX& x) const override;
    /// Cheap chart around the identity with the same first-order behavior as log
    virtual SX log_approx_sx(const SX& a) const { return log_sx(a);}
  };

  /** \brief Euclidean space R^n (additive group) */
  class CASADI_EXPORT EuclideanManifold : public LieGroupInternal {
  public:
    explicit EuclideanManifold(casadi_int n);
    void serialize_body(SerializingStream& s) const override;
    std::string class_name() const override {return "EuclideanManifold";}
    std::string repr() const override;
    bool is_euclidean() const override {return true;}
    std::vector<bool> euclidean_mask() const override;
    std::vector<bool> euclidean_mask_tangent() const override;
    DM identity() const override;
    casadi_int matrix_dim() const override {return nx_ + 1;}
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX hat_sx(const SX& v) const override;
    SX vee_sx(const SX& V) const override;
    SX as_matrix_sx(const SX& a) const override;
    SX Adj_sx(const SX& a) const override;
    SX Jr_sx(const SX& v) const override;
    SX Jl_sx(const SX& v) const override;
    SX Jr_inv_sx(const SX& v) const override;
    SX Jl_inv_sx(const SX& v) const override;
  };

  /** \brief SO(3) as unit quaternions [w, x, y, z] (Hamilton) */
  class CASADI_EXPORT SO3Manifold : public LieGroupInternal {
  public:
    SO3Manifold();
    std::string class_name() const override {return "SO3Manifold";}
    std::string repr() const override {return "SO3";}
    DM identity() const override;
    casadi_int point_dim() const override {return 3;}
    casadi_int matrix_dim() const override {return 3;}
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX log_approx_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX hat_sx(const SX& v) const override;
    SX vee_sx(const SX& V) const override;
    SX as_matrix_sx(const SX& a) const override;
    SX Adj_sx(const SX& a) const override;
    SX Jr_sx(const SX& v) const override;
    SX Jl_sx(const SX& v) const override;
    SX Jr_inv_sx(const SX& v) const override;
    SX Jl_inv_sx(const SX& v) const override;
  };

  /** \brief SO(2) as unit complex numbers [cos(theta), sin(theta)] */
  class CASADI_EXPORT SO2Manifold : public LieGroupInternal {
  public:
    SO2Manifold();
    std::string class_name() const override {return "SO2Manifold";}
    std::string repr() const override {return "SO2";}
    DM identity() const override;
    casadi_int point_dim() const override {return 2;}
    casadi_int matrix_dim() const override {return 2;}
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX log_approx_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX hat_sx(const SX& v) const override;
    SX vee_sx(const SX& V) const override;
    SX as_matrix_sx(const SX& a) const override;
    SX Adj_sx(const SX& a) const override;
    SX Jr_sx(const SX& v) const override;
    SX Jl_sx(const SX& v) const override;
    SX Jr_inv_sx(const SX& v) const override;
    SX Jl_inv_sx(const SX& v) const override;
  };

  /** \brief SE(2) as [x, y, cos(theta), sin(theta)], tangent [rho_x, rho_y, theta] */
  class CASADI_EXPORT SE2Manifold : public LieGroupInternal {
  public:
    SE2Manifold();
    std::string class_name() const override {return "SE2Manifold";}
    std::string repr() const override {return "SE2";}
    DM identity() const override;
    casadi_int point_dim() const override {return 2;}
    casadi_int matrix_dim() const override {return 3;}
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX log_approx_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX hat_sx(const SX& v) const override;
    SX vee_sx(const SX& V) const override;
    SX as_matrix_sx(const SX& a) const override;
  };

  /** \brief SE(3) as [p (3); q (4, quaternion [w, x, y, z])], tangent [rho (3); theta (3)] */
  class CASADI_EXPORT SE3Manifold : public LieGroupInternal {
  public:
    SE3Manifold();
    std::string class_name() const override {return "SE3Manifold";}
    std::string repr() const override {return "SE3";}
    DM identity() const override;
    casadi_int point_dim() const override {return 3;}
    casadi_int matrix_dim() const override {return 4;}
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX log_approx_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX hat_sx(const SX& v) const override;
    SX vee_sx(const SX& V) const override;
    SX as_matrix_sx(const SX& a) const override;
  };

  /** \brief User-defined manifold from retract and minus Functions */
  class CASADI_EXPORT CustomManifold : public ManifoldInternal {
  public:
    CustomManifold(const std::string& name, const Function& retract, const Function& minus,
                   const Dict& opts);
    std::string class_name() const override {return "CustomManifold";}
    std::string repr() const override {return name_;}
    std::vector<bool> euclidean_mask() const override {return mask_;}
    std::vector<bool> euclidean_mask_tangent() const override {return mask_tan_;}
    DM identity() const override {return identity_;}
    void init_custom() override;
    void serialize_body(SerializingStream& s) const override;
    static ManifoldInternal* deserialize(DeserializingStream& s);
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
  protected:
    /// Numerical consistency check
    void check() const;
    Function retract_, minus_, project_, constraint_;
    DM identity_;
    bool has_identity_, check_;
    std::vector<bool> mask_, mask_tan_;
  };

  /** \brief Cartesian product of manifolds */
  class CASADI_EXPORT ProductManifold : public ManifoldInternal {
  public:
    explicit ProductManifold(const std::vector<Manifold>& components);
    void serialize_body(SerializingStream& s) const override;
    std::string class_name() const override {return "ProductManifold";}
    std::string repr() const override;
    bool is_euclidean() const override;
    bool is_lie_group() const override;
    std::vector<bool> euclidean_mask() const override;
    std::vector<bool> euclidean_mask_tangent() const override;
    casadi_int n_components() const override {return comp_.size();}
    Manifold component(casadi_int i) const override;
    std::vector<casadi_int> offsets() const override {return off_;}
    std::vector<casadi_int> offsets_tangent() const override {return off_tan_;}
    DM identity() const override;
    casadi_int point_dim() const override;
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
    SX defect_sx(const SX& y, const SX& x) const override;
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
    SX compose_sx(const SX& a, const SX& b) const override;
    SX inverse_sx(const SX& a) const override;
    SX exp_sx(const SX& v) const override;
    SX log_sx(const SX& a) const override;
    SX act_sx(const SX& a, const SX& p) const override;
    SX Adj_sx(const SX& a) const override;
    SX Jr_sx(const SX& v) const override;
    SX Jl_sx(const SX& v) const override;
    SX Jr_inv_sx(const SX& v) const override;
    SX Jl_inv_sx(const SX& v) const override;
  private:
    /// Offsets of the points acted upon
    std::vector<casadi_int> point_offsets() const;
    std::vector<Manifold> comp_;
    std::vector<casadi_int> off_, off_tan_;
  };

  /// \name Numerically safe scalar helpers (exposed for testing)
  ///@{
  /** \brief Evaluate f(sqrt(t2)) with a Taylor series in t2 for t2 < thr

      The exact branch is evaluated at a safe argument for small t2, so that
      neither the value nor any derivative produces NaN/Inf.
      c holds the series coefficients in t2 (lowest order first).
  */
  CASADI_EXPORT SX manifold_series_switch(const SX& t2, double thr,
    const std::vector<double>& c, const std::function<SX(const SX&)>& exact);
  /// Skew-symmetric matrix [v]x of a 3-vector
  CASADI_EXPORT SX manifold_skew(const SX& v);
  ///@}

} // namespace casadi
/// \endcond

#endif // CASADI_MANIFOLD_INTERNAL_HPP
