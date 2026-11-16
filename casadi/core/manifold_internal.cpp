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


#include "manifold_internal.hpp"
#include "serializing_stream.hpp"

#include <algorithm>

namespace casadi {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  SX manifold_series_switch(const SX& t2, double thr, const std::vector<double>& c,
                            const std::function<SX(const SX&)>& exact) {
    SX small = t2 < thr;
    // Safe argument: the exact branch never sees the singular point
    SX t2_safe = if_else(small, SX(thr), t2);
    SX series = 0;
    for (auto it = c.rbegin(); it != c.rend(); ++it) series = series * t2 + *it;
    return if_else(small, series, exact(sqrt(t2_safe)));
  }

  SX manifold_skew(const SX& v) {
    casadi_assert_dev(v.size1()==3 && v.size2()==1);
    SX r = SX::zeros(3, 3);
    r(0, 1) = -v(2); r(0, 2) = v(1);
    r(1, 0) = v(2);  r(1, 2) = -v(0);
    r(2, 0) = -v(1); r(2, 1) = v(0);
    return r;
  }

  // ---------------------------------------------------------------------------
  // ManifoldInternal
  // ---------------------------------------------------------------------------

  ManifoldInternal::ManifoldInternal(const std::string& name, casadi_int nx, casadi_int ndx)
      : name_(name), nx_(nx), ndx_(ndx) {
    casadi_assert(nx>=0 && ndx>=0,
      "Manifold dimensions must be nonnegative, got nx=" + str(nx) + ", ndx=" + str(ndx));
    casadi_assert(ndx<=nx,
      "Manifold tangent dimension (" + str(ndx) + ") cannot exceed "
      "the ambient dimension (" + str(nx) + ")");
  }

  ManifoldInternal::~ManifoldInternal() {
  }

  void ManifoldInternal::disp(std::ostream& stream, bool more) const {
    stream << repr();
    if (more) stream << " [nx=" << nx_ << ", ndx=" << ndx_ << "]";
  }

  std::vector<bool> ManifoldInternal::euclidean_mask() const {
    return std::vector<bool>(nx_, false);
  }

  std::vector<bool> ManifoldInternal::euclidean_mask_tangent() const {
    return std::vector<bool>(ndx_, false);
  }

  Manifold ManifoldInternal::component(casadi_int i) const {
    casadi_assert_dev(i==0);
    return shared_from_this<Manifold>();
  }

  std::vector<casadi_int> ManifoldInternal::offsets() const {
    return {0, nx_};
  }

  std::vector<casadi_int> ManifoldInternal::offsets_tangent() const {
    return {0, ndx_};
  }

  casadi_int ManifoldInternal::point_dim() const {
    return nx_;
  }

  SX ManifoldInternal::project_sx(const SX& x) const {
    return x;
  }

  SX ManifoldInternal::constraint_sx(const SX& x) const {
    return SX(0, 1);
  }

  SX ManifoldInternal::jac_retract_dx_sx(const SX& x) const {
    SX dx = SX::sym("dx", ndx_);
    SX J = jacobian(retract_sx(x, dx), dx);
    return substitute(J, dx, SX::zeros(ndx_));
  }

  SX ManifoldInternal::jac_minus_y_sx(const SX& x) const {
    SX y = SX::sym("y", nx_);
    SX J = jacobian(minus_sx(y, x), y);
    return substitute(J, y, x);
  }

  void ManifoldInternal::lie_error(const std::string& op) const {
    if (!is_lie_group()) {
      casadi_error("Manifold::" + op + ": " + repr() + " is not a Lie group.");
    }
    casadi_error("Manifold::" + op + ": not available for " + repr() + ".");
  }

  SX ManifoldInternal::compose_sx(const SX& a, const SX& b) const { lie_error("compose");}
  SX ManifoldInternal::inverse_sx(const SX& a) const { lie_error("inverse");}
  SX ManifoldInternal::exp_sx(const SX& v) const { lie_error("exp");}
  SX ManifoldInternal::log_sx(const SX& a) const { lie_error("log");}
  SX ManifoldInternal::act_sx(const SX& a, const SX& p) const { lie_error("act");}
  SX ManifoldInternal::hat_sx(const SX& v) const { lie_error("hat");}
  SX ManifoldInternal::vee_sx(const SX& V) const { lie_error("vee");}
  SX ManifoldInternal::as_matrix_sx(const SX& a) const { lie_error("as_matrix");}

  SX ManifoldInternal::Adj_sx(const SX& a) const {
    if (!is_lie_group()) lie_error("Adj");
    // Adj(a) v = Log(a Exp(v) a^-1), linearized at v = 0
    SX v = SX::sym("v", ndx_);
    SX e = log_sx(compose_sx(compose_sx(a, exp_sx(v)), inverse_sx(a)));
    return substitute(jacobian(e, v), v, SX::zeros(ndx_));
  }

  SX ManifoldInternal::Jr_sx(const SX& v) const {
    if (!is_lie_group()) lie_error("Jr");
    // Exp(v + e) = Exp(v) Exp(Jr(v) e) + O(e^2)
    SX e = SX::sym("e", ndx_);
    SX d = log_sx(compose_sx(inverse_sx(exp_sx(v)), exp_sx(v + e)));
    return substitute(jacobian(d, e), e, SX::zeros(ndx_));
  }

  SX ManifoldInternal::Jl_sx(const SX& v) const {
    if (!is_lie_group()) lie_error("Jl");
    return Jr_sx(-v);
  }

  SX ManifoldInternal::Jr_inv_sx(const SX& v) const {
    if (!is_lie_group()) lie_error("Jr_inv");
    // Log(Exp(v) Exp(e)) = v + Jr(v)^-1 e + O(e^2)
    SX e = SX::sym("e", ndx_);
    SX d = log_sx(compose_sx(exp_sx(v), exp_sx(e)));
    return substitute(jacobian(d, e), e, SX::zeros(ndx_));
  }

  SX ManifoldInternal::Jl_inv_sx(const SX& v) const {
    if (!is_lie_group()) lie_error("Jl_inv");
    return Jr_inv_sx(-v);
  }

  std::vector<std::string> ManifoldInternal::function_names() {
    return {"retract", "minus", "project", "constraint", "jac_retract_dx", "jac_minus_y",
            "compose", "inverse", "exp", "log", "act", "hat", "vee", "as_matrix",
            "Adj", "Jr", "Jl", "Jr_inv", "Jl_inv"};
  }

  bool ManifoldInternal::is_lie_function(const std::string& fname) {
    std::vector<std::string> n = function_names();
    return std::find(n.begin() + 6, n.end(), fname) != n.end();
  }

  Function ManifoldInternal::create_function(const std::string& fname) const {
    std::string fn = name_ + "_" + fname;
    SX x = SX::sym("x", nx_);
    SX a = SX::sym("a", nx_);
    SX v = SX::sym("v", ndx_);
    if (fname=="retract") {
      SX dx = SX::sym("dx", ndx_);
      return Function(fn, {x, dx}, {retract_sx(x, dx)}, {"x", "dx"}, {"x_next"});
    } else if (fname=="minus") {
      SX y = SX::sym("y", nx_);
      return Function(fn, {y, x}, {minus_sx(y, x)}, {"y", "x"}, {"dx"});
    } else if (fname=="project") {
      return Function(fn, {x}, {project_sx(x)}, {"x"}, {"x_proj"});
    } else if (fname=="constraint") {
      return Function(fn, {x}, {constraint_sx(x)}, {"x"}, {"c"});
    } else if (fname=="jac_retract_dx") {
      return Function(fn, {x}, {jac_retract_dx_sx(x)}, {"x"}, {"jac"});
    } else if (fname=="jac_minus_y") {
      return Function(fn, {x}, {jac_minus_y_sx(x)}, {"x"}, {"jac"});
    } else if (fname=="compose") {
      SX b = SX::sym("b", nx_);
      return Function(fn, {a, b}, {compose_sx(a, b)}, {"a", "b"}, {"ab"});
    } else if (fname=="inverse") {
      return Function(fn, {a}, {inverse_sx(a)}, {"a"}, {"a_inv"});
    } else if (fname=="exp") {
      return Function(fn, {v}, {exp_sx(v)}, {"v"}, {"a"});
    } else if (fname=="log") {
      return Function(fn, {a}, {log_sx(a)}, {"a"}, {"v"});
    } else if (fname=="act") {
      SX p = SX::sym("p", point_dim());
      return Function(fn, {a, p}, {act_sx(a, p)}, {"a", "p"}, {"ap"});
    } else if (fname=="hat") {
      return Function(fn, {v}, {hat_sx(v)}, {"v"}, {"V"});
    } else if (fname=="vee") {
      casadi_int m = matrix_dim();
      if (m<0) lie_error("vee");
      SX V = SX::sym("V", m, m);
      return Function(fn, {V}, {vee_sx(V)}, {"V"}, {"v"});
    } else if (fname=="as_matrix") {
      return Function(fn, {a}, {as_matrix_sx(a)}, {"a"}, {"A"});
    } else if (fname=="Adj") {
      return Function(fn, {a}, {Adj_sx(a)}, {"a"}, {"Adj"});
    } else if (fname=="Jr") {
      return Function(fn, {v}, {Jr_sx(v)}, {"v"}, {"Jr"});
    } else if (fname=="Jl") {
      return Function(fn, {v}, {Jl_sx(v)}, {"v"}, {"Jl"});
    } else if (fname=="Jr_inv") {
      return Function(fn, {v}, {Jr_inv_sx(v)}, {"v"}, {"Jr_inv"});
    } else if (fname=="Jl_inv") {
      return Function(fn, {v}, {Jl_inv_sx(v)}, {"v"}, {"Jl_inv"});
    }
    casadi_error("Manifold::get_function: unknown function '" + fname
                 + "'. Available: " + str(function_names()));
  }

  Function ManifoldInternal::get_function(const std::string& fname) const {
    if (is_lie_function(fname) && !is_lie_group()) lie_error(fname);
#ifdef CASADI_WITH_THREAD
    std::lock_guard<std::mutex> lock(mtx_);
#endif // CASADI_WITH_THREAD
    auto it = cache_.find(fname);
    if (it!=cache_.end()) return it->second;
    Function f = create_function(fname);
    cache_[fname] = f;
    return f;
  }

  void ManifoldInternal::serialize(SerializingStream& s) const {
    s.version("Manifold", 1);
    s.pack("Manifold::type", class_name());
    serialize_body(s);
  }

  ManifoldInternal* ManifoldInternal::deserialize(DeserializingStream& s) {
    s.version("Manifold", 1);
    std::string type;
    s.unpack("Manifold::type", type);
    if (type=="EuclideanManifold") {
      casadi_int n;
      s.unpack("EuclideanManifold::n", n);
      return new EuclideanManifold(n);
    } else if (type=="SO2Manifold") {
      return new SO2Manifold();
    } else if (type=="SO3Manifold") {
      return new SO3Manifold();
    } else if (type=="SE2Manifold") {
      return new SE2Manifold();
    } else if (type=="SE3Manifold") {
      return new SE3Manifold();
    } else if (type=="ProductManifold") {
      std::vector<Manifold> c;
      s.unpack("ProductManifold::components", c);
      return new ProductManifold(c);
    } else if (type=="CustomManifold") {
      return CustomManifold::deserialize(s);
    }
    casadi_error("Manifold::deserialize: unknown type '" + type + "'.");
  }

  // ---------------------------------------------------------------------------
  // LieGroupInternal
  // ---------------------------------------------------------------------------

  SX LieGroupInternal::retract_sx(const SX& x, const SX& dx) const {
    return compose_sx(x, exp_sx(dx));
  }

  SX LieGroupInternal::minus_sx(const SX& y, const SX& x) const {
    return log_sx(compose_sx(inverse_sx(x), y));
  }

  // ---------------------------------------------------------------------------
  // EuclideanManifold
  // ---------------------------------------------------------------------------

  EuclideanManifold::EuclideanManifold(casadi_int n)
      : LieGroupInternal("R" + str(n), n, n) {
  }

  void EuclideanManifold::serialize_body(SerializingStream& s) const {
    s.pack("EuclideanManifold::n", nx_);
  }

  std::string EuclideanManifold::repr() const {
    return "Euclidean(" + str(nx_) + ")";
  }

  std::vector<bool> EuclideanManifold::euclidean_mask() const {
    return std::vector<bool>(nx_, true);
  }

  std::vector<bool> EuclideanManifold::euclidean_mask_tangent() const {
    return std::vector<bool>(ndx_, true);
  }

  DM EuclideanManifold::identity() const {
    return DM::zeros(nx_, 1);
  }

  SX EuclideanManifold::retract_sx(const SX& x, const SX& dx) const { return x + dx;}
  SX EuclideanManifold::minus_sx(const SX& y, const SX& x) const { return y - x;}
  SX EuclideanManifold::compose_sx(const SX& a, const SX& b) const { return a + b;}
  SX EuclideanManifold::inverse_sx(const SX& a) const { return -a;}
  SX EuclideanManifold::exp_sx(const SX& v) const { return v;}
  SX EuclideanManifold::log_sx(const SX& a) const { return a;}
  SX EuclideanManifold::act_sx(const SX& a, const SX& p) const { return p + a;}

  SX EuclideanManifold::hat_sx(const SX& v) const {
    SX r = SX::zeros(nx_ + 1, nx_ + 1);
    r(Slice(0, nx_), nx_) = v;
    return r;
  }

  SX EuclideanManifold::vee_sx(const SX& V) const {
    return V(Slice(0, nx_), nx_);
  }

  SX EuclideanManifold::as_matrix_sx(const SX& a) const {
    SX r = SX::eye(nx_ + 1);
    r(Slice(0, nx_), nx_) = a;
    return r;
  }

  SX EuclideanManifold::Adj_sx(const SX& a) const { return SX::eye(nx_);}
  SX EuclideanManifold::Jr_sx(const SX& v) const { return SX::eye(nx_);}
  SX EuclideanManifold::Jl_sx(const SX& v) const { return SX::eye(nx_);}
  SX EuclideanManifold::Jr_inv_sx(const SX& v) const { return SX::eye(nx_);}
  SX EuclideanManifold::Jl_inv_sx(const SX& v) const { return SX::eye(nx_);}

  // ---------------------------------------------------------------------------
  // CustomManifold
  // ---------------------------------------------------------------------------

  namespace {
    void check_io(const std::string& mname, const std::string& fname, const Function& f,
                  const std::vector<casadi_int>& n_in, casadi_int n_out) {
      std::string pre = "Manifold '" + mname + "': Function '" + fname + "' ";
      casadi_assert(!f.is_null(), pre + "is null.");
      casadi_assert(f.n_in()==n_in.size() && f.n_out()==1,
        pre + "must have " + str(n_in.size()) + " input(s) and 1 output, got "
        + str(f.n_in()) + " and " + str(f.n_out()) + ".");
      for (casadi_int i=0; i<n_in.size(); ++i) {
        casadi_assert(f.size1_in(i)==n_in[i] && f.size2_in(i)==1,
          pre + "input " + str(i) + " must be " + str(n_in[i]) + "-by-1, got "
          + f.sparsity_in(i).dim() + ".");
      }
      if (n_out>=0) {
        casadi_assert(f.size1_out(0)==n_out && f.size2_out(0)==1,
          pre + "output must be " + str(n_out) + "-by-1, got " + f.sparsity_out(0).dim() + ".");
      } else {
        casadi_assert(f.size2_out(0)==1 || f.numel_out(0)==0,
          pre + "output must be a column vector, got " + f.sparsity_out(0).dim() + ".");
      }
    }

  } // namespace

  CustomManifold::CustomManifold(const std::string& name, const Function& retract,
                                 const Function& minus, const Dict& opts)
      : ManifoldInternal(name,
          retract.is_null() || retract.n_in()<2 ? 0 : retract.size1_in(0),
          retract.is_null() || retract.n_in()<2 ? 0 : retract.size1_in(1)),
        retract_(retract), minus_(minus), has_identity_(false), check_(true) {
    casadi_assert(!retract.is_null() && retract.n_in()==2,
      "Manifold '" + name + "': retract must be a Function with inputs (x, dx).");
    check_io(name, "retract", retract_, {nx_, ndx_}, nx_);
    check_io(name, "minus", minus_, {nx_, nx_}, ndx_);
    mask_.resize(nx_, false);
    mask_tan_.resize(ndx_, false);
    identity_ = DM::zeros(nx_, 1);
    for (auto&& op : opts) {
      if (op.first=="project") {
        project_ = op.second;
        check_io(name, "project", project_, {nx_}, nx_);
      } else if (op.first=="constraint") {
        constraint_ = op.second;
        check_io(name, "constraint", constraint_, {nx_}, -1);
      } else if (op.first=="identity") {
        std::vector<double> e = op.second;
        casadi_assert(e.size()==nx_, "Manifold '" + name + "': 'identity' must have length "
          + str(nx_) + ", got " + str(e.size()) + ".");
        identity_ = DM(e);
        has_identity_ = true;
      } else if (op.first=="euclidean_mask") {
        mask_ = op.second.to_bool_vector();
        casadi_assert(mask_.size()==nx_, "Manifold '" + name
          + "': 'euclidean_mask' must have length " + str(nx_) + ".");
      } else if (op.first=="euclidean_mask_tangent") {
        mask_tan_ = op.second.to_bool_vector();
        casadi_assert(mask_tan_.size()==ndx_, "Manifold '" + name
          + "': 'euclidean_mask_tangent' must have length " + str(ndx_) + ".");
      } else if (op.first=="check") {
        check_ = op.second;
      } else {
        casadi_error("Manifold '" + name + "': unknown option '" + op.first + "'. Available: "
          "project, constraint, identity, euclidean_mask, euclidean_mask_tangent, check.");
      }
    }
  }

  void CustomManifold::serialize_body(SerializingStream& s) const {
    s.pack("CustomManifold::name", name_);
    s.pack("CustomManifold::retract", retract_);
    s.pack("CustomManifold::minus", minus_);
    s.pack("CustomManifold::project", project_);
    s.pack("CustomManifold::constraint", constraint_);
    s.pack("CustomManifold::identity", identity_);
    s.pack("CustomManifold::has_identity", has_identity_);
    s.pack("CustomManifold::euclidean_mask", mask_);
    s.pack("CustomManifold::euclidean_mask_tangent", mask_tan_);
  }

  ManifoldInternal* CustomManifold::deserialize(DeserializingStream& s) {
    std::string name;
    Function retract, minus, project, constraint;
    DM identity;
    bool has_identity;
    std::vector<bool> mask, mask_tan;
    s.unpack("CustomManifold::name", name);
    s.unpack("CustomManifold::retract", retract);
    s.unpack("CustomManifold::minus", minus);
    s.unpack("CustomManifold::project", project);
    s.unpack("CustomManifold::constraint", constraint);
    s.unpack("CustomManifold::identity", identity);
    s.unpack("CustomManifold::has_identity", has_identity);
    s.unpack("CustomManifold::euclidean_mask", mask);
    s.unpack("CustomManifold::euclidean_mask_tangent", mask_tan);
    Dict opts = {{"check", false},
                 {"euclidean_mask", mask},
                 {"euclidean_mask_tangent", mask_tan}};
    if (!project.is_null()) opts["project"] = project;
    if (!constraint.is_null()) opts["constraint"] = constraint;
    if (has_identity) opts["identity"] = identity.nonzeros();
    return new CustomManifold(name, retract, minus, opts);
  }

  void CustomManifold::init_custom() {
    if (check_) check();
  }

  SX CustomManifold::retract_sx(const SX& x, const SX& dx) const {
    return retract_(std::vector<SX>{x, dx}).at(0);
  }

  SX CustomManifold::minus_sx(const SX& y, const SX& x) const {
    return minus_(std::vector<SX>{y, x}).at(0);
  }

  SX CustomManifold::project_sx(const SX& x) const {
    return project_.is_null() ? x : project_(std::vector<SX>{x}).at(0);
  }

  SX CustomManifold::constraint_sx(const SX& x) const {
    return constraint_.is_null() ? SX(0, 1) : constraint_(std::vector<SX>{x}).at(0);
  }

  void CustomManifold::check() const {
    // Deterministic pseudo-random numbers in [-1, 1]
    casadi_int seed = 12345;
    auto rnd = [&seed](casadi_int n) {
      DM r = DM::zeros(n, 1);
      for (casadi_int i=0; i<n; ++i) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        r(i) = 2.0 * static_cast<double>(seed) / 2147483648.0 - 1.0;
      }
      return r;
    };
    std::vector<DM> pts;
    if (has_identity_) pts.push_back(identity_);
    if (!project_.is_null()) {
      for (casadi_int k=0; k<3; ++k) pts.push_back(project_(std::vector<DM>{rnd(nx_)}).at(0));
    }
    std::string pre = "Manifold '" + name_ + "': consistency check failed: ";
    std::string post = " (pass the option check=false to disable this check)";
    for (const DM& x : pts) {
      double scale = 1 + norm_inf(x).scalar();
      DM r0 = retract_(std::vector<DM>{x, DM::zeros(ndx_, 1)}).at(0);
      casadi_assert(norm_inf(r0 - x).scalar() <= 1e-9*scale,
        pre + "retract(x, 0) != x at x = " + str(x) + post);
      DM m0 = minus_(std::vector<DM>{x, x}).at(0);
      casadi_assert(norm_inf(m0).scalar() <= 1e-9*scale,
        pre + "minus(x, x) != 0 at x = " + str(x) + post);
      DM dx = 1e-4 * rnd(ndx_);
      DM y = retract_(std::vector<DM>{x, dx}).at(0);
      DM d = minus_(std::vector<DM>{y, x}).at(0);
      casadi_assert(norm_inf(d - dx).scalar() <= 1e-6*scale,
        pre + "minus(retract(x, dx), x) != dx at x = " + str(x) + ", dx = " + str(dx) + post);
    }
  }

  // ---------------------------------------------------------------------------
  // ProductManifold
  // ---------------------------------------------------------------------------

  namespace {
    casadi_int total_nx(const std::vector<Manifold>& c) {
      casadi_int r = 0;
      for (auto& m : c) {
        casadi_assert(!m.is_null(), "Manifold::product: null component");
        r += m.nx();
      }
      return r;
    }
    casadi_int total_ndx(const std::vector<Manifold>& c) {
      casadi_int r = 0;
      for (auto& m : c) r += m.ndx();
      return r;
    }
    SX vertcat_or_empty(const std::vector<SX>& v) {
      return v.empty() ? SX(0, 1) : vertcat(v);
    }
  } // namespace

  ProductManifold::ProductManifold(const std::vector<Manifold>& components)
      : ManifoldInternal("product", total_nx(components), total_ndx(components)),
        comp_(components) {
    off_ = {0};
    off_tan_ = {0};
    for (auto& m : comp_) {
      off_.push_back(off_.back() + m.nx());
      off_tan_.push_back(off_tan_.back() + m.ndx());
    }
  }

  void ProductManifold::serialize_body(SerializingStream& s) const {
    s.pack("ProductManifold::components", comp_);
  }

  std::string ProductManifold::repr() const {
    std::string r = "Product(";
    for (casadi_int i=0; i<comp_.size(); ++i) {
      if (i>0) r += ", ";
      r += comp_[i]->repr();
    }
    return r + ")";
  }

  bool ProductManifold::is_euclidean() const {
    for (auto& m : comp_) if (!m.is_euclidean()) return false;
    return true;
  }

  bool ProductManifold::is_lie_group() const {
    for (auto& m : comp_) if (!m.is_lie_group()) return false;
    return true;
  }

  std::vector<bool> ProductManifold::euclidean_mask() const {
    std::vector<bool> r;
    for (auto& m : comp_) {
      std::vector<bool> c = m.euclidean_mask();
      r.insert(r.end(), c.begin(), c.end());
    }
    return r;
  }

  std::vector<bool> ProductManifold::euclidean_mask_tangent() const {
    std::vector<bool> r;
    for (auto& m : comp_) {
      std::vector<bool> c = m.euclidean_mask_tangent();
      r.insert(r.end(), c.begin(), c.end());
    }
    return r;
  }

  Manifold ProductManifold::component(casadi_int i) const {
    return comp_.at(i);
  }

  DM ProductManifold::identity() const {
    std::vector<DM> r;
    for (auto& m : comp_) r.push_back(m.identity());
    return r.empty() ? DM::zeros(0, 1) : vertcat(r);
  }

  std::vector<casadi_int> ProductManifold::point_offsets() const {
    std::vector<casadi_int> r = {0};
    for (auto& m : comp_) r.push_back(r.back() + m->point_dim());
    return r;
  }

  casadi_int ProductManifold::point_dim() const {
    return point_offsets().back();
  }

  SX ProductManifold::retract_sx(const SX& x, const SX& dx) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> xs = vertsplit(x, off_), dxs = vertsplit(dx, off_tan_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->retract_sx(xs[i], dxs[i]));
    return vertcat(r);
  }

  SX ProductManifold::minus_sx(const SX& y, const SX& x) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> ys = vertsplit(y, off_), xs = vertsplit(x, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->minus_sx(ys[i], xs[i]));
    return vertcat(r);
  }

  SX ProductManifold::project_sx(const SX& x) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> xs = vertsplit(x, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->project_sx(xs[i]));
    return vertcat(r);
  }

  SX ProductManifold::constraint_sx(const SX& x) const {
    std::vector<SX> r;
    if (!comp_.empty()) {
      std::vector<SX> xs = vertsplit(x, off_);
      for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->constraint_sx(xs[i]));
    }
    return vertcat_or_empty(r);
  }

  SX ProductManifold::compose_sx(const SX& a, const SX& b) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> as = vertsplit(a, off_), bs = vertsplit(b, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->compose_sx(as[i], bs[i]));
    return vertcat(r);
  }

  SX ProductManifold::inverse_sx(const SX& a) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> as = vertsplit(a, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->inverse_sx(as[i]));
    return vertcat(r);
  }

  SX ProductManifold::exp_sx(const SX& v) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> vs = vertsplit(v, off_tan_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->exp_sx(vs[i]));
    return vertcat(r);
  }

  SX ProductManifold::log_sx(const SX& a) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> as = vertsplit(a, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->log_sx(as[i]));
    return vertcat(r);
  }

  SX ProductManifold::act_sx(const SX& a, const SX& p) const {
    if (comp_.empty()) return SX(0, 1);
    std::vector<SX> as = vertsplit(a, off_), ps = vertsplit(p, point_offsets()), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->act_sx(as[i], ps[i]));
    return vertcat(r);
  }

  SX ProductManifold::Adj_sx(const SX& a) const {
    if (comp_.empty()) return SX(0, 0);
    std::vector<SX> as = vertsplit(a, off_), r;
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->Adj_sx(as[i]));
    return diagcat(r);
  }

#define CASADI_PRODUCT_TANGENT_BLOCKDIAG(NAME) \
  SX ProductManifold::NAME(const SX& v) const { \
    if (comp_.empty()) return SX(0, 0); \
    std::vector<SX> vs = vertsplit(v, off_tan_), r; \
    for (casadi_int i=0; i<comp_.size(); ++i) r.push_back(comp_[i]->NAME(vs[i])); \
    return diagcat(r); \
  }
  CASADI_PRODUCT_TANGENT_BLOCKDIAG(Jr_sx)
  CASADI_PRODUCT_TANGENT_BLOCKDIAG(Jl_sx)
  CASADI_PRODUCT_TANGENT_BLOCKDIAG(Jr_inv_sx)
  CASADI_PRODUCT_TANGENT_BLOCKDIAG(Jl_inv_sx)
#undef CASADI_PRODUCT_TANGENT_BLOCKDIAG

} // namespace casadi
