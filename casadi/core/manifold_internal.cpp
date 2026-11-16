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

namespace casadi {

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

  SX ManifoldInternal::project_sx(const SX& x) const {
    return x;
  }

  SX ManifoldInternal::constraint_sx(const SX& x) const {
    return SX(0, 1);
  }

  std::vector<std::string> ManifoldInternal::function_names() {
    return {"retract", "minus", "project", "constraint"};
  }

  Function ManifoldInternal::create_function(const std::string& fname) const {
    SX x = SX::sym("x", nx_);
    if (fname=="retract") {
      SX dx = SX::sym("dx", ndx_);
      return Function(name_ + "_retract", {x, dx}, {retract_sx(x, dx)},
                      {"x", "dx"}, {"x_next"});
    } else if (fname=="minus") {
      SX y = SX::sym("y", nx_);
      return Function(name_ + "_minus", {y, x}, {minus_sx(y, x)},
                      {"y", "x"}, {"dx"});
    } else if (fname=="project") {
      return Function(name_ + "_project", {x}, {project_sx(x)},
                      {"x"}, {"x_proj"});
    } else if (fname=="constraint") {
      return Function(name_ + "_constraint", {x}, {constraint_sx(x)},
                      {"x"}, {"c"});
    }
    casadi_error("Manifold::get_function: unknown function '" + fname
                 + "'. Available: " + str(function_names()));
  }

  Function ManifoldInternal::get_function(const std::string& fname) const {
#ifdef CASADI_WITH_THREAD
    std::lock_guard<std::mutex> lock(mtx_);
#endif // CASADI_WITH_THREAD
    auto it = cache_.find(fname);
    if (it!=cache_.end()) return it->second;
    Function f = create_function(fname);
    cache_[fname] = f;
    return f;
  }

  // ---------------------------------------------------------------------------
  // EuclideanManifold
  // ---------------------------------------------------------------------------

  EuclideanManifold::EuclideanManifold(casadi_int n)
      : ManifoldInternal("R" + str(n), n, n) {
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

  SX EuclideanManifold::retract_sx(const SX& x, const SX& dx) const {
    return x + dx;
  }

  SX EuclideanManifold::minus_sx(const SX& y, const SX& x) const {
    return y - x;
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
    return r.empty() ? SX(0, 1) : vertcat(r);
  }

} // namespace casadi
