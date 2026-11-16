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


#include "manifold.hpp"
#include "manifold_internal.hpp"

namespace casadi {

  Manifold::Manifold() {
  }

  Manifold Manifold::create(ManifoldInternal* node) {
    Manifold ret;
    ret.own(node);
    return ret;
  }

  ManifoldInternal* Manifold::get() const {
    return static_cast<ManifoldInternal*>(SharedObject::get());
  }

  const ManifoldInternal* Manifold::operator->() const {
    return static_cast<const ManifoldInternal*>(SharedObject::operator->());
  }

  ManifoldInternal* Manifold::operator->() {
    return static_cast<ManifoldInternal*>(SharedObject::operator->());
  }

  bool Manifold::test_cast(const SharedObjectInternal* ptr) {
    return dynamic_cast<const ManifoldInternal*>(ptr)!=nullptr;
  }

  Manifold Manifold::Euclidean(casadi_int n) {
    return create(new EuclideanManifold(n));
  }

  Manifold Manifold::product(const std::vector<Manifold>& components) {
    return create(new ProductManifold(components));
  }

  std::string Manifold::name() const { return (*this)->name();}
  casadi_int Manifold::nx() const { return (*this)->nx();}
  casadi_int Manifold::ndx() const { return (*this)->ndx();}
  bool Manifold::is_euclidean() const { return (*this)->is_euclidean();}
  std::vector<bool> Manifold::euclidean_mask() const { return (*this)->euclidean_mask();}
  std::vector<bool> Manifold::euclidean_mask_tangent() const {
    return (*this)->euclidean_mask_tangent();
  }
  casadi_int Manifold::n_components() const { return (*this)->n_components();}
  std::vector<casadi_int> Manifold::offsets() const { return (*this)->offsets();}
  std::vector<casadi_int> Manifold::offsets_tangent() const { return (*this)->offsets_tangent();}
  DM Manifold::identity() const { return (*this)->identity();}

  Manifold Manifold::component(casadi_int i) const {
    casadi_assert(i>=0 && i<n_components(),
      "Manifold::component: index " + str(i) + " out of range [0, "
      + str(n_components()) + ")");
    return (*this)->component(i);
  }

  Function Manifold::get_function(const std::string& name) const {
    return (*this)->get_function(name);
  }

  std::vector<std::string> Manifold::function_names() {
    return ManifoldInternal::function_names();
  }

  /** Evaluate a manifold Function column-wise

      Every argument must have the number of rows of the corresponding Function
      input. Arguments have either 1 or K columns; single columns are broadcast.
  */
  template<typename M>
  static M call_columnwise(const Manifold& m, const std::string& op,
                           std::vector<M> arg) {
    Function f = m.get_function(op);
    casadi_assert_dev(arg.size()==f.n_in());
    casadi_int K = 1;
    for (casadi_int i=0; i<arg.size(); ++i) {
      casadi_assert(arg[i].size1()==f.size1_in(i),
        "Manifold::" + op + " (" + m.name() + "): argument '" + f.name_in(i)
        + "' must have " + str(f.size1_in(i)) + " rows, got "
        + str(arg[i].size1()) + ".");
      casadi_int c = arg[i].size2();
      casadi_assert(c>=1,
        "Manifold::" + op + " (" + m.name() + "): argument '" + f.name_in(i)
        + "' has no columns.");
      if (c==1) continue;
      casadi_assert(K==1 || c==K,
        "Manifold::" + op + " (" + m.name() + "): inconsistent number of columns: "
        + str(K) + " and " + str(c) + ".");
      K = c;
    }
    if (K==1) return f(arg).at(0);
    for (auto& a : arg) if (a.size2()==1) a = repmat(a, 1, K);
    return f.map(K)(arg).at(0);
  }

  DM Manifold::retract(const DM& x, const DM& dx) const {
    return call_columnwise<DM>(*this, "retract", {x, dx});
  }
  SX Manifold::retract(const SX& x, const SX& dx) const {
    return call_columnwise<SX>(*this, "retract", {x, dx});
  }
  MX Manifold::retract(const MX& x, const MX& dx) const {
    return call_columnwise<MX>(*this, "retract", {x, dx});
  }

  DM Manifold::minus(const DM& y, const DM& x) const {
    return call_columnwise<DM>(*this, "minus", {y, x});
  }
  SX Manifold::minus(const SX& y, const SX& x) const {
    return call_columnwise<SX>(*this, "minus", {y, x});
  }
  MX Manifold::minus(const MX& y, const MX& x) const {
    return call_columnwise<MX>(*this, "minus", {y, x});
  }

  DM Manifold::project(const DM& x) const {
    return call_columnwise<DM>(*this, "project", {x});
  }
  SX Manifold::project(const SX& x) const {
    return call_columnwise<SX>(*this, "project", {x});
  }
  MX Manifold::project(const MX& x) const {
    return call_columnwise<MX>(*this, "project", {x});
  }

  DM Manifold::constraint(const DM& x) const {
    return call_columnwise<DM>(*this, "constraint", {x});
  }
  SX Manifold::constraint(const SX& x) const {
    return call_columnwise<SX>(*this, "constraint", {x});
  }
  MX Manifold::constraint(const MX& x) const {
    return call_columnwise<MX>(*this, "constraint", {x});
  }

} // namespace casadi
