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
#include "serializing_stream.hpp"

#include <sstream>

namespace casadi {

  Manifold::Manifold() {
  }

  Manifold::Manifold(const std::string& name, const Function& retract, const Function& minus,
                     const Dict& opts) {
    own(new CustomManifold(name, retract, minus, opts));
    (*this)->init_custom();
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

  Manifold Manifold::SO3() {
    return create(new SO3Manifold());
  }

  Manifold Manifold::SO2() {
    return create(new SO2Manifold());
  }

  Manifold Manifold::SE2() {
    return create(new SE2Manifold());
  }

  Manifold Manifold::SE3() {
    return create(new SE3Manifold());
  }

  Manifold Manifold::product(const std::vector<Manifold>& components) {
    return create(new ProductManifold(components));
  }

  std::string Manifold::name() const { return (*this)->name();}
  casadi_int Manifold::nx() const { return (*this)->nx();}
  casadi_int Manifold::ndx() const { return (*this)->ndx();}
  bool Manifold::is_euclidean() const { return (*this)->is_euclidean();}
  bool Manifold::is_lie_group() const { return (*this)->is_lie_group();}
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

  void Manifold::serialize(SerializingStream& s) const {
    if (is_null()) {
      s.pack("Manifold::null", true);
    } else {
      s.pack("Manifold::null", false);
      (*this)->serialize(s);
    }
  }

  Manifold Manifold::deserialize(DeserializingStream& s) {
    bool null;
    s.unpack("Manifold::null", null);
    if (null) return Manifold();
    return create(ManifoldInternal::deserialize(s));
  }

  std::string Manifold::serialize() const {
    std::stringstream ss;
    SerializingStream s(ss);
    serialize(s);
    return ss.str();
  }

  Manifold Manifold::deserialize(const std::string& str) {
    std::stringstream ss;
    ss << str;
    DeserializingStream s(ss);
    return deserialize(s);
  }

  /** Evaluate a manifold Function column-wise

      Argument i must have the number of rows of Function input i and a number
      of columns that is a multiple K_i of its number of columns (matrix inputs
      are stacked horizontally). All K_i must be equal or 1 (broadcast).
  */
  template<typename M>
  static M call_columnwise(const Manifold& m, const std::string& op,
                           std::vector<M> arg) {
    Function f = m.get_function(op);
    casadi_assert_dev(arg.size()==f.n_in());
    casadi_int K = 1;
    for (casadi_int i=0; i<arg.size(); ++i) {
      casadi_int r = f.size1_in(i), c = f.size2_in(i);
      std::string expected = r==1 && c==1 ? "1-by-1" : str(r) + "-by-" + str(c);
      std::string msg = "Manifold::" + op + " (" + m->repr() + "): argument '"
        + f.name_in(i) + "' must be " + expected
        + (c==1 ? "" : " (or horizontally stacked " + expected + " blocks)")
        + ", got " + arg[i].dim(false) + ".";
      casadi_assert(arg[i].size1()==r, msg);
      casadi_assert(arg[i].size2()>=1 && arg[i].size2() % c == 0, msg);
      casadi_int k = arg[i].size2() / c;
      if (k==1) continue;
      casadi_assert(K==1 || k==K,
        "Manifold::" + op + " (" + m->repr() + "): inconsistent number of columns: "
        + str(K) + " and " + str(k) + " blocks.");
      K = k;
    }
    if (K==1) return f(arg).at(0);
    for (casadi_int i=0; i<arg.size(); ++i) {
      if (arg[i].size2()==f.size2_in(i)) arg[i] = repmat(arg[i], 1, K);
    }
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

  DM Manifold::jac_retract_dx(const DM& x) const {
    return call_columnwise<DM>(*this, "jac_retract_dx", {x});
  }
  SX Manifold::jac_retract_dx(const SX& x) const {
    return call_columnwise<SX>(*this, "jac_retract_dx", {x});
  }
  MX Manifold::jac_retract_dx(const MX& x) const {
    return call_columnwise<MX>(*this, "jac_retract_dx", {x});
  }
  DM Manifold::jac_minus_y(const DM& x) const {
    return call_columnwise<DM>(*this, "jac_minus_y", {x});
  }
  SX Manifold::jac_minus_y(const SX& x) const {
    return call_columnwise<SX>(*this, "jac_minus_y", {x});
  }
  MX Manifold::jac_minus_y(const MX& x) const {
    return call_columnwise<MX>(*this, "jac_minus_y", {x});
  }
  DM Manifold::compose(const DM& a, const DM& b) const {
    return call_columnwise<DM>(*this, "compose", {a, b});
  }
  SX Manifold::compose(const SX& a, const SX& b) const {
    return call_columnwise<SX>(*this, "compose", {a, b});
  }
  MX Manifold::compose(const MX& a, const MX& b) const {
    return call_columnwise<MX>(*this, "compose", {a, b});
  }
  DM Manifold::inverse(const DM& a) const {
    return call_columnwise<DM>(*this, "inverse", {a});
  }
  SX Manifold::inverse(const SX& a) const {
    return call_columnwise<SX>(*this, "inverse", {a});
  }
  MX Manifold::inverse(const MX& a) const {
    return call_columnwise<MX>(*this, "inverse", {a});
  }
  DM Manifold::exp(const DM& v) const {
    return call_columnwise<DM>(*this, "exp", {v});
  }
  SX Manifold::exp(const SX& v) const {
    return call_columnwise<SX>(*this, "exp", {v});
  }
  MX Manifold::exp(const MX& v) const {
    return call_columnwise<MX>(*this, "exp", {v});
  }
  DM Manifold::log(const DM& a) const {
    return call_columnwise<DM>(*this, "log", {a});
  }
  SX Manifold::log(const SX& a) const {
    return call_columnwise<SX>(*this, "log", {a});
  }
  MX Manifold::log(const MX& a) const {
    return call_columnwise<MX>(*this, "log", {a});
  }
  DM Manifold::act(const DM& a, const DM& p) const {
    return call_columnwise<DM>(*this, "act", {a, p});
  }
  SX Manifold::act(const SX& a, const SX& p) const {
    return call_columnwise<SX>(*this, "act", {a, p});
  }
  MX Manifold::act(const MX& a, const MX& p) const {
    return call_columnwise<MX>(*this, "act", {a, p});
  }
  DM Manifold::hat(const DM& v) const {
    return call_columnwise<DM>(*this, "hat", {v});
  }
  SX Manifold::hat(const SX& v) const {
    return call_columnwise<SX>(*this, "hat", {v});
  }
  MX Manifold::hat(const MX& v) const {
    return call_columnwise<MX>(*this, "hat", {v});
  }
  DM Manifold::vee(const DM& V) const {
    return call_columnwise<DM>(*this, "vee", {V});
  }
  SX Manifold::vee(const SX& V) const {
    return call_columnwise<SX>(*this, "vee", {V});
  }
  MX Manifold::vee(const MX& V) const {
    return call_columnwise<MX>(*this, "vee", {V});
  }
  DM Manifold::as_matrix(const DM& a) const {
    return call_columnwise<DM>(*this, "as_matrix", {a});
  }
  SX Manifold::as_matrix(const SX& a) const {
    return call_columnwise<SX>(*this, "as_matrix", {a});
  }
  MX Manifold::as_matrix(const MX& a) const {
    return call_columnwise<MX>(*this, "as_matrix", {a});
  }
  DM Manifold::Adj(const DM& a) const {
    return call_columnwise<DM>(*this, "Adj", {a});
  }
  SX Manifold::Adj(const SX& a) const {
    return call_columnwise<SX>(*this, "Adj", {a});
  }
  MX Manifold::Adj(const MX& a) const {
    return call_columnwise<MX>(*this, "Adj", {a});
  }
  DM Manifold::Jr(const DM& v) const {
    return call_columnwise<DM>(*this, "Jr", {v});
  }
  SX Manifold::Jr(const SX& v) const {
    return call_columnwise<SX>(*this, "Jr", {v});
  }
  MX Manifold::Jr(const MX& v) const {
    return call_columnwise<MX>(*this, "Jr", {v});
  }
  DM Manifold::Jl(const DM& v) const {
    return call_columnwise<DM>(*this, "Jl", {v});
  }
  SX Manifold::Jl(const SX& v) const {
    return call_columnwise<SX>(*this, "Jl", {v});
  }
  MX Manifold::Jl(const MX& v) const {
    return call_columnwise<MX>(*this, "Jl", {v});
  }
  DM Manifold::Jr_inv(const DM& v) const {
    return call_columnwise<DM>(*this, "Jr_inv", {v});
  }
  SX Manifold::Jr_inv(const SX& v) const {
    return call_columnwise<SX>(*this, "Jr_inv", {v});
  }
  MX Manifold::Jr_inv(const MX& v) const {
    return call_columnwise<MX>(*this, "Jr_inv", {v});
  }
  DM Manifold::Jl_inv(const DM& v) const {
    return call_columnwise<DM>(*this, "Jl_inv", {v});
  }
  SX Manifold::Jl_inv(const SX& v) const {
    return call_columnwise<SX>(*this, "Jl_inv", {v});
  }
  MX Manifold::Jl_inv(const MX& v) const {
    return call_columnwise<MX>(*this, "Jl_inv", {v});
  }

} // namespace casadi
