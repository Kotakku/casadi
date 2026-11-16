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
      (the *_sx methods). Those are turned into Functions on first use
      and cached (get_function).
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
    virtual std::vector<bool> euclidean_mask() const;
    virtual std::vector<bool> euclidean_mask_tangent() const;
    virtual casadi_int n_components() const {return 1;}
    virtual Manifold component(casadi_int i) const;
    virtual std::vector<casadi_int> offsets() const;
    virtual std::vector<casadi_int> offsets_tangent() const;
    virtual DM identity() const = 0;
    ///@}

    /// \name Symbolic primitives (column vectors)
    ///@{
    virtual SX retract_sx(const SX& x, const SX& dx) const = 0;
    virtual SX minus_sx(const SX& y, const SX& x) const = 0;
    virtual SX project_sx(const SX& x) const;
    virtual SX constraint_sx(const SX& x) const;
    ///@}

    /// Get a cached Function implementing an operation
    Function get_function(const std::string& fname) const;

    /// Names accepted by get_function
    static std::vector<std::string> function_names();

  protected:
    /// Create the Function implementing an operation (not cached)
    virtual Function create_function(const std::string& fname) const;

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

  /** \brief Euclidean space R^n */
  class CASADI_EXPORT EuclideanManifold : public ManifoldInternal {
  public:
    explicit EuclideanManifold(casadi_int n);
    std::string class_name() const override {return "EuclideanManifold";}
    std::string repr() const override;
    bool is_euclidean() const override {return true;}
    std::vector<bool> euclidean_mask() const override;
    std::vector<bool> euclidean_mask_tangent() const override;
    DM identity() const override;
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
  };

  /** \brief Cartesian product of manifolds */
  class CASADI_EXPORT ProductManifold : public ManifoldInternal {
  public:
    explicit ProductManifold(const std::vector<Manifold>& components);
    std::string class_name() const override {return "ProductManifold";}
    std::string repr() const override;
    bool is_euclidean() const override;
    std::vector<bool> euclidean_mask() const override;
    std::vector<bool> euclidean_mask_tangent() const override;
    casadi_int n_components() const override {return comp_.size();}
    Manifold component(casadi_int i) const override;
    std::vector<casadi_int> offsets() const override {return off_;}
    std::vector<casadi_int> offsets_tangent() const override {return off_tan_;}
    DM identity() const override;
    SX retract_sx(const SX& x, const SX& dx) const override;
    SX minus_sx(const SX& y, const SX& x) const override;
    SX project_sx(const SX& x) const override;
    SX constraint_sx(const SX& x) const override;
  private:
    std::vector<Manifold> comp_;
    std::vector<casadi_int> off_, off_tan_;
  };

} // namespace casadi
/// \endcond

#endif // CASADI_MANIFOLD_INTERNAL_HPP
