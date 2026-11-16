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

// Built-in matrix Lie groups. Conventions (see manifold.hpp):
//  - right perturbation, retract(x, dx) = x * Exp(dx)
//  - quaternions are Hamilton, stored [w, x, y, z]
//  - tangent of SE(n) is ordered [rho (translation); theta (rotation)]

#include "manifold_internal.hpp"

namespace casadi {

  // Threshold on theta^2 below which Taylor series are used for the
  // trigonometric coefficients (series truncated at theta^10)
  const double TRIG_THR = 0.04;
  // Threshold on z^2 for atan(z)/z (series truncated at z^18)
  const double ATAN_THR = 0.01;

  // sin(t/2)/t
  const std::vector<double> C_SINC_HALF = {1./2, -1./48, 1./3840, -1./645120,
    1./185794560, -1./81749606400};
  // cos(t/2)
  const std::vector<double> C_COS_HALF = {1., -1./8, 1./384, -1./46080,
    1./10321920, -1./3715891200};
  // sin(t)/t
  const std::vector<double> C_SINC = {1., -1./6, 1./120, -1./5040,
    1./362880, -1./39916800};
  // (1-cos(t))/t^2
  const std::vector<double> C_COSC = {1./2, -1./24, 1./720, -1./40320,
    1./3628800, -1./479001600};
  // (t-sin(t))/t^3
  const std::vector<double> C_SINC3 = {1./6, -1./120, 1./5040, -1./362880,
    1./39916800, -1./6227020800};
  // 1/t^2 - (1+cos(t))/(2 t sin(t))
  const std::vector<double> C_JINV = {1./12, 1./720, 1./30240, 1./1209600,
    1./47900160, 691./1307674368000};
  // atan(z)/z
  const std::vector<double> C_ATANC = {1., -1./3, 1./5, -1./7, 1./9,
    -1./11, 1./13, -1./15, 1./17, -1./19};

  static SX sinc_half(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_SINC_HALF,
      [](const SX& t) { return sin(t/2)/t;});
  }
  static SX cos_half(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_COS_HALF,
      [](const SX& t) { return cos(t/2);});
  }
  static SX sinc(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_SINC,
      [](const SX& t) { return sin(t)/t;});
  }
  static SX cosc(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_COSC,
      [](const SX& t) { SX s = sin(t/2); return 2*s*s/(t*t);});
  }
  static SX sinc3(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_SINC3,
      [](const SX& t) { return (t - sin(t))/(t*t*t);});
  }
  static SX jinv_coeff(const SX& t2) {
    return manifold_series_switch(t2, TRIG_THR, C_JINV,
      [](const SX& t) { return 1/(t*t) - (1 + cos(t))/(2*t*sin(t));});
  }

  // ---- quaternion helpers ([w, x, y, z]) ----
  static SX quat_mul(const SX& a, const SX& b) {
    return vertcat(std::vector<SX>{
      a(0)*b(0) - a(1)*b(1) - a(2)*b(2) - a(3)*b(3),
      a(0)*b(1) + a(1)*b(0) + a(2)*b(3) - a(3)*b(2),
      a(0)*b(2) - a(1)*b(3) + a(2)*b(0) + a(3)*b(1),
      a(0)*b(3) + a(1)*b(2) - a(2)*b(1) + a(3)*b(0)});
  }
  static SX quat_conj(const SX& q) {
    return vertcat(std::vector<SX>{q(0), -q(1), -q(2), -q(3)});
  }
  static SX quat_to_rot(const SX& q) {
    SX w = q(0), x = q(1), y = q(2), z = q(3);
    SX R = SX::zeros(3, 3);
    R(0, 0) = 1 - 2*(y*y + z*z); R(0, 1) = 2*(x*y - w*z);     R(0, 2) = 2*(x*z + w*y);
    R(1, 0) = 2*(x*y + w*z);     R(1, 1) = 1 - 2*(x*x + z*z); R(1, 2) = 2*(y*z - w*x);
    R(2, 0) = 2*(x*z - w*y);     R(2, 1) = 2*(y*z + w*x);     R(2, 2) = 1 - 2*(x*x + y*y);
    return R;
  }
  static SX so3_exp(const SX& v) {
    SX t2 = dot(v, v);
    return vertcat(cos_half(t2), sinc_half(t2)*v);
  }
  static SX so3_log(const SX& q_in) {
    // Shortest rotation: q and -q represent the same rotation
    SX q = if_else(q_in(0) < 0, -q_in, q_in);
    SX w = q(0);
    SX u = q(Slice(1, 4));
    SX s2 = dot(u, u), w2 = w*w;
    // log = 2*atan2(|u|, w)/|u| * u (scale invariant in q)
    SX small = s2 < ATAN_THR*w2;
    SX w_s = if_else(small, w, SX(1));
    SX z2 = s2/(w_s*w_s);
    SX series = 0;
    for (auto it = C_ATANC.rbegin(); it != C_ATANC.rend(); ++it) series = series*z2 + *it;
    series = 2*series/w_s;
    SX s_safe = sqrt(if_else(small, SX(1), s2));
    SX exact = 2*atan2(s_safe, w)/s_safe;
    return if_else(small, series, exact)*u;
  }
  // J_l(v) = I + cosc [v]x + sinc3 [v]x^2, J_r(v) = J_l(-v)
  static SX so3_Jl(const SX& v) {
    SX t2 = dot(v, v), K = manifold_skew(v);
    return SX::eye(3) + cosc(t2)*K + sinc3(t2)*mtimes(K, K);
  }
  // J_l(v)^-1 = I - [v]x/2 + jinv [v]x^2, J_r(v)^-1 = J_l(-v)^-1
  static SX so3_Jl_inv(const SX& v) {
    SX t2 = dot(v, v), K = manifold_skew(v);
    return SX::eye(3) - 0.5*K + jinv_coeff(t2)*mtimes(K, K);
  }

  // ---------------------------------------------------------------------------
  // SO3Manifold
  // ---------------------------------------------------------------------------

  SO3Manifold::SO3Manifold() : LieGroupInternal("SO3", 4, 3) {
  }

  DM SO3Manifold::identity() const {
    return DM(std::vector<double>{1, 0, 0, 0});
  }

  SX SO3Manifold::project_sx(const SX& x) const {
    return x/norm_2(x);
  }

  SX SO3Manifold::constraint_sx(const SX& x) const {
    return dot(x, x) - 1;
  }

  SX SO3Manifold::compose_sx(const SX& a, const SX& b) const { return quat_mul(a, b);}
  SX SO3Manifold::inverse_sx(const SX& a) const { return quat_conj(a);}
  SX SO3Manifold::exp_sx(const SX& v) const { return so3_exp(v);}
  SX SO3Manifold::log_sx(const SX& a) const { return so3_log(a);}
  SX SO3Manifold::act_sx(const SX& a, const SX& p) const { return mtimes(quat_to_rot(a), p);}
  SX SO3Manifold::hat_sx(const SX& v) const { return manifold_skew(v);}
  SX SO3Manifold::vee_sx(const SX& V) const {
    return vertcat(std::vector<SX>{V(2, 1), V(0, 2), V(1, 0)});
  }
  SX SO3Manifold::as_matrix_sx(const SX& a) const { return quat_to_rot(a);}
  SX SO3Manifold::Adj_sx(const SX& a) const { return quat_to_rot(a);}
  SX SO3Manifold::Jr_sx(const SX& v) const { return so3_Jl(-v);}
  SX SO3Manifold::Jl_sx(const SX& v) const { return so3_Jl(v);}
  SX SO3Manifold::Jr_inv_sx(const SX& v) const { return so3_Jl_inv(-v);}
  SX SO3Manifold::Jl_inv_sx(const SX& v) const { return so3_Jl_inv(v);}


  // ---------------------------------------------------------------------------
  // SO2Manifold
  // ---------------------------------------------------------------------------

  // complex multiplication of [c, s] pairs
  static SX so2_mul(const SX& a, const SX& b) {
    return vertcat(a(0)*b(0) - a(1)*b(1), a(1)*b(0) + a(0)*b(1));
  }
  static SX so2_rot(const SX& a) {
    SX R = SX::zeros(2, 2);
    R(0, 0) = a(0); R(0, 1) = -a(1);
    R(1, 0) = a(1); R(1, 1) = a(0);
    return R;
  }
  static SX so2_hat(const SX& th) {
    SX r = SX::zeros(2, 2);
    r(0, 1) = -th; r(1, 0) = th;
    return r;
  }
  static SX unit_rows(const SX& x, casadi_int i0) {
    SX r = x;
    SX c = x(Slice(i0, x.size1()));
    r(Slice(i0, x.size1())) = c/norm_2(c);
    return r;
  }

  SO2Manifold::SO2Manifold() : LieGroupInternal("SO2", 2, 1) {
  }

  DM SO2Manifold::identity() const {
    return DM(std::vector<double>{1, 0});
  }

  SX SO2Manifold::project_sx(const SX& x) const { return x/norm_2(x);}
  SX SO2Manifold::constraint_sx(const SX& x) const { return dot(x, x) - 1;}
  SX SO2Manifold::compose_sx(const SX& a, const SX& b) const { return so2_mul(a, b);}
  SX SO2Manifold::inverse_sx(const SX& a) const { return vertcat(a(0), -a(1));}
  SX SO2Manifold::exp_sx(const SX& v) const { return vertcat(cos(v), sin(v));}
  SX SO2Manifold::log_sx(const SX& a) const { return atan2(a(1), a(0));}
  SX SO2Manifold::act_sx(const SX& a, const SX& p) const { return mtimes(so2_rot(a), p);}
  SX SO2Manifold::hat_sx(const SX& v) const { return so2_hat(v);}
  SX SO2Manifold::vee_sx(const SX& V) const { return V(1, 0);}
  SX SO2Manifold::as_matrix_sx(const SX& a) const { return so2_rot(a);}
  SX SO2Manifold::Adj_sx(const SX& a) const { return SX(1);}
  SX SO2Manifold::Jr_sx(const SX& v) const { return SX(1);}
  SX SO2Manifold::Jl_sx(const SX& v) const { return SX(1);}
  SX SO2Manifold::Jr_inv_sx(const SX& v) const { return SX(1);}
  SX SO2Manifold::Jl_inv_sx(const SX& v) const { return SX(1);}

  // ---------------------------------------------------------------------------
  // SE2Manifold: [x, y, c, s], tangent [rho_x, rho_y, theta]
  // ---------------------------------------------------------------------------

  SE2Manifold::SE2Manifold() : LieGroupInternal("SE2", 4, 3) {
  }

  DM SE2Manifold::identity() const {
    return DM(std::vector<double>{0, 0, 1, 0});
  }

  SX SE2Manifold::project_sx(const SX& x) const { return unit_rows(x, 2);}

  SX SE2Manifold::constraint_sx(const SX& x) const {
    SX r = x(Slice(2, 4));
    return dot(r, r) - 1;
  }

  SX SE2Manifold::compose_sx(const SX& a, const SX& b) const {
    SX ra = a(Slice(2, 4)), rb = b(Slice(2, 4));
    return vertcat(a(Slice(0, 2)) + mtimes(so2_rot(ra), b(Slice(0, 2))), so2_mul(ra, rb));
  }

  SX SE2Manifold::inverse_sx(const SX& a) const {
    SX c = a(2), s = a(3), x = a(0), y = a(1);
    return vertcat(std::vector<SX>{-(c*x + s*y), s*x - c*y, c, -s});
  }

  SX SE2Manifold::exp_sx(const SX& v) const {
    SX th = v(2), t2 = th*th;
    // V(theta) = [A -B; B A], A = sin(th)/th, B = (1-cos(th))/th
    SX A = sinc(t2), B = th*cosc(t2);
    return vertcat(std::vector<SX>{A*v(0) - B*v(1), B*v(0) + A*v(1), cos(th), sin(th)});
  }

  SX SE2Manifold::log_sx(const SX& a) const {
    SX th = atan2(a(3), a(2)), t2 = th*th;
    SX A = sinc(t2), B = th*cosc(t2);
    // V^-1 = [A B; -B A]/(A^2 + B^2)
    SX d = A*A + B*B;
    return vertcat(std::vector<SX>{(A*a(0) + B*a(1))/d, (-B*a(0) + A*a(1))/d, th});
  }

  SX SE2Manifold::act_sx(const SX& a, const SX& p) const {
    return mtimes(so2_rot(a(Slice(2, 4))), p) + a(Slice(0, 2));
  }

  SX SE2Manifold::hat_sx(const SX& v) const {
    SX r = SX::zeros(3, 3);
    r(Slice(0, 2), Slice(0, 2)) = so2_hat(v(2));
    r(Slice(0, 2), 2) = v(Slice(0, 2));
    return r;
  }

  SX SE2Manifold::vee_sx(const SX& V) const {
    return vertcat(V(0, 2), V(1, 2), V(1, 0));
  }

  SX SE2Manifold::as_matrix_sx(const SX& a) const {
    SX r = SX::eye(3);
    r(Slice(0, 2), Slice(0, 2)) = so2_rot(a(Slice(2, 4)));
    r(Slice(0, 2), 2) = a(Slice(0, 2));
    return r;
  }

  // ---------------------------------------------------------------------------
  // SE3Manifold: [p (3); q (4)], tangent [rho (3); theta (3)]
  // ---------------------------------------------------------------------------

  SE3Manifold::SE3Manifold() : LieGroupInternal("SE3", 7, 6) {
  }

  DM SE3Manifold::identity() const {
    return DM(std::vector<double>{0, 0, 0, 1, 0, 0, 0});
  }

  SX SE3Manifold::project_sx(const SX& x) const { return unit_rows(x, 3);}

  SX SE3Manifold::constraint_sx(const SX& x) const {
    SX q = x(Slice(3, 7));
    return dot(q, q) - 1;
  }

  SX SE3Manifold::compose_sx(const SX& a, const SX& b) const {
    SX qa = a(Slice(3, 7));
    return vertcat(a(Slice(0, 3)) + mtimes(quat_to_rot(qa), b(Slice(0, 3))),
                   quat_mul(qa, b(Slice(3, 7))));
  }

  SX SE3Manifold::inverse_sx(const SX& a) const {
    SX qi = quat_conj(a(Slice(3, 7)));
    return vertcat(-mtimes(quat_to_rot(qi), a(Slice(0, 3))), qi);
  }

  SX SE3Manifold::exp_sx(const SX& v) const {
    SX th = v(Slice(3, 6));
    // translation: V(theta) rho with V = J_l of SO(3)
    return vertcat(mtimes(so3_Jl(th), v(Slice(0, 3))), so3_exp(th));
  }

  SX SE3Manifold::log_sx(const SX& a) const {
    SX th = so3_log(a(Slice(3, 7)));
    return vertcat(mtimes(so3_Jl_inv(th), a(Slice(0, 3))), th);
  }

  SX SE3Manifold::act_sx(const SX& a, const SX& p) const {
    return mtimes(quat_to_rot(a(Slice(3, 7))), p) + a(Slice(0, 3));
  }

  SX SE3Manifold::hat_sx(const SX& v) const {
    SX r = SX::zeros(4, 4);
    r(Slice(0, 3), Slice(0, 3)) = manifold_skew(v(Slice(3, 6)));
    r(Slice(0, 3), 3) = v(Slice(0, 3));
    return r;
  }

  SX SE3Manifold::vee_sx(const SX& V) const {
    return vertcat(std::vector<SX>{V(0, 3), V(1, 3), V(2, 3), V(2, 1), V(0, 2), V(1, 0)});
  }

  SX SE3Manifold::as_matrix_sx(const SX& a) const {
    SX r = SX::eye(4);
    r(Slice(0, 3), Slice(0, 3)) = quat_to_rot(a(Slice(3, 7)));
    r(Slice(0, 3), 3) = a(Slice(0, 3));
    return r;
  }

} // namespace casadi
