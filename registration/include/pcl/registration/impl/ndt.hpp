/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_REGISTRATION_NDT_IMPL_H_
#define PCL_REGISTRATION_NDT_IMPL_H_
#include <iostream>
namespace pcl {

template <typename PointSource, typename PointTarget, typename Scalar>
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::
    NormalDistributionsTransform()
: source_cells_()
, target_cells_()
, use_d2d_(false)
, resolution_(1.0f)
, step_size_(0.1)
, outlier_ratio_(0.55)
, gauss_d1_()
, gauss_d2_()
, trans_likelihood_()
{
  reg_name_ = "NormalDistributionsTransform";

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10.0 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  gauss_d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                    gauss_d1_);

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeTransformation(
    PointCloudSource& output, const Matrix4& guess)
{
  nr_iterations_ = 0;
  converged_ = false;
  if (target_cells_.getCentroids()->empty()) {
    PCL_ERROR("[%s::computeTransformation] Voxel grid is not searchable!\n",
              getClassName().c_str());
    return;
  }

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  gauss_d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                    gauss_d1_);

  if (guess != Matrix4::Identity()) {
    // Initialise final transformation to the guessed one
    final_transformation_ = guess;
    // Apply guessed transformation prior to search for neighbours
    transformPointCloud(output, output, guess);
    // TODO trans leave?
  }

  Eigen::Transform<Scalar, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = final_transformation_;

  // Convert initial guess matrix to 6 element transformation vector
  Eigen::Matrix<double, 6, 1> transform, score_gradient;
  transform.head(3) = eig_transformation.translation().template cast<double>();
  transform.tail(3) = eig_transformation.rotation().eulerAngles(0, 1, 2).template cast<double>();

  Eigen::Matrix<double, 6, 6> hessian;

  // Calculate derivates of initial transform vector, subsequent derivative calculations
  // are done in the step length determination.
  double score;
  std::vector<TransLeaf> trans_leaves;
  if (!use_d2d_) {
    std::cout << "Init: " << output.size() << std::endl;
    score = computeDerivatives(score_gradient, hessian, output, transform);
  } else {
    initSourceCells();
    for (const auto& elem : source_cells_.getLeaves()) {
      if (elem.second.nr_points < 6)
        continue;
      trans_leaves.push_back(TransLeaf(elem.second.mean_, elem.second.cov_));
    }
    std::cout << "tl size: " << trans_leaves.size() << std::endl;
    score = computeDerivativesD2D(score_gradient, hessian, trans_leaves, transform);
  }

  while (!converged_) {
    // Store previous transformation
    previous_transformation_ = transformation_;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson
    // 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
        hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    Eigen::Matrix<double, 6, 1> delta = sv.solve(-score_gradient);

    // Calculate step length with guaranteed sufficient decrease [More, Thuente 1994]
    double delta_norm = delta.norm();

    if (delta_norm == 0 || std::isnan(delta_norm)) {
      trans_likelihood_ = score / static_cast<double>(input_->size());
      converged_ = delta_norm == 0;
      return;
    }

    delta /= delta_norm;
    delta_norm = computeStepLengthMT(transform,
                                     delta,
                                     delta_norm,
                                     step_size_,
                                     transformation_epsilon_ / 2,
                                     score,
                                     score_gradient,
                                     hessian,
                                     output);
    delta *= delta_norm;

    // Convert delta into matrix form
    convertTransform(delta, transformation_);

    transform += delta;

    // Update Visualizer (untested)
    if (update_visualizer_)
      update_visualizer_(output, pcl::Indices(), *target_, pcl::Indices());

    const double cos_angle =
        0.5 * (transformation_.template block<3, 3>(0, 0).trace() - 1);
    const double translation_sqr =
        transformation_.template block<3, 1>(0, 3).squaredNorm();

    nr_iterations_++;

    if (nr_iterations_ >= max_iterations_ ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ <= 0) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ <= 0))) {
      converged_ = true;
    }
  }

  // Store transformation likelihood.  The relative differences within each scan
  // registration are accurate but the normalization constants need to be modified for
  // it to be globally accurate
  trans_likelihood_ = score / static_cast<double>(input_->size());
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const PointCloudSource& trans_cloud,
    const Eigen::Matrix<double, 6, 1>& transform,
    bool compute_hessian)
{
  std::cout << __FUNCTION__ << ": " << transform.transpose() << std::endl;
  score_gradient.setZero();
  hessian.setZero();
  double score = 0;

  // Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
  computeAngleDerivatives(transform);

  // Update gradient and hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
  for (std::size_t idx = 0; idx < input_->size(); idx++) {
    // Transformed Point
    const auto& x_trans_pt = trans_cloud[idx];

    // Find neighbors (Radius search has been experimentally faster than direct neighbor
    // checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);

    for (const auto& cell : neighborhood) {
      // Original Point
      const auto& x_pt = (*input_)[idx];
      const Eigen::Vector3d x = x_pt.getVector3fMap().template cast<double>();

      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      const Eigen::Vector3d x_trans =
          x_trans_pt.getVector3fMap().template cast<double>() - cell->getMean();
      // Inverse Covariance of Occupied Voxel
      // Uses precomputed covariance for speed.
      const Eigen::Matrix3d c_inv = cell->getInverseCov();

      // Compute derivative of transform function w.r.t. transform vector, J_E and H_E
      // in Equations 6.18 and 6.20 [Magnusson 2009]
      computePointDerivatives(x);
      // Update score, gradient and hessian, lines 19-21 in Algorithm 2, according to
      // Equations 6.10, 6.12 and 6.13, respectively [Magnusson 2009]
      score +=
          updateDerivatives(score_gradient, hessian, x_trans, c_inv, compute_hessian);
    }
  }
  return score;
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeDerivativesD2D(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const std::vector<TransLeaf>& trans_leaves,
    const Eigen::Matrix<double, 6, 1>& transform,
    bool compute_hessian)
{
  std::cout << __FUNCTION__ << ": " << transform.transpose() << std::endl;
  score_gradient.setZero();
  hessian.setZero();
  double score = 0;
  computeAngleDerivatives(transform);

  for (const auto& leaf : trans_leaves) {
    const auto& mean = leaf.mean;
    const auto& cov = leaf.cov;
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    PointSource meanpt;
    meanpt.x = mean(0), meanpt.y = mean(1), meanpt.z = mean(2);
    target_cells_.radiusSearch(meanpt, resolution_, neighborhood, distances);
    // XXX: Nearest Neighbor / Set size 1
    // target_cells_.nearestKSearch(meanpt, 1, neighborhood, distances);
    for (const auto& cell : neighborhood) {
      if (cell->nr_points < 6)
        continue;
      Eigen::Vector3d uij = mean - cell->getMean();
      Eigen::Matrix3d B = (R_ * cov * R_.transpose() + cell->getCov()).inverse();
      computePointDerivatives(uij, compute_hessian);
      computeCovarianceDerivatives(cov, compute_hessian);
      score += updateDerivativesD2D(score_gradient, hessian, uij, B, compute_hessian);
    }
  }
  return score;
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeAngleDerivatives(
    const Eigen::Matrix<double, 6, 1>& transform, bool compute_hessian)
{
  // Simplified math for near 0 angles
  const auto calculate_cos_sin = [](double angle, double& c, double& s) {
    if (std::abs(angle) < 10e-5) {
      c = 1.0;
      s = 0.0;
    }
    else {
      c = std::cos(angle);
      s = std::sin(angle);
    }
  };

  double cx, cy, cz, sx, sy, sz;
  calculate_cos_sin(transform(3), cx, sx);
  calculate_cos_sin(transform(4), cy, sy);
  calculate_cos_sin(transform(5), cz, sz);

  if (use_d2d_) {
    R_.row(0) << cy * cz, -sz * cy, sy;
    R_.row(1) << sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy;
    R_.row(2) << sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy;
    t_ = transform.head(3);
  }

  // Precomputed angular gradient components. Letters correspond to Equation 6.19
  // [Magnusson 2009]
  angular_jacobian_.setZero();
  angular_jacobian_.row(1).noalias() = Eigen::Vector4d(
      (-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy), 1.0); // a
  angular_jacobian_.row(2).noalias() = Eigen::Vector4d(
      (cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy), 1.0); // b

  angular_jacobian_.row(3).noalias() =
      Eigen::Vector4d((-sy * cz), sy * sz, cy, 1.0); // c
  angular_jacobian_.row(4).noalias() =
      Eigen::Vector4d(sx * cy * cz, (-sx * cy * sz), sx * sy, 1.0); // d
  angular_jacobian_.row(5).noalias() =
      Eigen::Vector4d((-cx * cy * cz), cx * cy * sz, (-cx * sy), 1.0); // e

  angular_jacobian_.row(6).noalias() =
      Eigen::Vector4d((-cy * sz), (-cy * cz), 0, 1.0); // f
  angular_jacobian_.row(7).noalias() =
      Eigen::Vector4d((cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0, 1.0); // g
  angular_jacobian_.row(8).noalias() =
      Eigen::Vector4d((sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0, 1.0); // h

  if (compute_hessian) {
    // Precomputed angular hessian components. Letters correspond to Equation 6.21 and
    // numbers correspond to row index [Magnusson 2009]
    angular_hessian_.setZero();
    angular_hessian_.row(1).noalias() = Eigen::Vector4d(
        (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy, 0.0f); // a2
    angular_hessian_.row(2).noalias() = Eigen::Vector4d(
        (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy), 0.0f); // a3

    angular_hessian_.row(4).noalias() =
        Eigen::Vector4d((cx * cy * cz), (-cx * cy * sz), (cx * sy), 0.0f); // b2
    angular_hessian_.row(5).noalias() =
        Eigen::Vector4d((sx * cy * cz), (-sx * cy * sz), (sx * sy), 0.0f); // b3

    // The sign of 'sx * sz' in c2 is incorrect in the thesis, and is fixed here.
    angular_hessian_.row(7).noalias() = Eigen::Vector4d(
        (-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0, 0.0f); // c2
    angular_hessian_.row(8).noalias() = Eigen::Vector4d(
        (cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0, 0.0f); // c3

    angular_hessian_.row(9).noalias() =
        Eigen::Vector4d((-cy * cz), (cy * sz), (-sy), 0.0f); // d1
    angular_hessian_.row(10).noalias() =
        Eigen::Vector4d((-sx * sy * cz), (sx * sy * sz), (sx * cy), 0.0f); // d2
    angular_hessian_.row(11).noalias() =
        Eigen::Vector4d((cx * sy * cz), (-cx * sy * sz), (-cx * cy), 0.0f); // d3

    angular_hessian_.row(12).noalias() =
        Eigen::Vector4d((sy * sz), (sy * cz), 0, 0.0f); // e1
    angular_hessian_.row(13).noalias() =
        Eigen::Vector4d((-sx * cy * sz), (-sx * cy * cz), 0, 0.0f); // e2
    angular_hessian_.row(14).noalias() =
        Eigen::Vector4d((cx * cy * sz), (cx * cy * cz), 0, 0.0f); // e3

    angular_hessian_.row(15).noalias() =
        Eigen::Vector4d((-cy * cz), (cy * sz), 0, 0.0f); // f1
    angular_hessian_.row(16).noalias() = Eigen::Vector4d(
        (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0, 0.0f); // f2
    angular_hessian_.row(17).noalias() = Eigen::Vector4d(
        (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0, 0.0f); // f3
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computePointDerivatives(
    const Eigen::Vector3d& x, bool compute_hessian)
{
  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector.
  // Derivative w.r.t. ith element of transform vector corresponds to column i,
  // Equation 6.18 and 6.19 [Magnusson 2009]
  Eigen::Matrix<double, 9, 1> point_angular_jacobian =
      angular_jacobian_ * Eigen::Vector4d(x[0], x[1], x[2], 0.0);
  point_jacobian_.block<3, 3>(0, 0).setIdentity();
  point_jacobian_.col(3) = point_angular_jacobian.segment(0, 3);
  point_jacobian_.col(4) = point_angular_jacobian.segment(3, 3);
  point_jacobian_.col(5) = point_angular_jacobian.segment(6, 3);

  if (compute_hessian) {
    // Vectors from Equation 6.21 [Magnusson 2009]
    // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform
    // vector. Derivative w.r.t. ith and jth elements of transform vector corresponds to
    // the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
    Eigen::Matrix<double, 18, 1> point_angular_hessian =
        angular_hessian_ * Eigen::Vector4d(x[0], x[1], x[2], 0.0);

    point_hessian_.setZero();
    point_hessian_.block<3, 1>(9, 3) = point_angular_hessian.segment(0, 3);
    point_hessian_.block<3, 1>(9, 4) = point_angular_hessian.segment(3, 3);
    point_hessian_.block<3, 1>(9, 5) = point_angular_hessian.segment(6, 3);
    point_hessian_.block<3, 1>(12, 4) = point_angular_hessian.segment(9, 3);
    point_hessian_.block<3, 1>(12, 5) = point_angular_hessian.segment(12, 3);
    point_hessian_.block<3, 1>(15, 5) = point_angular_hessian.segment(15, 3);
    point_hessian_.block<3, 1>(12, 3) = point_hessian_.block<3, 1>(9, 4);
    point_hessian_.block<3, 1>(15, 3) = point_hessian_.block<3, 1>(9, 5);
    point_hessian_.block<3, 1>(15, 4) = point_hessian_.block<3, 1>(12, 5);
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeCovarianceDerivatives(
    const Eigen::Matrix3d& cov, bool compute_hessian)
{
  Eigen::Ref<const Eigen::Matrix<double, 9, 3>> dR(angular_jacobian_.block<9, 3>(0, 0));
  Eigen::Ref<const Eigen::Matrix<double, 18, 3>> ddR(angular_hessian_.block<18, 3>(0, 0));

  Eigen::Matrix<double, 9, 3> za = dR * cov * R_.transpose();
  Zas_.setZero();
  Zas_.block<3, 3>(0, 9) = za.block<3, 3>(0, 0) + za.block<3, 3>(0, 0).transpose();
  Zas_.block<3, 3>(0, 12) = za.block<3, 3>(3, 0) + za.block<3, 3>(3, 0).transpose();
  Zas_.block<3, 3>(0, 15) = za.block<3, 3>(6, 0) + za.block<3, 3>(6, 0).transpose();

  if (compute_hessian) {
    Eigen::Matrix<double, 18, 3> z = ddR * cov * R_.transpose();
    int i = 0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        z.block<3, 3>(i * 3, 0) +=
            Zas_.block<3, 3>(a * 3, 0) * cov * Zas_.block<3, 3>(b * 3, 0).transpose();
        Eigen::Matrix3d tp = z.block<3, 3>(i * 3, 0).transpose();
        z.block<3, 3>(i * 3, 0) += tp;
        ++i;
      }
    }
    Zabs_.setZero();
    Zabs_.block<3, 3>(9, 9) = z.block<3, 3>(0, 0);
    Zabs_.block<3, 3>(9, 12) = Zabs_.block<3, 3>(12, 9) = z.block<3, 3>(3, 0);
    Zabs_.block<3, 3>(9, 15) = Zabs_.block<3, 3>(15, 9) = z.block<3, 3>(6, 0);
    Zabs_.block<3, 3>(12, 12) = z.block<3, 3>(9, 0);
    Zabs_.block<3, 3>(12, 15) = Zabs_.block<3, 3>(15, 12) = z.block<3, 3>(12, 0);
    Zabs_.block<3, 3>(15, 15) = z.block<3, 3>(15, 0);
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::updateDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& x_trans,
    const Eigen::Matrix3d& c_inv,
    bool compute_hessian) const
{
  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  double e_x_cov_x = std::exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
  // Calculate likelihood of transformed points existence, Equation 6.9 [Magnusson
  // 2009]
  const double score_inc = -gauss_d1_ * e_x_cov_x;

  e_x_cov_x = gauss_d2_ * e_x_cov_x;

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x)) {
    return 0;
  }

  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
  e_x_cov_x *= gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson
    // 2009]
    const Eigen::Vector3d cov_dxd_pi = c_inv * point_jacobian_.col(i);

    // Update gradient, Equation 6.12 [Magnusson 2009]
    score_gradient(i) += x_trans.dot(cov_dxd_pi) * e_x_cov_x;

    if (compute_hessian) {
      for (Eigen::Index j = 0; j < hessian.cols(); j++) {
        // Update hessian, Equation 6.13 [Magnusson 2009]
        hessian(i, j) +=
            e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) *
                             x_trans.dot(c_inv * point_jacobian_.col(j)) +
                         x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
                         point_jacobian_.col(j).dot(cov_dxd_pi));
      }
    }
  }

  return score_inc;
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::updateDerivativesD2D(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& uij,
    const Eigen::Matrix3d& B,
    bool compute_hessian) const
{
  Eigen::Ref<const Eigen::Matrix<double, 3, 6>> jas(point_jacobian_);
  Eigen::Ref<const Eigen::Matrix<double, 18, 6>> Habs(point_hessian_);

  Eigen::Transpose<const Eigen::Vector3d> uijT(uij);
  double score = -gauss_d1_ * std::exp(-0.5 * gauss_d2_ * uijT * B * uij);
  for (int a = 0; a < 6; ++a) {
    Eigen::Ref<const Eigen::Vector3d> ja(jas.block<3, 1>(0, a));
    Eigen::Ref<const Eigen::Matrix3d> Za(Zas_.block<3, 3>(0, 3 * a));
    double qa = (2 * uijT * B * ja - uijT * B * Za * B * uij)(0);
    score_gradient(a) += -0.5 * gauss_d2_ * score * qa;

    if (compute_hessian) {
      for (int b = 0; b < 6; ++b) {
        Eigen::Ref<const Eigen::Vector3d> jb(jas.block<3, 1>(0, b));
        Eigen::Transpose<const Eigen::Vector3d> jbT(jas.block<3, 1>(0, b));
        Eigen::Ref<const Eigen::Matrix3d> Zb(Zas_.block<3, 3>(0, 3 * b));
        Eigen::Ref<const Eigen::Vector3d> Hab(Habs.block<3, 1>(3 * a, b));
        Eigen::Ref<const Eigen::Matrix3d> Zab(Zabs_.block<3, 3>(3 * a, 3 * b));
        double qb = (2 * uijT * B * jb - uijT * B * Zb * B * uij)(0);
        hessian(a, b) += -gauss_d2_ * score *
                         ((jbT * B * ja - uijT * B * Zb * B * ja + uijT * B * Hab -
                           uijT * B * Za * B * jb + uijT * B * Za * B * Zb * B * uij -
                           0.5 * uijT * B * Zab * B * uij)(0) -
                          0.25 * gauss_d2_ * qa * qb);
      }
    }
  }
  return score;
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeHessian(
    Eigen::Matrix<double, 6, 6>& hessian, const PointCloudSource& trans_cloud)
{
  hessian.setZero();

  // Precompute Angular Derivatives unnecessary because only used after regular
  // derivative calculation Update hessian for each point, line 17 in Algorithm 2
  // [Magnusson 2009]
  for (std::size_t idx = 0; idx < input_->size(); idx++) {
    // Transformed Point
    const auto& x_trans_pt = trans_cloud[idx];

    // Find neighbors (Radius search has been experimentally faster than direct neighbor
    // checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);

    for (const auto& cell : neighborhood) {
      // Original Point
      const auto& x_pt = (*input_)[idx];
      const Eigen::Vector3d x = x_pt.getVector3fMap().template cast<double>();

      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      const Eigen::Vector3d x_trans =
          x_trans_pt.getVector3fMap().template cast<double>() - cell->getMean();
      // Inverse Covariance of Occupied Voxel
      // Uses precomputed covariance for speed.
      const Eigen::Matrix3d c_inv = cell->getInverseCov();

      // Compute derivative of transform function w.r.t. transform vector, J_E and H_E
      // in Equations 6.18 and 6.20 [Magnusson 2009]
      computePointDerivatives(x);
      // Update hessian, lines 21 in Algorithm 2, according to Equations 6.10, 6.12
      // and 6.13, respectively [Magnusson 2009]
      updateHessian(hessian, x_trans, c_inv);
    }
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeHessianD2D(
    Eigen::Matrix<double, 6, 6>& hessian, const std::vector<TransLeaf>& trans_leaves)
{
  hessian.setZero();

  for (const auto& leaf : trans_leaves) {
    const auto& mean = leaf.mean;
    const auto& cov = leaf.cov;
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    PointSource meanpt;
    meanpt.x = mean(0), meanpt.y = mean(1), meanpt.z = mean(2);
    target_cells_.radiusSearch(meanpt, resolution_, neighborhood, distances);
    // XXX: Nearest Neighbor / Set size 1
    // target_cells_.nearestKSearch(meanpt, 1, neighborhood, distances);
    for (const auto& cell : neighborhood) {
      if (cell->nr_points < 6)
        continue;
      Eigen::Vector3d uij = mean - cell->getMean();
      Eigen::Matrix3d B = (R_ * cov * R_.transpose() + cell->getCov()).inverse();
      computePointDerivatives(uij);
      computeCovarianceDerivatives(cov);
      updateHessianD2D(hessian, uij, B);
    }
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::updateHessian(
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& x_trans,
    const Eigen::Matrix3d& c_inv) const
{
  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  double e_x_cov_x =
      gauss_d2_ * std::exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x)) {
    return;
  }

  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
  e_x_cov_x *= gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson
    // 2009]
    const Eigen::Vector3d cov_dxd_pi = c_inv * point_jacobian_.col(i);

    for (Eigen::Index j = 0; j < hessian.cols(); j++) {
      // Update hessian, Equation 6.13 [Magnusson 2009]
      hessian(i, j) +=
          e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) *
                           x_trans.dot(c_inv * point_jacobian_.col(j)) +
                       x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
                       point_jacobian_.col(j).dot(cov_dxd_pi));
    }
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::updateHessianD2D(
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& uij,
    const Eigen::Matrix3d& B) const
{
  Eigen::Ref<const Eigen::Matrix<double, 3, 6>> jas(point_jacobian_);
  Eigen::Ref<const Eigen::Matrix<double, 18, 6>> Habs(point_hessian_);
  Eigen::Transpose<const Eigen::Vector3d> uijT(uij);
  double d1d2exp = gauss_d1_ * gauss_d2_ * std::exp(-0.5 * gauss_d2_ * uijT * B * uij);
  for (int a = 0; a < 6; ++a) {
    Eigen::Ref<const Eigen::Vector3d> ja(jas.block<3, 1>(0, a));
    Eigen::Ref<const Eigen::Matrix3d> Za(Zas_.block<3, 3>(0, 3 * a));
    double qa = (2 * uijT * B * ja - uijT * B * Za * B * uij)(0);

    for (int b = 0; b < 6; ++b) {
      Eigen::Ref<const Eigen::Vector3d> jb(jas.block<3, 1>(0, b));
      Eigen::Transpose<const Eigen::Vector3d> jbT(jas.block<3, 1>(0, b));
      Eigen::Ref<const Eigen::Matrix3d> Zb(Zas_.block<3, 3>(0, 3 * b));
      Eigen::Ref<const Eigen::Vector3d> Hab(Habs.block<3, 1>(3 * a, b));
      Eigen::Ref<const Eigen::Matrix3d> Zab(Zabs_.block<3, 3>(3 * a, 3 * b));
      double qb = (2 * uijT * B * jb - uijT * B * Zb * B * uij)(0);
      hessian(a, b) +=
          d1d2exp * ((jbT * B * ja - uijT * B * Zb * B * ja + uijT * B * Hab -
                      uijT * B * Za * B * jb + uijT * B * Za * B * Zb * B * uij -
                      0.5 * uijT * B * Zab * B * uij)(0) -
                     0.25 * gauss_d2_ * qa * qb);
    }
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
bool
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::updateIntervalMT(
    double& a_l,
    double& f_l,
    double& g_l,
    double& a_u,
    double& f_u,
    double& g_u,
    double a_t,
    double f_t,
    double g_t) const
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente
  // 1994]
  if (f_t > f_l) {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return false;
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente
  // 1994]
  if (g_t * (a_l - a_t) > 0) {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente
  // 1994]
  if (g_t * (a_l - a_t) < 0) {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Interval Converged
  return true;
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::trialValueSelectionMT(
    double a_l,
    double f_l,
    double g_l,
    double a_u,
    double f_u,
    double g_u,
    double a_t,
    double f_t,
    double g_t) const
{
  if (a_t == a_l && a_t == a_u) {
    return a_t;
  }

  // Endpoints condition check [More, Thuente 1994], p.299 - 300
  enum class EndpointsCondition { Case1, Case2, Case3, Case4 };
  EndpointsCondition condition;

  if (a_t == a_l) {
    condition = EndpointsCondition::Case4;
  }
  else if (f_t > f_l) {
    condition = EndpointsCondition::Case1;
  }
  else if (g_t * g_l < 0) {
    condition = EndpointsCondition::Case2;
  }
  else if (std::fabs(g_t) <= std::fabs(g_l)) {
    condition = EndpointsCondition::Case3;
  }
  else {
    condition = EndpointsCondition::Case4;
  }

  switch (condition) {
  case EndpointsCondition::Case1: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    const double a_q =
        a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l)) {
      return a_c;
    }
    return 0.5 * (a_q + a_c);
  }

  case EndpointsCondition::Case2: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t)) {
      return a_c;
    }
    return a_s;
  }

  case EndpointsCondition::Case3: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t)) {
      a_t_next = a_c;
    }
    else {
      a_t_next = a_s;
    }

    if (a_t > a_l) {
      return std::min(a_t + 0.66 * (a_u - a_t), a_t_next);
    }
    return std::max(a_t + 0.66 * (a_u - a_t), a_t_next);
  }

  default:
  case EndpointsCondition::Case4: {
    // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    const double w = std::sqrt(z * z - g_t * g_u);
    // Equation 2.4.56 [Sun, Yuan 2006]
    return a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w);
  }
  }
}

template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1>& x,
    Eigen::Matrix<double, 6, 1>& step_dir,
    double step_init,
    double step_max,
    double step_min,
    double& score,
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    PointCloudSource& trans_cloud)
{
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  const double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0) {
      return 0;
    }
    // Reverse step direction and calculate optimal step.
    d_phi_0 *= -1;
    step_dir *= -1;
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  const int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
  const double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  const double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval,
  // Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making
  // step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  Eigen::Matrix<double, 6, 1> x_t = x + step_dir * a_t;

  // Convert x_t into matrix form
  convertTransform(x_t, final_transformation_);

  // New transformed point cloud
  transformPointCloud(*input_, trans_cloud, final_transformation_);

  // Updates score, gradient and hessian.  Hessian calculation is unnecessary but
  // testing showed that most step calculations use the initial step suggestion and
  // recalculation the reusable portions of the hessian would intail more computation
  // time.
  std::cout << "MT(0), ";
  std::vector<TransLeaf> trans_leaves;
  if (!use_d2d_) {
    score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, true);
  } else {
    // D2D-NDT transformed source leaves
    Eigen::Affine3d tf(final_transformation_.template cast<double>());
    for (const auto& elem : source_cells_.getLeaves()) {
      if (elem.second.nr_points < 6)
        continue;
      Eigen::Vector3d trans_mean;
      transformPoint(elem.second.mean_, trans_mean, tf);
      Eigen::Matrix3d trans_cov =
          tf.rotation() * elem.second.cov_ * tf.rotation().transpose();
      trans_leaves.push_back(TransLeaf(trans_mean, trans_cov));
    }
    score = computeDerivativesD2D(score_gradient, hessian, trans_leaves, x_t, true);
  }

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergance or a value satisfies
  // the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More,
  // Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations &&
         !(psi_t <= 0 /*Sufficient Decrease*/ &&
           d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    x_t = x + step_dir * a_t;

    // Convert x_t into matrix form
    convertTransform(x_t, final_transformation_);

    // New transformed point cloud
    // Done on final cloud to prevent wasted computation
    transformPointCloud(*input_, trans_cloud, final_transformation_);

    // Updates score, gradient. Values stored to prevent wasted computation.
    std::cout << "MT(n), ";
    if (!use_d2d_) {
      score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, false);
    } else {
      trans_leaves.clear();
      Eigen::Affine3d tf(final_transformation_.template cast<double>());
      for (const auto& elem : source_cells_.getLeaves()) {
        if (elem.second.nr_points < 6)
          continue;
        Eigen::Vector3d trans_mean;
        transformPoint(elem.second.mean_, trans_mean, tf);
        Eigen::Matrix3d trans_cov =
            tf.rotation() * elem.second.cov_ * tf.rotation().transpose();
        trans_leaves.push_back(TransLeaf(trans_mean, trans_cov));
      }
      score = computeDerivativesD2D(score_gradient, hessian, trans_leaves, x_t, false);
    }

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l += phi_0 - mu * d_phi_0 * a_l;
      g_l += mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u += phi_0 - mu * d_phi_0 * a_u;
      g_u += mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente
      // 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnecessary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    if (!use_d2d_) {
      computeHessian(hessian, trans_cloud);
    } else {
      computeHessianD2D(hessian, trans_leaves);
    }
  }

  return a_t;
}

} // namespace pcl

#endif // PCL_REGISTRATION_NDT_IMPL_H_
