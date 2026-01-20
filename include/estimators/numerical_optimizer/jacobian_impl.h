// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "camera_pose.h"
#include "colmap_models.h"
#include "essential.h"
#include "types.h"

namespace poselib {

// For the accumulators we support supplying a vector<double> with point-wise weights for the residuals
// In case we don't want to have weighted residuals, we can pass UniformWeightVector instead of filling a std::vector
// with 1.0 The multiplication is then hopefully is optimized away since it always returns 1.0
class UniformWeightVector {
  public:
    UniformWeightVector() {}
    constexpr double operator[](std::size_t idx) const { return 1.0; }
};
class UniformWeightVectors { // this corresponds to std::vector<std::vector<double>> used for generalized cameras etc
  public:
    UniformWeightVectors() {}
    constexpr const UniformWeightVector &operator[](std::size_t idx) const { return w; }
    const UniformWeightVector w;
    typedef UniformWeightVector value_type;
};

template <typename CameraModel, typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class CameraJacobianAccumulator {
  public:
    CameraJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                              const Camera &cam, const LossFunction &loss,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera(cam), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.apply(X[i]);
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
            CameraModel::project(camera.params, p, &p);
            const double r0 = p(0) - x[i](0);
            const double r1 = p(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        const Eigen::Matrix3d R = pose.R();
        const Eigen::Vector3d &t = pose.t;
        Eigen::Matrix2d Jcam;
        Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
        size_t num_residuals = 0;

        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d &Xi = X[i];
            const Eigen::Vector3d Z = R * Xi + t;

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            const double Z2 = Z(2);
            if (Z2 < 0)
                continue;

            const double inv_Z2 = 1.0 / Z2;
            const Eigen::Vector2d z(Z(0) * inv_Z2, Z(1) * inv_Z2);

            // Project with intrinsics
            Eigen::Vector2d zp = z;
            CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

            // Setup residual
            const Eigen::Vector2d r = zp - x[i];
            const double r_squared = r.squaredNorm();
            const double weight = weights[i] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            Eigen::Matrix<double, 2, 3> dZ;
            dZ.block<2, 2>(0, 0) = Jcam;
            dZ.col(2) = -Jcam * z;
            dZ *= inv_Z2;  // Use pre-computed reciprocal
            dZ *= R;

            const double X0 = X[i](0);
            const double X1 = X[i](1);
            const double X2 = X[i](2);

            const double dZ00 = dZ(0, 0);
            const double dZ01 = dZ(0, 1);
            const double dZ02 = dZ(0, 2);
            const double dZ10 = dZ(1, 0);
            const double dZ11 = dZ(1, 1);
            const double dZ12 = dZ(1, 2);

            const double dZtdZ_0_0 = weight * (dZ00 * dZ00 + dZ10 * dZ10);
            const double dZtdZ_1_0 = weight * (dZ01 * dZ00 + dZ11 * dZ10);
            const double dZtdZ_1_1 = weight * (dZ01 * dZ01 + dZ11 * dZ11);
            const double dZtdZ_2_0 = weight * (dZ02 * dZ00 + dZ12 * dZ10);
            const double dZtdZ_2_1 = weight * (dZ02 * dZ01 + dZ12 * dZ11);
            const double dZtdZ_2_2 = weight * (dZ02 * dZ02 + dZ12 * dZ12);

            const double X0_X0 = X0 * X0;
            const double X0_X1 = X0 * X1;
            const double X0_X2 = X0 * X2;
            const double X1_X1 = X1 * X1;
            const double X1_X2 = X1 * X2;
            const double X2_X2 = X2 * X2;

            const double X2_dZtdZ_1_1 = X2 * dZtdZ_1_1;
            const double X1_dZtdZ_2_1 = X1 * dZtdZ_2_1;
            const double X1_dZtdZ_2_2 = X1 * dZtdZ_2_2;
            const double X2_dZtdZ_2_1 = X2 * dZtdZ_2_1;
            const double X2_dZtdZ_1_0 = X2 * dZtdZ_1_0;
            const double X0_dZtdZ_2_1 = X0 * dZtdZ_2_1;
            const double X0_dZtdZ_2_2 = X0 * dZtdZ_2_2;
            const double X2_dZtdZ_2_0 = X2 * dZtdZ_2_0;
            const double X0_dZtdZ_1_1 = X0 * dZtdZ_1_1;
            const double X1_dZtdZ_2_0 = X1 * dZtdZ_2_0;
            const double X1_dZtdZ_1_0 = X1 * dZtdZ_1_0;
            const double X2_dZtdZ_0_0 = X2 * dZtdZ_0_0;
            const double X0_dZtdZ_2_0 = X0 * dZtdZ_2_0;
            const double X1_dZtdZ_0_0 = X1 * dZtdZ_0_0;
            const double X0_dZtdZ_1_0 = X0 * dZtdZ_1_0;

            JtJ(0, 0) += X2_X2 * dZtdZ_1_1 - X1_X2 * dZtdZ_2_1 - X1_X2 * dZtdZ_2_1 + X1_X1 * dZtdZ_2_2;
            JtJ(1, 0) += -X2_X2 * dZtdZ_1_0 + X0_X2 * dZtdZ_2_1 - X0_X1 * dZtdZ_2_2 + X1_X2 * dZtdZ_2_0;
            JtJ(2, 0) += X0_X1 * dZtdZ_2_1 - X1_X1 * dZtdZ_2_0 - X0_X2 * dZtdZ_1_1 + X1_X2 * dZtdZ_1_0;
            JtJ(3, 0) += X1_dZtdZ_2_0 - X2_dZtdZ_1_0;
            JtJ(4, 0) += X1_dZtdZ_2_1 - X2_dZtdZ_1_1;
            JtJ(5, 0) += X1 * dZtdZ_2_2 - X2_dZtdZ_2_1;
            JtJ(1, 1) += X2_X2 * dZtdZ_0_0 - X0_X2 * dZtdZ_2_0 - X0_X2 * dZtdZ_2_0 + X0_X0 * dZtdZ_2_2;
            JtJ(2, 1) += -X1_X2 * dZtdZ_0_0 + X0_X2 * dZtdZ_1_0 - X0_X0 * dZtdZ_2_1 + X0_X1 * dZtdZ_2_0;
            JtJ(3, 1) += X2_dZtdZ_0_0 - X0_dZtdZ_2_0;
            JtJ(4, 1) += X2 * dZtdZ_1_0 - X0_dZtdZ_2_1;
            JtJ(5, 1) += X2_dZtdZ_2_0 - X0_dZtdZ_2_2;
            JtJ(2, 2) += X1_X1 * dZtdZ_0_0 - X0_X1 * dZtdZ_1_0 - X0_X1 * dZtdZ_1_0 + X0_X0 * dZtdZ_1_1;
            JtJ(3, 2) += X0_dZtdZ_1_0 - X1_dZtdZ_0_0;
            JtJ(4, 2) += X0_dZtdZ_1_1 - X1_dZtdZ_1_0;
            JtJ(5, 2) += X0 * dZtdZ_2_1 - X1_dZtdZ_2_0;
            JtJ(3, 3) += dZtdZ_0_0;
            JtJ(4, 3) += dZtdZ_1_0;
            JtJ(5, 3) += dZtdZ_2_0;
            JtJ(4, 4) += dZtdZ_1_1;
            JtJ(5, 4) += dZtdZ_2_1;
            JtJ(5, 5) += dZtdZ_2_2;

            const double wr0 = weight * r(0);
            const double wr1 = weight * r(1);

            const double X1_dZ02_minus_X2_dZ01 = X1 * dZ02 - X2 * dZ01;
            const double X1_dZ12_minus_X2_dZ11 = X1 * dZ12 - X2 * dZ11;
            const double X0_dZ02_minus_X2_dZ00 = X0 * dZ02 - X2 * dZ00;
            const double X0_dZ12_minus_X2_dZ10 = X0 * dZ12 - X2 * dZ10;
            const double X0_dZ01_minus_X1_dZ00 = X0 * dZ01 - X1 * dZ00;
            const double X0_dZ11_minus_X1_dZ10 = X0 * dZ11 - X1 * dZ10;

            Jtr(0) += wr0 * X1_dZ02_minus_X2_dZ01 + wr1 * X1_dZ12_minus_X2_dZ11;
            Jtr(1) += -wr0 * X0_dZ02_minus_X2_dZ00 - wr1 * X0_dZ12_minus_X2_dZ10;
            Jtr(2) += wr0 * X0_dZ01_minus_X1_dZ00 + wr1 * X0_dZ11_minus_X1_dZ10;
            Jtr(3) += dZ00 * wr0 + dZ10 * wr1;
            Jtr(4) += dZ01 * wr0 + dZ11 * wr1;
            Jtr(5) += dZ02 * wr0 + dZ12 * wr1;
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));

        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const Camera &camera;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedCameraJacobianAccumulator {
  public:
    GeneralizedCameraJacobianAccumulator(const std::vector<std::vector<Point2D>> &points2D,
                                         const std::vector<std::vector<Point3D>> &points3D,
                                         const std::vector<CameraPose> &camera_ext,
                                         const std::vector<Camera> &camera_int, const LossFunction &l,
                                         const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), loss_fn(l),
          weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        cost += accum.residual(full_pose);                                                                             \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        size_t num_residuals = 0;

        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        num_residuals += accum.accumulate(full_pose, JtJ, Jtr);                                                        \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector> class LineJacobianAccumulator {
  public:
    LineJacobianAccumulator(const std::vector<Line2D> &lines2D_, const std::vector<Line3D> &lines3D_,
                            const LossFunction &loss, const ResidualWeightVector &w = ResidualWeightVector())
        : lines2D(lines2D_), lines3D(lines3D_), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d R = pose.R();
        double cost = 0;
        for (size_t i = 0; i < lines2D.size(); ++i) {
            const Eigen::Vector3d Z1 = R * lines3D[i].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[i].X2 + pose.t;
            Eigen::Vector3d l = Z1.cross(Z2);
            l /= l.topRows<2>().norm();

            const double r0 = l.dot(lines2D[i].x1.homogeneous());
            const double r1 = l.dot(lines2D[i].x2.homogeneous());
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {

        Eigen::Matrix3d E, R;
        R = pose.R();
        E << pose.t.cross(R.col(0)), pose.t.cross(R.col(1)), pose.t.cross(R.col(2));
        size_t num_residuals = 0;
        for (size_t k = 0; k < lines2D.size(); ++k) {
            const Eigen::Vector3d Z1 = R * lines3D[k].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[k].X2 + pose.t;

            const Eigen::Vector3d X12 = lines3D[k].X1.cross(lines3D[k].X2);
            const Eigen::Vector3d dX = lines3D[k].X1 - lines3D[k].X2;

            // Projected line
            const Eigen::Vector3d l = Z1.cross(Z2);

            // Normalized line by first two coordinates
            Eigen::Vector2d alpha = l.topRows<2>();
            double beta = l(2);
            const double n_alpha = alpha.norm();
            alpha /= n_alpha;
            beta /= n_alpha;

            // Compute residual
            Eigen::Vector2d r;
            r << alpha.dot(lines2D[k].x1) + beta, alpha.dot(lines2D[k].x2) + beta;

            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            Eigen::Matrix<double, 3, 6> dl_drt;
            // Differentiate line with respect to rotation parameters
            dl_drt.block<1, 3>(0, 0) = E.row(0).cross(dX) - R.row(0).cross(X12);
            dl_drt.block<1, 3>(1, 0) = E.row(1).cross(dX) - R.row(1).cross(X12);
            dl_drt.block<1, 3>(2, 0) = E.row(2).cross(dX) - R.row(2).cross(X12);
            // and translation params
            dl_drt.block<1, 3>(0, 3) = R.row(0).cross(dX);
            dl_drt.block<1, 3>(1, 3) = R.row(1).cross(dX);
            dl_drt.block<1, 3>(2, 3) = R.row(2).cross(dX);

            // Differentiate normalized line w.r.t. original line
            Eigen::Matrix3d dln_dl;
            dln_dl.block<2, 2>(0, 0) = (Eigen::Matrix2d::Identity() - alpha * alpha.transpose()) / n_alpha;
            dln_dl.block<1, 2>(2, 0) = -beta * alpha / n_alpha;
            dln_dl.block<2, 1>(0, 2).setZero();
            dln_dl(2, 2) = 1 / n_alpha;

            // Differentiate residual w.r.t. line
            Eigen::Matrix<double, 2, 3> dr_dl;
            dr_dl.row(0) << lines2D[k].x1.transpose(), 1.0;
            dr_dl.row(1) << lines2D[k].x2.transpose(), 1.0;

            Eigen::Matrix<double, 2, 6> J = dr_dl * dln_dl * dl_drt;
            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 6; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename PointLossFunction, typename LineLossFunction, typename PointResidualsVector = UniformWeightVector,
          typename LineResidualsVector = UniformWeightVector>
class PointLineJacobianAccumulator {
  public:
    PointLineJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                 const PointLossFunction &l_point, const LineLossFunction &l_line,
                                 const PointResidualsVector &weights_pts = PointResidualsVector(),
                                 const LineResidualsVector &weights_l = LineResidualsVector())
        : pts_accum(points2D, points3D, trivial_camera, l_point, weights_pts),
          line_accum(lines2D, lines3D, l_line, weights_l) {
        trivial_camera.model_id = NullCameraModel::model_id;
    }

    double residual(const CameraPose &pose) const { return pts_accum.residual(pose) + line_accum.residual(pose); }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return pts_accum.accumulate(pose, JtJ, Jtr) + line_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        // Both CameraJacobianAccumulator and LineJacobianAccumulator have the same step!
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    CameraJacobianAccumulator<NullCameraModel, PointLossFunction, PointResidualsVector> pts_accum;
    LineJacobianAccumulator<LineLossFunction, LineResidualsVector> line_accum;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                    const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class SharedFocalRelativePoseJacobianAccumulator {
  public:
    SharedFocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                               const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const ImagePair &image_pair) const {
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();

        Eigen::Matrix3d F = K_inv * (E * K_inv);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const ImagePair &image_pair, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.y())) {
            // x < y
            if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(image_pair.pose.t.y()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(image_pair.pose.t).normalized();

        double focal = image_pair.camera1.focal();
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, focal;

        Eigen::Matrix3d E, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d F = K_inv * (E * K_inv);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;

        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(0, 5) = dF * df;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
            JtJ(5, 0) += weight * (J(5) * J(0));
            JtJ(5, 1) += weight * (J(5) * J(1));
            JtJ(5, 2) += weight * (J(5) * J(2));
            JtJ(5, 3) += weight * (J(5) * J(3));
            JtJ(5, 4) += weight * (J(5) * J(4));
            JtJ(5, 5) += weight * (J(5) * J(5));
        }
        return num_residuals;
    }

    ImagePair step(Eigen::Matrix<double, 6, 1> dp, const ImagePair &image_pair) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera camera_new =
            Camera("SIMPLE_PINHOLE",
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(5, 0), 0.0), 0.0, 0.0}, -1, -1);
        ImagePair calib_pose_new(pose_new, camera_new, camera_new);
        return calib_pose_new;
    }
    typedef ImagePair param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedRelativePoseJacobianAccumulator {
  public:
    GeneralizedRelativePoseJacobianAccumulator(const std::vector<PairwiseMatches> &pairwise_matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<CameraPose> &camera2_ext, const LossFunction &l,
                                               const ResidualWeightVectors &w = ResidualWeightVectors())
        : matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];
            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + quat_rotate(q2, pose.t) - relpose.rotate(t1);
            RelativePoseJacobianAccumulator<LossFunction, typename ResidualWeightVectors::value_type> accum(
                m.x1, m.x2, loss_fn, weights[match_k]);
            cost += accum.residual(relpose);
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];

            // Cameras are
            // [R1 t1]
            // [R2 t2] * [R t; 0 1] = [R2*R t2+R2*t]

            // Relative pose is
            // [R2*R*R1' t2+R2*t-R2*R*R1'*t1]
            // Essential matrix is
            // [t2]_x*R2*R*R1' + [R2*t]_x*R2*R*R1' - R2*R*R1'*[t1]_x

            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Matrix3d R1 = quat_to_rotmat(q1);
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Matrix3d R2 = quat_to_rotmat(q2);
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + R2 * pose.t - relpose.rotate(t1);
            Eigen::Matrix3d E;
            essential_from_motion(relpose, &E);

            Eigen::Matrix3d R2R = R2 * R;
            Eigen::Vector3d Rt = R.transpose() * pose.t;

            // The messy expressions below compute
            // dRdw = [vec(S1) vec(S2) vec(S3)];
            // dR = (kron(R1,skew(t2)*R2R+ R2*skew(t)*R) + kron(skew(t1)*R1,R2*R)) * dRdw
            // dt = [vec(R2*R*S1*R1.') vec(R2*R*S2*R1.') vec(R2*R*S3*R1.')]

            // TODO: Replace with something nice
            Eigen::Matrix<double, 9, 3> dR;
            Eigen::Matrix<double, 9, 3> dt;
            dR(0, 0) = R2R(0, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(0, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(0, 1) = R2R(0, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(0, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(0, 2) = R2R(0, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(0, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(0, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(1, 0) = R2R(1, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(1, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(1, 1) = R2R(1, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(1, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(1, 2) = R2R(1, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(1, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(0, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(2, 0) = R2R(2, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(2, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(2, 1) = R2R(2, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(2, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(2, 2) = R2R(2, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(2, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(0, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(3, 0) = R2R(0, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(3, 1) = R2R(0, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(3, 2) = R2R(0, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(1, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(4, 0) = R2R(1, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(4, 1) = R2R(1, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(4, 2) = R2R(1, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(1, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(5, 0) = R2R(2, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(5, 1) = R2R(2, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(5, 2) = R2R(2, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(1, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(6, 0) = R2R(0, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(6, 1) = R2R(0, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(6, 2) = R2R(0, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(2, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(7, 0) = R2R(1, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(7, 1) = R2R(1, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(7, 2) = R2R(1, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(2, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(8, 0) = R2R(2, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(8, 1) = R2R(2, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(8, 2) = R2R(2, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(2, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dt(0, 0) = R2R(0, 2) * R1(0, 1) - R2R(0, 1) * R1(0, 2);
            dt(0, 1) = R2R(0, 0) * R1(0, 2) - R2R(0, 2) * R1(0, 0);
            dt(0, 2) = R2R(0, 1) * R1(0, 0) - R2R(0, 0) * R1(0, 1);
            dt(1, 0) = R2R(1, 2) * R1(0, 1) - R2R(1, 1) * R1(0, 2);
            dt(1, 1) = R2R(1, 0) * R1(0, 2) - R2R(1, 2) * R1(0, 0);
            dt(1, 2) = R2R(1, 1) * R1(0, 0) - R2R(1, 0) * R1(0, 1);
            dt(2, 0) = R2R(2, 2) * R1(0, 1) - R2R(2, 1) * R1(0, 2);
            dt(2, 1) = R2R(2, 0) * R1(0, 2) - R2R(2, 2) * R1(0, 0);
            dt(2, 2) = R2R(2, 1) * R1(0, 0) - R2R(2, 0) * R1(0, 1);
            dt(3, 0) = R2R(0, 2) * R1(1, 1) - R2R(0, 1) * R1(1, 2);
            dt(3, 1) = R2R(0, 0) * R1(1, 2) - R2R(0, 2) * R1(1, 0);
            dt(3, 2) = R2R(0, 1) * R1(1, 0) - R2R(0, 0) * R1(1, 1);
            dt(4, 0) = R2R(1, 2) * R1(1, 1) - R2R(1, 1) * R1(1, 2);
            dt(4, 1) = R2R(1, 0) * R1(1, 2) - R2R(1, 2) * R1(1, 0);
            dt(4, 2) = R2R(1, 1) * R1(1, 0) - R2R(1, 0) * R1(1, 1);
            dt(5, 0) = R2R(2, 2) * R1(1, 1) - R2R(2, 1) * R1(1, 2);
            dt(5, 1) = R2R(2, 0) * R1(1, 2) - R2R(2, 2) * R1(1, 0);
            dt(5, 2) = R2R(2, 1) * R1(1, 0) - R2R(2, 0) * R1(1, 1);
            dt(6, 0) = R2R(0, 2) * R1(2, 1) - R2R(0, 1) * R1(2, 2);
            dt(6, 1) = R2R(0, 0) * R1(2, 2) - R2R(0, 2) * R1(2, 0);
            dt(6, 2) = R2R(0, 1) * R1(2, 0) - R2R(0, 0) * R1(2, 1);
            dt(7, 0) = R2R(1, 2) * R1(2, 1) - R2R(1, 1) * R1(2, 2);
            dt(7, 1) = R2R(1, 0) * R1(2, 2) - R2R(1, 2) * R1(2, 0);
            dt(7, 2) = R2R(1, 1) * R1(2, 0) - R2R(1, 0) * R1(2, 1);
            dt(8, 0) = R2R(2, 2) * R1(2, 1) - R2R(2, 1) * R1(2, 2);
            dt(8, 1) = R2R(2, 0) * R1(2, 2) - R2R(2, 2) * R1(2, 0);
            dt(8, 2) = R2R(2, 1) * R1(2, 0) - R2R(2, 0) * R1(2, 1);

            for (size_t k = 0; k < m.x1.size(); ++k) {
                double C = m.x2[k].homogeneous().dot(E * m.x1[k].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * m.x2[k].homogeneous(),
                    E.block<2, 3>(0, 0) * m.x1[k].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                const double weight = weights[match_k][k] * loss_fn.weight(r * r);
                if (weight == 0.0) {
                    continue;
                }
                num_residuals++;

                // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                Eigen::Matrix<double, 1, 9> dF;
                dF << m.x1[k](0) * m.x2[k](0), m.x1[k](0) * m.x2[k](1), m.x1[k](0), m.x1[k](1) * m.x2[k](0),
                    m.x1[k](1) * m.x2[k](1), m.x1[k](1), m.x2[k](0), m.x2[k](1), 1.0;
                const double s = C * inv_nJ_C * inv_nJ_C;
                dF(0) -= s * (J_C(2) * m.x1[k](0) + J_C(0) * m.x2[k](0));
                dF(1) -= s * (J_C(3) * m.x1[k](0) + J_C(0) * m.x2[k](1));
                dF(2) -= s * (J_C(0));
                dF(3) -= s * (J_C(2) * m.x1[k](1) + J_C(1) * m.x2[k](0));
                dF(4) -= s * (J_C(3) * m.x1[k](1) + J_C(1) * m.x2[k](1));
                dF(5) -= s * (J_C(1));
                dF(6) -= s * (J_C(2));
                dF(7) -= s * (J_C(3));
                dF *= inv_nJ_C;

                // and then w.r.t. the pose parameters
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = dF * dR;
                J.block<1, 3>(0, 3) = dF * dt;

                // Accumulate into JtJ and Jtr
                Jtr += weight * C * inv_nJ_C * J.transpose();
                for (size_t i = 0; i < 6; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        JtJ(i, j) += weight * (J(i) * J(j));
                    }
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename AbsResidualsVector = UniformWeightVector,
          typename RelResidualsVectors = UniformWeightVectors>
class HybridPoseJacobianAccumulator {
  public:
    HybridPoseJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                  const std::vector<PairwiseMatches> &pairwise_matches,
                                  const std::vector<CameraPose> &map_ext, const LossFunction &l,
                                  const LossFunction &l_epi,
                                  const AbsResidualsVector &weights_abs = AbsResidualsVector(),
                                  const RelResidualsVectors &weights_rel = RelResidualsVectors())
        : abs_pose_accum(points2D, points3D, trivial_camera, l, weights_abs),
          gen_rel_accum(pairwise_matches, map_ext, trivial_rig, l_epi, weights_rel) {
        trivial_camera.model_id = NullCameraModel::model_id;
        trivial_rig.emplace_back();
    }

    double residual(const CameraPose &pose) const {
        return abs_pose_accum.residual(pose) + gen_rel_accum.residual(pose);
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return abs_pose_accum.accumulate(pose, JtJ, Jtr) + gen_rel_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    std::vector<CameraPose> trivial_rig;
    CameraJacobianAccumulator<NullCameraModel, LossFunction, AbsResidualsVector> abs_pose_accum;
    GeneralizedRelativePoseJacobianAccumulator<LossFunction, RelResidualsVectors> gen_rel_accum;
};

// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        if (U.determinant() < 0) {
            U = -U;
        }
        if (V.determinant() < 0) {
            V = -V;
        }
        qU = rotmat_to_quat(U);
        qV = rotmat_to_quat(V);
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        Eigen::Matrix3d U = quat_to_rotmat(qU);
        Eigen::Matrix3d V = quat_to_rotmat(qV);
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Vector4d qU, qV;
    double sigma;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class FundamentalJacobianAccumulator {
  public:
    FundamentalJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ,
                      Eigen::Matrix<double, 7, 1> &Jtr) const {

        const Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        const Eigen::Matrix3d U = quat_to_rotmat(FF.qU);
        const Eigen::Matrix3d V = quat_to_rotmat(FF.qV);

        const Eigen::Matrix3d d_sigma = U.col(1) * V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0), -F(2, 0), 0, F(0, 0), 0, F(1, 2),
            -F(1, 1), d_sigma(1, 0), F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0), 0, F(2, 1), -F(1, 1),
            -F(0, 2), 0, F(0, 0), d_sigma(0, 1), -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1), F(1, 1),
            -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1), 0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2), F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0,
            d_sigma(2, 2);

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    FactorizedFundamentalMatrix step(Eigen::Matrix<double, 7, 1> dp, const FactorizedFundamentalMatrix &F) const {
        FactorizedFundamentalMatrix F_new;
        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);
        return F_new;
    }
    typedef FactorizedFundamentalMatrix param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Non-linear refinement of transfer error |x2 - pi(H*x1)|^2, parameterized by fixing H(2,2) = 1
// I did some preliminary experiments comparing different error functions (e.g. symmetric and transfer)
// as well as other parameterizations (different affine patches, SVD as in Bartoli/Sturm, etc)
// but it does not seem to have a big impact (and is sometimes even worse)
// Implementations of these can be found at https://github.com/vlarsson/homopt
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class HomographyJacobianAccumulator {
  public:
    HomographyJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                  const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const Eigen::Matrix3d &H) const {
        double cost = 0.0;

        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
            const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;
            cost += weights[k] * loss_fn.loss(r2);
        }
        return cost;
    }

    size_t accumulate(const Eigen::Matrix3d &H, Eigen::Matrix<double, 8, 8> &JtJ, Eigen::Matrix<double, 8, 1> &Jtr) {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r2);
            if (weight == 0.0)
                continue;
            num_residuals++;

            dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
            dH = dH * inv_Hx1_2;

            // accumulate into JtJ and Jtr
            Jtr += dH.transpose() * (weight * Eigen::Vector2d(r0, r1));
            for (size_t i = 0; i < 8; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * dH.col(i).dot(dH.col(j));
                }
            }
        }
        return num_residuals;
    }

    Eigen::Matrix3d step(Eigen::Matrix<double, 8, 1> dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }
    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class Radial1DJacobianAccumulator {
  public:
    Radial1DJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        Eigen::Matrix3d R = pose.R();
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector2d z = (R * X[k] + pose.t).template topRows<2>().normalized();
            double alpha = z.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;
            double r2 = (alpha * z - x[k]).squaredNorm();
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ,
                      Eigen::Matrix<double, 5, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector3d RX = R * X[k];
            const Eigen::Vector2d z = (RX + pose.t).topRows<2>();

            const double n_z = z.norm();
            const Eigen::Vector2d zh = z / n_z;
            const double alpha = zh.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;

            // Setup residual
            Eigen::Vector2d r = alpha * zh - x[k];
            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // differentiate residual with respect to z
            Eigen::Matrix2d dr_dz = (zh * x[k].transpose() + alpha * Eigen::Matrix2d::Identity()) *
                                    (Eigen::Matrix2d::Identity() - zh * zh.transpose()) / n_z;

            Eigen::Matrix<double, 2, 5> dz;
            dz << 0.0, RX(2), -RX(1), 1.0, 0.0, -RX(2), 0.0, RX(0), 0.0, 1.0;

            Eigen::Matrix<double, 2, 5> J = dr_dz * dz;

            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_pre(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t(0) = pose.t(0) + dp(3);
        pose_new.t(1) = pose.t(1) + dp(4);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Parameter structure for radial fundamental matrix (F + 2 distortion parameters)
struct RadialFundamentalMatrixParams {
    RadialFundamentalMatrixParams()
        : F(Eigen::Matrix3d::Identity()), lam1(0.0), lam2(0.0) {}

    RadialFundamentalMatrixParams(const Eigen::Matrix3d &kF, double kLam1, double kLam2)
        : F(kF), lam1(kLam1), lam2(kLam2) {}

    Eigen::Matrix3d F;
    double lam1;
    double lam2;
};

// Jacobian accumulator for radial fundamental matrix refinement
// Refines F and lambda1, lambda2 distortion parameters
// Uses division distortion model: x_dist = [u, v, 1 + lam1*r^2 + lam2*r^4]
// This must match the model used in the 9-point solver and estimator
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class RadialFundamentalMatrixJacobianAccumulator {
  public:
    RadialFundamentalMatrixJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                               const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const RadialFundamentalMatrixParams &params) const {
        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            // Apply division distortion model: [u, v, 1 + lam*r^2]
            double u1 = x1[k](0), v1 = x1[k](1);
            double r2_1 = u1 * u1 + v1 * v1;
            double z1 = 1.0 + params.lam1 * r2_1;
            Eigen::Vector3d p1_dist(u1, v1, z1);

            double u2 = x2[k](0), v2 = x2[k](1);
            double r2_2 = u2 * u2 + v2 * v2;
            double z2 = 1.0 + params.lam2 * r2_2;
            Eigen::Vector3d p2_dist(u2, v2, z2);

            // Epipolar constraint: p2^T * F * p1 = 0
            double C = p2_dist.dot(params.F * p1_dist);

            // Compute Sampson error denominator
            Eigen::Vector3d Fp1 = params.F * p1_dist;
            Eigen::Vector3d FTp2 = params.F.transpose() * p2_dist;
            double denom = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1) +
                           FTp2(0) * FTp2(0) + FTp2(1) * FTp2(1);

            if (denom < 1e-10)
                continue;

            double r2 = (C * C) / denom;
            cost += weights[k] * loss_fn.loss(r2);
        }
        return cost;
    }

    size_t accumulate(const RadialFundamentalMatrixParams &params, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) const {
        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            // Apply division distortion model: [u, v, 1 + lam*r^2]
            double u1 = x1[k](0), v1 = x1[k](1);
            double r2_1 = u1 * u1 + v1 * v1;
            double z1 = 1.0 + params.lam1 * r2_1;
            Eigen::Vector3d p1_dist(u1, v1, z1);

            double u2 = x2[k](0), v2 = x2[k](1);
            double r2_2 = u2 * u2 + v2 * v2;
            double z2 = 1.0 + params.lam2 * r2_2;
            Eigen::Vector3d p2_dist(u2, v2, z2);

            // Epipolar constraint
            double C = p2_dist.dot(params.F * p1_dist);

            // Compute Fp1 and F^T p2 for Sampson error
            Eigen::Vector3d Fp1 = params.F * p1_dist;
            Eigen::Vector3d FTp2 = params.F.transpose() * p2_dist;

            // Sampson denominator uses only first two components
            double nJ_C_sq = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1) +
                             FTp2(0) * FTp2(0) + FTp2(1) * FTp2(1);

            if (nJ_C_sq < 1e-10)
                continue;

            double inv_nJ_C = 1.0 / std::sqrt(nJ_C_sq);
            double r = C * inv_nJ_C;
            double weight = weights[k] * loss_fn.weight(r * r);

            if (weight == 0.0)
                continue;

            num_residuals++;

            // Jacobian of epipolar constraint w.r.t F (9 elements in row-major order)
            // C = p2^T * F * p1, so dC/dF[i,j] = p2[i] * p1[j]
            Eigen::Matrix<double, 1, 9> dC_dF;
            dC_dF << p2_dist(0) * p1_dist(0), p2_dist(0) * p1_dist(1), p2_dist(0) * p1_dist(2),
                     p2_dist(1) * p1_dist(0), p2_dist(1) * p1_dist(1), p2_dist(1) * p1_dist(2),
                     p2_dist(2) * p1_dist(0), p2_dist(2) * p1_dist(1), p2_dist(2) * p1_dist(2);

            // Jacobian of denominator w.r.t F
            // nJ_C_sq = (F*p1)[0]^2 + (F*p1)[1]^2 + (F^T*p2)[0]^2 + (F^T*p2)[1]^2
            // d(nJ_C_sq)/dF[i,j] = 2*(F*p1)[i]*p1[j] (for i<2) + 2*(F^T*p2)[j]*p2[i] (for j<2)
            Eigen::Matrix<double, 1, 9> dnJ_C_sq_dF;
            dnJ_C_sq_dF.setZero();
            // Contribution from (F*p1)[0]^2 and (F*p1)[1]^2
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    dnJ_C_sq_dF(i * 3 + j) += 2.0 * Fp1(i) * p1_dist(j);
                }
            }
            // Contribution from (F^T*p2)[0]^2 and (F^T*p2)[1]^2
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 2; ++j) {
                    dnJ_C_sq_dF(i * 3 + j) += 2.0 * FTp2(j) * p2_dist(i);
                }
            }

            // Sampson error: E = C / sqrt(nJ_C_sq)
            // dE/dF = (dC/dF * sqrt(nJ_C_sq) - C * 0.5 * dnJ_C_sq/dF / sqrt(nJ_C_sq)) / nJ_C_sq
            //       = dC/dF / sqrt(nJ_C_sq) - 0.5 * C * dnJ_C_sq/dF / nJ_C_sq^(3/2)
            double inv_nJ_C_cubed = inv_nJ_C * inv_nJ_C * inv_nJ_C;
            Eigen::Matrix<double, 1, 9> dE_dF = dC_dF * inv_nJ_C - 0.5 * C * dnJ_C_sq_dF * inv_nJ_C_cubed;

            // Jacobian w.r.t. lam1 and lam2
            // Division model: p1_dist = [u1, v1, 1 + lam1*r2_1]
            // d(p1_dist)/d(lam1) = [0, 0, r2_1]
            Eigen::Vector3d dp1_dlam1(0.0, 0.0, r2_1);
            Eigen::Vector3d dp2_dlam2(0.0, 0.0, r2_2);

            // dC/dlam1 = p2^T * F * dp1_dlam1 = p2^T * F * [0,0,r2_1]^T = r2_1 * (p2^T * F)[:,2]
            double dC_dlam1 = p2_dist.dot(params.F * dp1_dlam1);
            // dC/dlam2 = dp2_dlam2^T * F * p1 = r2_2 * (F * p1)[2] ... no wait
            // dC/dlam2 = dp2_dlam2^T * F * p1 = [0,0,r2_2] * F * p1 = r2_2 * Fp1(2)
            double dC_dlam2 = dp2_dlam2.dot(params.F * p1_dist);

            // Derivative of denominator w.r.t. lambda
            // d(Fp1)/dlam1 = F * [0,0,r2_1]^T = r2_1 * F.col(2)
            Eigen::Vector3d dFp1_dlam1 = params.F.col(2) * r2_1;
            // d(FTp2)/dlam2 = F^T * [0,0,r2_2]^T = r2_2 * F.row(2)^T
            Eigen::Vector3d dFTp2_dlam2 = params.F.row(2).transpose() * r2_2;

            // dnJ_C_sq/dlam1 = 2 * (Fp1[0]*dFp1_dlam1[0] + Fp1[1]*dFp1_dlam1[1])
            double dnJ_C_sq_dlam1 = 2.0 * (Fp1(0) * dFp1_dlam1(0) + Fp1(1) * dFp1_dlam1(1));
            // dnJ_C_sq/dlam2 = 2 * (FTp2[0]*dFTp2_dlam2[0] + FTp2[1]*dFTp2_dlam2[1])
            double dnJ_C_sq_dlam2 = 2.0 * (FTp2(0) * dFTp2_dlam2(0) + FTp2(1) * dFTp2_dlam2(1));

            // dE/dlam = dC/dlam / sqrt(nJ_C_sq) - 0.5 * C * dnJ_C_sq/dlam / nJ_C_sq^(3/2)
            double dE_dlam1 = dC_dlam1 * inv_nJ_C - 0.5 * C * dnJ_C_sq_dlam1 * inv_nJ_C_cubed;
            double dE_dlam2 = dC_dlam2 * inv_nJ_C - 0.5 * C * dnJ_C_sq_dlam2 * inv_nJ_C_cubed;

            Eigen::Matrix<double, 1, 11> J;
            J.block<1, 9>(0, 0) = dE_dF;
            J(0, 9) = dE_dlam1;
            J(0, 10) = dE_dlam2;

            // Accumulate into JtJ and Jtr
            // residual is r = C / sqrt(nJ_C_sq)
            Jtr += weight * r * J.transpose();
            for (size_t i = 0; i < 11; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    RadialFundamentalMatrixParams step(Eigen::Matrix<double, 11, 1> dp,
                                       const RadialFundamentalMatrixParams &params) const {
        RadialFundamentalMatrixParams params_new;

        // Update F matrix (reshape dp[0:9] as 3x3 row-major)
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> dF(dp.data());
        params_new.F = params.F + dF;

        // Update distortion parameters
        params_new.lam1 = params.lam1 + dp(9);
        params_new.lam2 = params.lam2 + dp(10);

        return params_new;
    }

    typedef RadialFundamentalMatrixParams param_t;
    static constexpr size_t num_params = 11;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Parameter structure for focal fundamental matrix (F + 2 focal lengths)
// The model normalizes image coordinates by focal lengths:
// [u1/f1, v1/f1, 1]^T and [u2/f2, v2/f2, 1]^T
struct FocalFundamentalMatrixParams {
    FocalFundamentalMatrixParams()
        : F(Eigen::Matrix3d::Identity()), f1(1.0), f2(1.0) {}

    FocalFundamentalMatrixParams(const Eigen::Matrix3d &kF, double kF1, double kF2)
        : F(kF), f1(kF1), f2(kF2) {}

    Eigen::Matrix3d F;
    double f1;  // Focal length of camera 1
    double f2;  // Focal length of camera 2
};

// Jacobian accumulator for focal fundamental matrix refinement
// Refines F and focal lengths f1, f2 using the essential matrix constraint.
//
// The key insight is that F relates to the essential matrix E via:
// E = K2^(-T) * F * K1^(-1) where K = diag(f, f, 1) for simple focal length model
//
// The essential matrix E must satisfy: E has two equal singular values (rank-2 with σ1 = σ2)
// This provides the missing constraint to resolve the scale ambiguity.
//
// We parameterize F = K2^T * E * K1 where E is a valid essential matrix
// and optimize over E's 5 DOF (via 3x3 matrix with constraints) plus f1, f2.
//
// For efficiency, we use an alternative formulation:
// - Optimize F directly with pixel-space Sampson error
// - Add a soft constraint that the derived E = K2^(-T) * F * K1^(-1) satisfies the essential matrix constraint
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class FocalFundamentalMatrixJacobianAccumulator {
  public:
    FocalFundamentalMatrixJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                              const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                              const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    // Compute essential matrix from F and focal lengths
    // E = K2^(-T) * F * K1^(-1) = diag(1/f2, 1/f2, 1) * F * diag(1/f1, 1/f1, 1)
    static Eigen::Matrix3d computeEssential(const Eigen::Matrix3d &F, double f1, double f2) {
        Eigen::Matrix3d K1_inv = Eigen::Matrix3d::Identity();
        K1_inv(0, 0) = 1.0 / f1;
        K1_inv(1, 1) = 1.0 / f1;

        Eigen::Matrix3d K2_inv_T = Eigen::Matrix3d::Identity();
        K2_inv_T(0, 0) = 1.0 / f2;
        K2_inv_T(1, 1) = 1.0 / f2;

        return K2_inv_T * F * K1_inv;
    }

    // Essential matrix constraint residual: (σ1 - σ2)^2 where σ1, σ2 are the two largest singular values
    // For a valid essential matrix, σ1 = σ2
    static double essentialConstraintResidual(const Eigen::Matrix3d &E) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
        Eigen::Vector3d s = svd.singularValues();
        // Normalize by the average to make it scale-independent
        double avg = (s(0) + s(1)) / 2.0;
        if (avg < 1e-10) return 0.0;
        double diff = (s(0) - s(1)) / avg;
        return diff * diff;
    }

    double residual(const FocalFundamentalMatrixParams &params) const {
        double cost = 0.0;
        double inv_f1 = 1.0 / params.f1;
        double inv_f2 = 1.0 / params.f2;

        // Use normalized Sampson error - points normalized by focal lengths
        for (size_t k = 0; k < x1.size(); ++k) {
            // Normalize points by focal lengths
            double u1 = x1[k](0) * inv_f1, v1 = x1[k](1) * inv_f1;
            double u2 = x2[k](0) * inv_f2, v2 = x2[k](1) * inv_f2;
            Eigen::Vector3d p1(u1, v1, 1.0);
            Eigen::Vector3d p2(u2, v2, 1.0);

            // Epipolar constraint: p2^T * F * p1 = 0
            double C = p2.dot(params.F * p1);

            // Compute Sampson error denominator
            Eigen::Vector3d Fp1 = params.F * p1;
            Eigen::Vector3d FTp2 = params.F.transpose() * p2;
            double denom = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1) +
                           FTp2(0) * FTp2(0) + FTp2(1) * FTp2(1);

            if (denom < 1e-10)
                continue;

            double r2 = (C * C) / denom;
            cost += weights[k] * loss_fn.loss(r2);
        }

        // Add essential matrix constraint with weight proportional to number of points
        // This ensures the scale ambiguity is resolved
        Eigen::Matrix3d E = computeEssential(params.F, params.f1, params.f2);
        double essential_weight = 100.0 * x1.size();  // Strong weight to enforce constraint
        cost += essential_weight * essentialConstraintResidual(E);

        return cost;
    }

    size_t accumulate(const FocalFundamentalMatrixParams &params, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) const {
        size_t num_residuals = 0;
        double inv_f1 = 1.0 / params.f1;
        double inv_f2 = 1.0 / params.f2;
        double inv_f1_sq = inv_f1 * inv_f1;
        double inv_f2_sq = inv_f2 * inv_f2;

        // Accumulate normalized Sampson error residuals
        // Points are normalized by focal lengths: [u/f, v/f, 1]
        for (size_t k = 0; k < x1.size(); ++k) {
            // Original pixel coordinates
            double px1 = x1[k](0), py1 = x1[k](1);
            double px2 = x2[k](0), py2 = x2[k](1);

            // Normalized points
            double u1 = px1 * inv_f1, v1 = py1 * inv_f1;
            double u2 = px2 * inv_f2, v2 = py2 * inv_f2;
            Eigen::Vector3d p1(u1, v1, 1.0);
            Eigen::Vector3d p2(u2, v2, 1.0);

            // Epipolar constraint on normalized coordinates
            double C = p2.dot(params.F * p1);

            // Compute Fp1 and F^T p2 for Sampson error
            Eigen::Vector3d Fp1 = params.F * p1;
            Eigen::Vector3d FTp2 = params.F.transpose() * p2;

            // Sampson denominator uses only first two components
            double nJ_C_sq = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1) +
                             FTp2(0) * FTp2(0) + FTp2(1) * FTp2(1);

            if (nJ_C_sq < 1e-10)
                continue;

            double inv_nJ_C = 1.0 / std::sqrt(nJ_C_sq);
            double r = C * inv_nJ_C;
            double weight = weights[k] * loss_fn.weight(r * r);

            if (weight == 0.0)
                continue;

            num_residuals++;

            // Jacobian of epipolar constraint w.r.t F (9 elements in row-major order)
            // C = p2^T * F * p1, so dC/dF[i,j] = p2[i] * p1[j]
            Eigen::Matrix<double, 1, 9> dC_dF;
            dC_dF << p2(0) * p1(0), p2(0) * p1(1), p2(0) * p1(2),
                     p2(1) * p1(0), p2(1) * p1(1), p2(1) * p1(2),
                     p2(2) * p1(0), p2(2) * p1(1), p2(2) * p1(2);

            // Jacobian of denominator w.r.t F
            Eigen::Matrix<double, 1, 9> dnJ_C_sq_dF;
            dnJ_C_sq_dF.setZero();
            // Contribution from (F*p1)[0]^2 and (F*p1)[1]^2
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    dnJ_C_sq_dF(i * 3 + j) += 2.0 * Fp1(i) * p1(j);
                }
            }
            // Contribution from (F^T*p2)[0]^2 and (F^T*p2)[1]^2
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 2; ++j) {
                    dnJ_C_sq_dF(i * 3 + j) += 2.0 * FTp2(j) * p2(i);
                }
            }

            // Sampson error: r = C / sqrt(nJ_C_sq)
            double inv_nJ_C_cubed = inv_nJ_C * inv_nJ_C * inv_nJ_C;
            Eigen::Matrix<double, 1, 9> dr_dF = dC_dF * inv_nJ_C - 0.5 * C * dnJ_C_sq_dF * inv_nJ_C_cubed;

            // For normalized Sampson error, focal lengths affect via point normalization
            // p1 = [px1/f1, py1/f1, 1], so dp1/df1 = [-px1/f1^2, -py1/f1^2, 0]
            // p2 = [px2/f2, py2/f2, 1], so dp2/df2 = [-px2/f2^2, -py2/f2^2, 0]
            Eigen::Vector3d dp1_df1(-px1 * inv_f1_sq, -py1 * inv_f1_sq, 0.0);
            Eigen::Vector3d dp2_df2(-px2 * inv_f2_sq, -py2 * inv_f2_sq, 0.0);

            // dC/df1 = p2^T * F * dp1_df1
            double dC_df1 = p2.dot(params.F * dp1_df1);
            // dC/df2 = dp2_df2^T * F * p1
            double dC_df2 = dp2_df2.dot(params.F * p1);

            // Derivative of denominator w.r.t. focal lengths
            // d(Fp1)/df1 = F * dp1_df1
            Eigen::Vector3d dFp1_df1 = params.F * dp1_df1;
            // d(FTp2)/df2 = F^T * dp2_df2
            Eigen::Vector3d dFTp2_df2 = params.F.transpose() * dp2_df2;

            // dnJ_C_sq/df1 = 2 * (Fp1[0]*dFp1_df1[0] + Fp1[1]*dFp1_df1[1])
            double dnJ_C_sq_df1 = 2.0 * (Fp1(0) * dFp1_df1(0) + Fp1(1) * dFp1_df1(1));
            // dnJ_C_sq/df2 = 2 * (FTp2[0]*dFTp2_df2[0] + FTp2[1]*dFTp2_df2[1])
            double dnJ_C_sq_df2 = 2.0 * (FTp2(0) * dFTp2_df2(0) + FTp2(1) * dFTp2_df2(1));

            // dr/df = dC/df / sqrt(nJ_C_sq) - 0.5 * C * dnJ_C_sq/df / nJ_C_sq^(3/2)
            double dr_df1 = dC_df1 * inv_nJ_C - 0.5 * C * dnJ_C_sq_df1 * inv_nJ_C_cubed;
            double dr_df2 = dC_df2 * inv_nJ_C - 0.5 * C * dnJ_C_sq_df2 * inv_nJ_C_cubed;

            Eigen::Matrix<double, 1, 11> J;
            J.block<1, 9>(0, 0) = dr_dF;
            J(0, 9) = dr_df1;
            J(0, 10) = dr_df2;

            // Accumulate into JtJ and Jtr
            Jtr += weight * r * J.transpose();
            for (size_t i = 0; i < 11; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }

        // Now add essential matrix constraint residual with numerical Jacobian
        // E = diag(1/f2, 1/f2, 1) * F * diag(1/f1, 1/f1, 1)
        // We want (σ1 - σ2) / avg(σ1, σ2) ≈ 0

        Eigen::Matrix3d E = computeEssential(params.F, params.f1, params.f2);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
        Eigen::Vector3d s = svd.singularValues();

        double avg_sv = (s(0) + s(1)) / 2.0;
        if (avg_sv > 1e-10) {
            double diff = (s(0) - s(1)) / avg_sv;
            double essential_weight = 100.0 * x1.size();
            double r_essential = std::sqrt(essential_weight) * diff;

            // Numerical Jacobian for essential constraint
            const double eps = 1e-7;
            Eigen::Matrix<double, 1, 11> J_essential;

            // Jacobian w.r.t. F elements
            for (int idx = 0; idx < 9; ++idx) {
                Eigen::Matrix3d F_pert = params.F;
                int row = idx / 3;
                int col = idx % 3;
                F_pert(row, col) += eps;

                Eigen::Matrix3d E_pert = computeEssential(F_pert, params.f1, params.f2);
                Eigen::JacobiSVD<Eigen::Matrix3d> svd_pert(E_pert);
                Eigen::Vector3d s_pert = svd_pert.singularValues();
                double avg_pert = (s_pert(0) + s_pert(1)) / 2.0;
                double diff_pert = (avg_pert > 1e-10) ? (s_pert(0) - s_pert(1)) / avg_pert : 0.0;

                J_essential(0, idx) = std::sqrt(essential_weight) * (diff_pert - diff) / eps;
            }

            // Jacobian w.r.t. f1
            {
                Eigen::Matrix3d E_pert = computeEssential(params.F, params.f1 + eps, params.f2);
                Eigen::JacobiSVD<Eigen::Matrix3d> svd_pert(E_pert);
                Eigen::Vector3d s_pert = svd_pert.singularValues();
                double avg_pert = (s_pert(0) + s_pert(1)) / 2.0;
                double diff_pert = (avg_pert > 1e-10) ? (s_pert(0) - s_pert(1)) / avg_pert : 0.0;
                J_essential(0, 9) = std::sqrt(essential_weight) * (diff_pert - diff) / eps;
            }

            // Jacobian w.r.t. f2
            {
                Eigen::Matrix3d E_pert = computeEssential(params.F, params.f1, params.f2 + eps);
                Eigen::JacobiSVD<Eigen::Matrix3d> svd_pert(E_pert);
                Eigen::Vector3d s_pert = svd_pert.singularValues();
                double avg_pert = (s_pert(0) + s_pert(1)) / 2.0;
                double diff_pert = (avg_pert > 1e-10) ? (s_pert(0) - s_pert(1)) / avg_pert : 0.0;
                J_essential(0, 10) = std::sqrt(essential_weight) * (diff_pert - diff) / eps;
            }

            // Accumulate essential constraint into JtJ and Jtr
            Jtr += r_essential * J_essential.transpose();
            for (size_t i = 0; i < 11; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += J_essential(0, i) * J_essential(0, j);
                }
            }

            num_residuals++;
        }

        return num_residuals;
    }

    FocalFundamentalMatrixParams step(Eigen::Matrix<double, 11, 1> dp,
                                      const FocalFundamentalMatrixParams &params) const {
        FocalFundamentalMatrixParams params_new;

        // Update F matrix (reshape dp[0:9] as 3x3 row-major)
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> dF(dp.data());
        params_new.F = params.F + dF;

        // Update focal lengths (use multiplicative update for positivity)
        // dp represents log-scale update: f_new = f * exp(dp)
        params_new.f1 = params.f1 + dp(9);
        params_new.f2 = params.f2 + dp(10);

        // Ensure focal lengths stay positive and reasonable
        if (params_new.f1 < 10.0) params_new.f1 = 10.0;
        if (params_new.f2 < 10.0) params_new.f2 = 10.0;
        if (params_new.f1 > 100000.0) params_new.f1 = 100000.0;
        if (params_new.f2 > 100000.0) params_new.f2 = 100000.0;

        // Project F to rank-2 to maintain the fundamental matrix constraint
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(params_new.F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d s = svd.singularValues();
        s(2) = 0.0;  // Enforce rank-2
        params_new.F = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();

        return params_new;
    }

    typedef FocalFundamentalMatrixParams param_t;
    static constexpr size_t num_params = 11;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Parameter structure for F+focal length parameterized through R, t, f1, f2
// This provides a more constrained parameterization than optimizing F directly
// - F = K2^T * E * K1^{-1} where E = [t]_x * R
// - K1 = diag(f1, f1, 1), K2 = diag(f2, f2, 1)
// Total: 7 DOF (3 rotation + 2 translation on unit sphere + 2 focal lengths)
struct FocalRelativePoseParams {
    FocalRelativePoseParams()
        : pose(), f1(1.0), f2(1.0) {}

    FocalRelativePoseParams(const CameraPose &kPose, double kF1, double kF2)
        : pose(kPose), f1(kF1), f2(kF2) {}

    CameraPose pose;  // Relative pose (R, t with ||t|| = 1)
    double f1;        // Focal length of camera 1
    double f2;        // Focal length of camera 2

    // Compute F from pose and focal lengths
    // F = K2^{-T} * E * K1^{-1} = K2^{-T} * [t]_x * R * K1^{-1}
    Eigen::Matrix3d F() const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        // K^{-1} = diag(1/f, 1/f, 1)
        Eigen::Matrix3d K1_inv = Eigen::Matrix3d::Identity();
        K1_inv(0, 0) = 1.0 / f1;
        K1_inv(1, 1) = 1.0 / f1;

        Eigen::Matrix3d K2_inv_T = Eigen::Matrix3d::Identity();
        K2_inv_T(0, 0) = 1.0 / f2;
        K2_inv_T(1, 1) = 1.0 / f2;

        return K2_inv_T * E * K1_inv;
    }
};

// Jacobian accumulator for F+focal using rotation+translation parameterization
// This is more constrained than FocalFundamentalMatrixJacobianAccumulator since
// it enforces that F comes from a valid essential matrix with focal lengths.
//
// Parameters: 7 DOF
// - 3 for rotation (Lie algebra)
// - 2 for translation (on tangent plane of unit sphere)
// - 2 for focal lengths (f1, f2)
//
// The Sampson error is computed in pixel space.
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class FocalRelativePoseJacobianAccumulator {
  public:
    FocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                         const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                         const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const FocalRelativePoseParams &params) const {
        Eigen::Matrix3d F = params.F();

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            if (nJc_sq < 1e-10)
                continue;

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const FocalRelativePoseParams &params, Eigen::Matrix<double, 7, 7> &JtJ,
                      Eigen::Matrix<double, 7, 1> &Jtr) {
        // Set up tangent basis for translation (orthogonal to t)
        const Eigen::Vector3d &t = params.pose.t;
        if (std::abs(t.x()) < std::abs(t.y())) {
            if (std::abs(t.x()) < std::abs(t.z())) {
                tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            if (std::abs(t.y()) < std::abs(t.z())) {
                tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(t).normalized();

        double f1 = params.f1;
        double f2 = params.f2;
        double inv_f1 = 1.0 / f1;
        double inv_f2 = 1.0 / f2;

        Eigen::Matrix3d E, R;
        R = params.pose.R();
        essential_from_motion(params.pose, &E);

        // F = K2^{-T} * E * K1^{-1}
        // F[i,j] = E[i,j] * scale_row[i] * scale_col[j]
        // scale_row = [1/f2, 1/f2, 1], scale_col = [1/f1, 1/f1, 1]
        Eigen::Matrix3d F = params.F();

        // Jacobian of E w.r.t. rotation (3 params) and translation (2 params on tangent)
        // dE/dR: Each column is vec(E*skew(e_k)) for Lie algebra basis
        Eigen::Matrix<double, 9, 3> dE_dR;
        dE_dR.block<3, 1>(0, 0).setZero();
        dE_dR.block<3, 1>(0, 1) = -E.col(2);
        dE_dR.block<3, 1>(0, 2) = E.col(1);
        dE_dR.block<3, 1>(3, 0) = E.col(2);
        dE_dR.block<3, 1>(3, 1).setZero();
        dE_dR.block<3, 1>(3, 2) = -E.col(0);
        dE_dR.block<3, 1>(6, 0) = -E.col(1);
        dE_dR.block<3, 1>(6, 1) = E.col(0);
        dE_dR.block<3, 1>(6, 2).setZero();

        // dE/dt: Each column is vec(skew(tangent_basis[k])*R)
        Eigen::Matrix<double, 9, 2> dE_dt;
        dE_dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dE_dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dE_dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dE_dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dE_dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dE_dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        // Now compute dF/dR, dF/dt, dF/df1, dF/df2
        // F[i,j] = E[i,j] * scale_row[i] * scale_col[j]
        // where scale_row = [1/f2, 1/f2, 1], scale_col = [1/f1, 1/f1, 1]

        // dF/dR = diag(scale_row) * dE/dR * diag(scale_col) in matrix form
        // For vectorized form: dF_vec[9*1]/dR[3*1]
        Eigen::Matrix<double, 9, 3> dF_dR;
        for (int col = 0; col < 3; ++col) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double scale_row = (i < 2) ? inv_f2 : 1.0;
                    double scale_col = (j < 2) ? inv_f1 : 1.0;
                    dF_dR(i * 3 + j, col) = dE_dR(i * 3 + j, col) * scale_row * scale_col;
                }
            }
        }

        Eigen::Matrix<double, 9, 2> dF_dt;
        for (int col = 0; col < 2; ++col) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double scale_row = (i < 2) ? inv_f2 : 1.0;
                    double scale_col = (j < 2) ? inv_f1 : 1.0;
                    dF_dt(i * 3 + j, col) = dE_dt(i * 3 + j, col) * scale_row * scale_col;
                }
            }
        }

        // dF/df1: F[i,j] = E[i,j] / f1 (for j<2) or E[i,j] (for j=2) scaled by row
        // d(1/f1)/df1 = -1/f1^2
        Eigen::Matrix<double, 9, 1> dF_df1;
        dF_df1.setZero();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {  // Only columns 0,1 depend on f1
                double scale_row = (i < 2) ? inv_f2 : 1.0;
                dF_df1(i * 3 + j) = E(i, j) * scale_row * (-inv_f1 * inv_f1);
            }
        }

        // dF/df2: F[i,j] = E[i,j] scaled by row (for i<2) / f2
        // d(1/f2)/df2 = -1/f2^2
        Eigen::Matrix<double, 9, 1> dF_df2;
        dF_df2.setZero();
        for (int i = 0; i < 2; ++i) {  // Only rows 0,1 depend on f2
            for (int j = 0; j < 3; ++j) {
                double scale_col = (j < 2) ? inv_f1 : 1.0;
                dF_df2(i * 3 + j) = E(i, j) * (-inv_f2 * inv_f2) * scale_col;
            }
        }

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(),
                   F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();

            if (nJ_C < 1e-10)
                continue;

            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t F (vectorized as 9x1)
            // d(C/||J_C||)/dF = dC/dF / ||J_C|| - C * d||J_C||/dF / ||J_C||^2
            Eigen::Matrix<double, 1, 9> dC_dF;
            dC_dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0),
                     x1[k](1) * x2[k](0), x1[k](1) * x2[k](1), x1[k](1),
                     x2[k](0), x2[k](1), 1.0;

            const double s = C * inv_nJ_C * inv_nJ_C;
            Eigen::Matrix<double, 1, 9> dF_sampson = dC_dF;
            dF_sampson(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF_sampson(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF_sampson(2) -= s * (J_C(0));
            dF_sampson(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF_sampson(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF_sampson(5) -= s * (J_C(1));
            dF_sampson(6) -= s * (J_C(2));
            dF_sampson(7) -= s * (J_C(3));
            dF_sampson *= inv_nJ_C;

            // Chain rule: dr/d(params) = dr/dF * dF/d(params)
            Eigen::Matrix<double, 1, 7> J;
            J.block<1, 3>(0, 0) = dF_sampson * dF_dR;
            J.block<1, 2>(0, 3) = dF_sampson * dF_dt;
            J(0, 5) = dF_sampson * dF_df1;
            J(0, 6) = dF_sampson * dF_df2;

            // Accumulate into JtJ and Jtr
            Jtr += weight * r * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    FocalRelativePoseParams step(Eigen::Matrix<double, 7, 1> dp, const FocalRelativePoseParams &params) const {
        FocalRelativePoseParams params_new;

        // Update rotation using Lie algebra (post-multiplication)
        params_new.pose.q = quat_step_post(params.pose.q, dp.block<3, 1>(0, 0));

        // Update translation on the tangent plane of unit sphere
        params_new.pose.t = params.pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        // Re-normalize to stay on unit sphere
        params_new.pose.t.normalize();

        // Update focal lengths (additive)
        params_new.f1 = params.f1 + dp(5);
        params_new.f2 = params.f2 + dp(6);

        // Clamp focal lengths to reasonable range
        params_new.f1 = std::max(10.0, std::min(100000.0, params_new.f1));
        params_new.f2 = std::max(10.0, std::min(100000.0, params_new.f2));

        return params_new;
    }

    typedef FocalRelativePoseParams param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    mutable Eigen::Matrix<double, 3, 2> tangent_basis;
};

} // namespace poselib