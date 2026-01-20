// Copyright (C) 2024 ETH Zurich.
// All rights reserved.

#pragma once

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/jacobian_impl.h"
#include "numerical_optimizer/essential.h"

#include <Eigen/SVD>
#include <cmath>

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// Nonminimal LM refinement solver for Focal Fundamental Matrix
			// using rotation + translation + focal length parameterization.
			//
			// This solver refines F and focal lengths f1, f2 by parameterizing
			// F = K2^{-T} * E * K1^{-1} where E = [t]_x * R
			//
			// This is more constrained than directly optimizing F elements because
			// it ensures F always corresponds to a valid essential matrix with
			// the given focal lengths.
			//
			// Parameters: 7 DOF
			// - 3 for rotation (Lie algebra)
			// - 2 for translation (on tangent plane of unit sphere)
			// - 2 for focal lengths (f1, f2)
			class FocalFundamentalMatrixRTLMSolver : public AbstractSolver
			{
			public:
				FocalFundamentalMatrixRTLMSolver()
				{
				}

				~FocalFundamentalMatrixRTLMSolver()
				{
				}

				bool returnMultipleModels() const override
				{
					return false;
				}

				size_t maximumSolutions() const override
				{
					return 1;
				}

				size_t sampleSize() const override
				{
					// Nonminimal - needs at least 7 points but can use more
					return 7;
				}

				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_,
					const size_t *kSample_,
					const size_t kSampleNumber_,
					std::vector<models::Model> &models_,
					const double *kWeights_ = nullptr) const override
				{
					try
					{
						// Need at least 7 points
						if (kSampleNumber_ < 7)
							return false;

						// If we only have 7 points, use minimal solver only
						if (kSampleNumber_ == 7)
							return false;

						// We need an initial model to refine
						if (models_.empty())
							return false;

						// Extract initial model
						Eigen::MatrixXd descriptor = models_[0].getData();
						Eigen::Matrix3d F_init = descriptor.block<3, 3>(0, 0);
						double f1_init = descriptor(0, 3);
						double f2_init = descriptor(1, 3);

						// Decompose F into R, t using the initial focal lengths
						// E = K2^T * F * K1 (note: K^T for left multiply)
						Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
						K1(0, 0) = f1_init;
						K1(1, 1) = f1_init;
						Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
						K2(0, 0) = f2_init;
						K2(1, 1) = f2_init;

						Eigen::Matrix3d E_init = K2.transpose() * F_init * K1;

						// Normalize E via SVD to ensure valid essential matrix
						Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
						Eigen::Matrix3d U = svd.matrixU();
						Eigen::Matrix3d V = svd.matrixV();
						Eigen::Vector3d s = svd.singularValues();

						// Project to valid essential matrix (σ1 = σ2, σ3 = 0)
						double avg_sv = (s(0) + s(1)) / 2.0;
						Eigen::Matrix3d E_valid = U * Eigen::Vector3d(avg_sv, avg_sv, 0.0).asDiagonal() * V.transpose();

						// Prepare normalized points for pose recovery
						std::vector<Eigen::Vector3d> x1_normalized, x2_normalized;
						x1_normalized.reserve(kSampleNumber_);
						x2_normalized.reserve(kSampleNumber_);

						for (size_t i = 0; i < kSampleNumber_; ++i)
						{
							size_t idx = kSample_[i];
							if (idx >= static_cast<size_t>(kData_.rows()))
								return false;

							x1_normalized.emplace_back(kData_(idx, 0) / f1_init, kData_(idx, 1) / f1_init, 1.0);
							x2_normalized.emplace_back(kData_(idx, 2) / f2_init, kData_(idx, 3) / f2_init, 1.0);
						}

						// Decompose E into R, t using motion_from_essential
						poselib::CameraPoseVector relative_poses;
						poselib::motion_from_essential(E_valid, x1_normalized, x2_normalized, &relative_poses);

						if (relative_poses.empty())
							return false;

						// Use the first valid pose
						poselib::CameraPose best_pose = relative_poses[0];

						// Prepare points for LM refinement
						std::vector<poselib::Point2D> points1, points2;
						points1.reserve(kSampleNumber_);
						points2.reserve(kSampleNumber_);

						for (size_t i = 0; i < kSampleNumber_; ++i)
						{
							size_t idx = kSample_[i];
							points1.emplace_back(kData_(idx, 0), kData_(idx, 1));
							points2.emplace_back(kData_(idx, 2), kData_(idx, 3));
						}

						// Set up LM refinement with R+t+focal parameterization
						poselib::FocalRelativePoseParams params(best_pose, f1_init, f2_init);

						poselib::BundleOptions opts;
						opts.loss_type = poselib::BundleOptions::LossType::CAUCHY;
						opts.loss_scale = 1.0;
						opts.max_iterations = 50;
						opts.gradient_tol = 1e-10;
						opts.step_tol = 1e-8;

						// Run refinement
						std::vector<double> weights_vec;
						if (kWeights_ != nullptr)
						{
							weights_vec.assign(kWeights_, kWeights_ + kSampleNumber_);
						}

						poselib::BundleStats stats = poselib::refine_focal_relpose(
							points1, points2, &params, opts, weights_vec);

						// Extract refined F from params
						Eigen::Matrix3d F_refined = params.F();

						// Validate focal lengths
						if (params.f1 < 10.0 || params.f2 < 10.0 ||
							params.f1 > 100000.0 || params.f2 > 100000.0)
							return false;

						// Create refined model descriptor
						Eigen::MatrixXd refined_descriptor(3, 4);
						refined_descriptor.block<3, 3>(0, 0) = F_refined;
						refined_descriptor(0, 3) = params.f1;
						refined_descriptor(1, 3) = params.f2;
						refined_descriptor(2, 3) = 0.0;

						if (!refined_descriptor.allFinite())
							return false;

						// Replace the model with refined version
						models::Model refined_model;
						refined_model.setData(refined_descriptor);
						models_[0] = refined_model;

						return true;
					}
					catch (const std::exception& e)
					{
						return false;
					}
					catch (...)
					{
						return false;
					}
				}
			};
		}
	}
}
