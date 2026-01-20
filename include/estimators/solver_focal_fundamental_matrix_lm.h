// Copyright (C) 2024 ETH Zurich.
// All rights reserved.

#pragma once

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/jacobian_impl.h"

#include <Eigen/SVD>
#include <cmath>

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// Nonminimal LM refinement solver for Focal Fundamental Matrix
			// Refines F and focal lengths f1, f2 using all provided points
			class FocalFundamentalMatrixLMSolver : public AbstractSolver
			{
			public:
				FocalFundamentalMatrixLMSolver()
				{
				}

				~FocalFundamentalMatrixLMSolver()
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
					// Nonminimal - needs at least 6 points but can use more
					return 6;
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
						// Need at least 6 points
						if (kSampleNumber_ < 6)
							return false;

						// If we only have 6 points, we can't refine much
						// Just return false and let minimal solver handle it
						if (kSampleNumber_ == 6)
							return false;

						// We need an initial model to refine
						if (models_.empty())
							return false;

						// Extract initial model
						Eigen::MatrixXd descriptor = models_[0].getData();
						Eigen::Matrix3d F_init = descriptor.block<3, 3>(0, 0);
						double f1_init = descriptor(0, 3);
						double f2_init = descriptor(1, 3);

						// Prepare points for refinement
						std::vector<poselib::Point2D> points1, points2;
						points1.reserve(kSampleNumber_);
						points2.reserve(kSampleNumber_);

						for (size_t i = 0; i < kSampleNumber_; ++i)
						{
							size_t idx = kSample_[i];
							if (idx >= static_cast<size_t>(kData_.rows()))
								return false;

							points1.emplace_back(kData_(idx, 0), kData_(idx, 1));
							points2.emplace_back(kData_(idx, 2), kData_(idx, 3));
						}

						// Set up LM refinement
						poselib::FocalFundamentalMatrixParams params(F_init, f1_init, f2_init);

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

						// Run refinement
						poselib::BundleStats stats = poselib::refine_focal_fundamental(
							points1, points2, &params, opts, weights_vec);

						// Create refined model descriptor
						Eigen::MatrixXd refined_descriptor(3, 4);
						refined_descriptor.block<3, 3>(0, 0) = params.F;
						refined_descriptor(0, 3) = params.f1;
						refined_descriptor(1, 3) = params.f2;
						refined_descriptor(2, 3) = 0.0;

						if (!refined_descriptor.allFinite())
							return false;

						// Validate focal lengths
						if (params.f1 < 1.0 || params.f2 < 1.0 ||
							params.f1 > 100000.0 || params.f2 > 100000.0)
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
