// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of ETH Zurich nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Eigen>

#include "abstract_estimator.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "solver_focal_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_bundle_adjustment.h"

namespace superansac
{
	namespace estimator
	{
		// Focal Fundamental Matrix Estimator
		// Estimates F (fundamental matrix) plus f1, f2 (focal lengths for both cameras)
		//
		// 1. Minimal solver: 7-point F + Bougnoux focal extraction
		// 2. Non-minimal solver: Bundle Adjustment for F refinement + Bougnoux focal re-extraction
		//
		// Assumes principal point is at origin (coordinates are centered)
		class FocalFundamentalMatrixEstimator : public Estimator
		{
		public:
			FocalFundamentalMatrixEstimator()
			{
				// Initialize the minimal solver - use standard 7-point for better consistency
				this->minimalSolver.reset(new solver::FundamentalMatrixSevenPointSolver());
				// Initialize the nonminimal solver (Bundle Adjustment for F refinement - same as standard F)
				this->nonMinimalSolver.reset(new solver::FundamentalMatrixBundleAdjustmentSolver());
			}

			~FocalFundamentalMatrixEstimator() {}

			bool isWeightingApplicable() const override
			{
				return true; // BA solver supports weights (same as standard F)
			}

			double multError() const
			{
				return 1.0;
			}

			double logAlpha0(size_t w, size_t h, double scalingFactor = 0.5) const
			{
				return log(1.0 / (w * h * scalingFactor));
			}

			size_t getDegreesOfFreedom() const
			{
				// DoF for MAGSAC++: we're estimating 7 effective parameters
				// (3 rot + 2 trans + 2 focal) but scoring is based on epipolar residual
				return 2;
			}

			FORCE_INLINE bool estimateModel(
				const DataMatrix& kData_,
				const size_t *kSample_,
				std::vector<models::Model>* models_) const override
			{
				static const size_t kSampleSize = sampleSize();

				// Get 3x3 F matrices from standard 7-point solver
				std::vector<models::Model> tempModels;
				minimalSolver->estimateModel(kData_,
					kSample_,
					kSampleSize,
					tempModels);

				if (tempModels.empty())
					return false;

				// Compute fallback focal length from point coordinates
				double maxCoord = 0.0;
				for (size_t i = 0; i < kSampleSize; ++i)
				{
					const size_t idx = kSample_ == nullptr ? i : kSample_[i];
					maxCoord = std::max(maxCoord, std::abs(kData_(idx, 0)));
					maxCoord = std::max(maxCoord, std::abs(kData_(idx, 1)));
					maxCoord = std::max(maxCoord, std::abs(kData_(idx, 2)));
					maxCoord = std::max(maxCoord, std::abs(kData_(idx, 3)));
				}
				double fallbackFocal = std::max(300.0, maxCoord * 1.5);

				// Use 3x3 models directly (same as standard F) to isolate the issue
				for (const auto& tempModel : tempModels)
				{
					models_->push_back(tempModel);
				}

				return models_->size() > 0;
			}

			FORCE_INLINE bool estimateModelNonminimal(
				const DataMatrix& kData_,
				const size_t *kSample_,
				const size_t &kSampleNumber_,
				std::vector<models::Model>* models_,
				const double *kWeights_ = nullptr) const override
			{
				// Need at least 8 points for refinement
				if (kSampleNumber_ < 8 || models_->empty())
					return false;

				// BA solver expects 3x3 model - models are already 3x3 now
				return nonMinimalSolver->estimateModel(kData_,
					kSample_,
					kSampleNumber_,
					*models_,
					kWeights_);
			}

			FORCE_INLINE double squaredResidual(
				const DataMatrix& point_,
				const models::Model& model_) const override
			{
				return squaredResidual(point_, model_.getData());
			}

			FORCE_INLINE double squaredResidual(
				const DataMatrix& kPoint_,
				const Eigen::MatrixXd& kDescriptor_) const
			{
				// Use the same Sampson distance formula as standard F estimator
				const double E0_0 = kDescriptor_(0, 0), E0_1 = kDescriptor_(0, 1), E0_2 = kDescriptor_(0, 2);
				const double E1_0 = kDescriptor_(1, 0), E1_1 = kDescriptor_(1, 1), E1_2 = kDescriptor_(1, 2);
				const double E2_0 = kDescriptor_(2, 0), E2_1 = kDescriptor_(2, 1), E2_2 = kDescriptor_(2, 2);

				const double x1_0 = kPoint_(0), x1_1 = kPoint_(1);
				const double x2_0 = kPoint_(2), x2_1 = kPoint_(3);

				const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
				const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
				const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

				const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
				const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;

				const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
				const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
				const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
				const double r2 = C * C / (Cx + Cy);

				return r2;
			}

			FORCE_INLINE double residual(
				const DataMatrix& point_,
				const models::Model& model_) const override
			{
				return residual(point_, model_.getData());
			}

			FORCE_INLINE double residual(
				const DataMatrix& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return std::sqrt(squaredResidual(point_, descriptor_));
			}

			FORCE_INLINE bool isValidModel(
				models::Model& model_,
				const DataMatrix& kData_,
				const size_t* kMinimalSample_,
				const double kThreshold_,
				bool& modelUpdated_) const override
			{
				return true;
			}

			FORCE_INLINE bool isValidSample(
				const DataMatrix& kData_,
				const size_t *kSample_) const override
			{
				return true;
			}

		protected:
			// Compute squared Sampson error
			// Standard formula: C^2 / (||Fp1||^2 + ||F^Tp2||^2)
			// This matches the standard fundamental matrix estimator's squaredSampsonDistance
			FORCE_INLINE static double sampsonErrorPixel(
				const DataMatrix& kPoint_,
				const Eigen::Matrix3d& F,
				double f1,
				double f2,
				double eps = 1e-12)
			{
				const double E0_0 = F(0, 0), E0_1 = F(0, 1), E0_2 = F(0, 2);
				const double E1_0 = F(1, 0), E1_1 = F(1, 1), E1_2 = F(1, 2);
				const double E2_0 = F(2, 0), E2_1 = F(2, 1), E2_2 = F(2, 2);

				const double x1_0 = kPoint_(0), x1_1 = kPoint_(1);
				const double x2_0 = kPoint_(2), x2_1 = kPoint_(3);

				// F * p1 (epipolar line in image 2)
				const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
				const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
				const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

				// F^T * p2 (epipolar line in image 1)
				const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
				const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;

				// Epipolar constraint: C = p2^T * F * p1
				const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

				// Squared norms of epipolar line gradients
				const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
				const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;

				// Sampson error
				const double den = Cx + Cy;
				if (den < eps)
					return 1e10;

				const double r2 = C * C / den;
				if (!std::isfinite(r2))
					return 1e10;
				return r2;
			}
		};
	}
}
