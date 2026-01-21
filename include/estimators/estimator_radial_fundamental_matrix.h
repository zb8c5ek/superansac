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
//     * Neither the name of Czech Technical University nor the
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
#include "solver_radial_fundamental_matrix_nine_point.h"
#include "solver_radial_fundamental_matrix_lm.h"

namespace superansac
{
	namespace estimator
	{
		// Radial Fundamental Matrix Estimator
		// Estimates F (fundamental matrix) plus lambda1, lambda2 (division distortion parameters)
		// Uses 9-point minimal solver
		class RadialFundamentalMatrixEstimator : public Estimator
		{
		public:
			RadialFundamentalMatrixEstimator()
			{
				// Initialize the minimal solver (9-point)
				this->minimalSolver.reset(new solver::RadialFundamentalMatrixNinePointSolver());
				// Initialize the nonminimal LM refinement solver
				this->nonMinimalSolver.reset(new solver::RadialFundamentalMatrixLMSolver());
			}

			~RadialFundamentalMatrixEstimator() {}

			bool isWeightingApplicable() const override
			{
				return false; // LM solver doesn't support weights currently
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
				// DoF for MAGSAC++: we're estimating 10 parameters (8 for F + 2 for lambda)
				// but scoring is based on epipolar residual in 2D image space
				return 2;
			}

			FORCE_INLINE bool estimateModel(
				const DataMatrix& kData_,
				const size_t *kSample_,
				std::vector<models::Model>* models_) const override
			{
				static const size_t kSampleSize = sampleSize();
				
				minimalSolver->estimateModel(kData_,
					kSample_,
					kSampleSize,
					*models_);

				return models_->size() > 0;
			}

			FORCE_INLINE bool estimateModelNonminimal(
				const DataMatrix& kData_,
				const size_t *kSample_,
				const size_t &kSampleNumber_,
				std::vector<models::Model>* models_,
				const double *kWeights_ = nullptr) const override
			{
				// Need at least 9 points and an initial model
				if (kSampleNumber_ < nonMinimalSampleSize() || models_->empty())
					return false;

				// Use the LM refinement solver
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
				// Extract F and lambda values from descriptor
				// descriptor is 3x4: [F (3x3) | lam1, lam2, 0]
				Eigen::Matrix3d F = kDescriptor_.block<3, 3>(0, 0);
				double lam1 = kDescriptor_(0, 3);
				double lam2 = kDescriptor_(1, 3);

				// Compute Sampson-like error with distortion model
				double err = sampsonLikeError(kPoint_, F, lam1, lam2);
				return err;
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

			// Optimized batch residual computation
			FORCE_INLINE void squaredResidualBatch(
				const DataMatrix& kData_,
				const models::Model& kModel_,
				double* __restrict residuals_,
				const size_t kCount_) const override
			{
				const Eigen::MatrixXd& D = kModel_.getData();

				// Extract F and lambda values once
				const double F00 = D(0, 0), F01 = D(0, 1), F02 = D(0, 2);
				const double F10 = D(1, 0), F11 = D(1, 1), F12 = D(1, 2);
				const double F20 = D(2, 0), F21 = D(2, 1), F22 = D(2, 2);
				const double lam1 = D(0, 3);
				const double lam2 = D(1, 3);
				constexpr double eps = 1e-12;

				const double* __restrict dataPtr = kData_.data();
				const Eigen::Index cols = kData_.cols();

				#ifdef __GNUC__
				#pragma GCC ivdep
				#endif
				for (size_t i = 0; i < kCount_; ++i)
				{
					const double* __restrict row = dataPtr + i * cols;
					const double x1_u = row[0], x1_v = row[1];
					const double x2_u = row[2], x2_v = row[3];

					// Distorted homogeneous coordinates
					const double r1_sq = x1_u * x1_u + x1_v * x1_v;
					const double r2_sq = x2_u * x2_u + x2_v * x2_v;
					const double x1_w = 1.0 + lam1 * r1_sq;
					const double x2_w = 1.0 + lam2 * r2_sq;

					// F * x1
					const double Fx1_0 = F00 * x1_u + F01 * x1_v + F02 * x1_w;
					const double Fx1_1 = F10 * x1_u + F11 * x1_v + F12 * x1_w;
					const double Fx1_2 = F20 * x1_u + F21 * x1_v + F22 * x1_w;

					// F^T * x2
					const double Ftx2_0 = F00 * x2_u + F10 * x2_v + F20 * x2_w;
					const double Ftx2_1 = F01 * x2_u + F11 * x2_v + F21 * x2_w;
					const double Ftx2_2 = F02 * x2_u + F12 * x2_v + F22 * x2_w;

					// x2^T * F * x1
					const double num = x2_u * Fx1_0 + x2_v * Fx1_1 + x2_w * Fx1_2;
					const double num_sq = num * num;

					const double den = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 +
									   Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1 + eps;

					const double result = num_sq / den;
					residuals_[i] = std::isfinite(result) ? result : 1e10;
				}
			}

			// Batch residual with indices
			FORCE_INLINE void squaredResidualBatch(
				const DataMatrix& kData_,
				const models::Model& kModel_,
				const size_t* __restrict kIndices_,
				double* __restrict residuals_,
				const size_t kCount_) const override
			{
				const Eigen::MatrixXd& D = kModel_.getData();

				const double F00 = D(0, 0), F01 = D(0, 1), F02 = D(0, 2);
				const double F10 = D(1, 0), F11 = D(1, 1), F12 = D(1, 2);
				const double F20 = D(2, 0), F21 = D(2, 1), F22 = D(2, 2);
				const double lam1 = D(0, 3);
				const double lam2 = D(1, 3);
				constexpr double eps = 1e-12;

				const double* __restrict dataPtr = kData_.data();
				const Eigen::Index cols = kData_.cols();

				#ifdef __GNUC__
				#pragma GCC ivdep
				#endif
				for (size_t i = 0; i < kCount_; ++i)
				{
					const double* __restrict row = dataPtr + kIndices_[i] * cols;
					const double x1_u = row[0], x1_v = row[1];
					const double x2_u = row[2], x2_v = row[3];

					const double r1_sq = x1_u * x1_u + x1_v * x1_v;
					const double r2_sq = x2_u * x2_u + x2_v * x2_v;
					const double x1_w = 1.0 + lam1 * r1_sq;
					const double x2_w = 1.0 + lam2 * r2_sq;

					const double Fx1_0 = F00 * x1_u + F01 * x1_v + F02 * x1_w;
					const double Fx1_1 = F10 * x1_u + F11 * x1_v + F12 * x1_w;
					const double Fx1_2 = F20 * x1_u + F21 * x1_v + F22 * x1_w;

					const double Ftx2_0 = F00 * x2_u + F10 * x2_v + F20 * x2_w;
					const double Ftx2_1 = F01 * x2_u + F11 * x2_v + F21 * x2_w;
					const double Ftx2_2 = F02 * x2_u + F12 * x2_v + F22 * x2_w;

					const double num = x2_u * Fx1_0 + x2_v * Fx1_1 + x2_w * Fx1_2;
					const double num_sq = num * num;

					const double den = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 +
									   Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1 + eps;

					const double result = num_sq / den;
					residuals_[i] = std::isfinite(result) ? result : 1e10;
				}
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
			// Helper functions for residual calculation

			// Compute homogeneous distorted point using division model
			FORCE_INLINE static Eigen::Vector3d undistHDivision(
				const Eigen::Vector2d& p_centered,
				double lam)
			{
				double u = p_centered(0);
				double v = p_centered(1);
				double r2 = u * u + v * v;
				return Eigen::Vector3d(u, v, 1.0 + lam * r2);
			}

			// Compute Sampson-like error with division distortion
			FORCE_INLINE static double sampsonLikeError(
				const DataMatrix& kPoint_,
				const Eigen::Matrix3d& F,
				double lam1,
				double lam2,
				double eps = 1e-12)
			{
				// Extract point coordinates
				double x1_u = kPoint_(0);
				double x1_v = kPoint_(1);
				double x2_u = kPoint_(2);
				double x2_v = kPoint_(3);

				// Create distorted homogeneous points
				Eigen::Vector2d p1_centered(x1_u, x1_v);
				Eigen::Vector2d p2_centered(x2_u, x2_v);
				Eigen::Vector3d x1 = undistHDivision(p1_centered, lam1);
				Eigen::Vector3d x2 = undistHDivision(p2_centered, lam2);

				// Compute residual
				Eigen::Vector3d Fx1 = F * x1;
				Eigen::Vector3d Ftx2 = F.transpose() * x2;

				double num = x2.dot(Fx1);
				num = num * num;

				double den = Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) +
							 Ftx2(0) * Ftx2(0) + Ftx2(1) * Ftx2(1) + eps;

				double result = num / den;
				if (!std::isfinite(result))
				{
					return 1e10; // Return large error instead of NaN/Inf
				}
				return result;
			}

		public:
			// We rely on the base class sampleSize() which calls minimalSolver->sampleSize()
			// The minimalSolver member is initialized in the constructor using this->minimalSolver

		};
	}
}
