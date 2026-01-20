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

#include <unsupported/Eigen/Polynomials>

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// Focal Fundamental Matrix Solver using 7-point algorithm + Bougnoux focal extraction
			//
			// This solver:
			// 1. Uses the standard 7-point algorithm to estimate F (up to 3 solutions)
			// 2. Extracts focal lengths from F using the Bougnoux formula
			//    (based on PoseLib's robust implementation)
			// 3. Assumes principal point is at origin (coordinates are centered)
			//
			// The Bougnoux formula uses epipoles and skew-symmetric matrices to
			// compute focal lengths from the fundamental matrix.
			class FocalFundamentalMatrixSevenPointSolver : public AbstractSolver
			{
			public:
				FocalFundamentalMatrixSevenPointSolver()
				{
				}

				~FocalFundamentalMatrixSevenPointSolver()
				{
				}

				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				size_t maximumSolutions() const override
				{
					return 3; // 7-point can return up to 3 F matrices
				}

				size_t sampleSize() const override
				{
					return 7;
				}

				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_,
					const size_t *kSample_,
					const size_t kSampleNumber_,
					std::vector<models::Model> &models_,
					const double *kWeights_ = nullptr) const override;

			public:
				// Robust focal length extraction using the Bougnoux formula
				// (from PoseLib - more numerically stable than Kanatani-Matsunaga)
				// Principal point is assumed at origin.
				// Returns true if valid focal lengths were extracted.
				// If extraction fails but allowFallback is true, uses fallbackFocal.
				FORCE_INLINE static bool extractFocalLengths(
					const Eigen::Matrix3d& F,
					double& f1, double& f2,
					double fallbackFocal = 500.0,
					bool allowFallback = true)
				{
					// Principal point at origin
					Eigen::Vector3d p1(0.0, 0.0, 1.0);
					Eigen::Vector3d p2(0.0, 0.0, 1.0);

					// SVD to get epipoles
					Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
					Eigen::Vector3d e1 = svd.matrixV().col(2);  // Right null vector
					Eigen::Vector3d e2 = svd.matrixU().col(2);  // Left null vector

					// II = diag(1,1,0) as regular matrix for proper multiplication
					Eigen::Matrix3d II = Eigen::Matrix3d::Zero();
					II(0, 0) = 1.0;
					II(1, 1) = 1.0;

					// Skew-symmetric matrices for epipoles
					Eigen::Matrix3d s_e1, s_e2;
					s_e1 << 0, -e1(2), e1(1),
					        e1(2), 0, -e1(0),
					        -e1(1), e1(0), 0;
					s_e2 << 0, -e2(2), e2(1),
					        e2(2), 0, -e2(0),
					        -e2(1), e2(0), 0;

					// Bougnoux formula for f1^2
					// f1^2 = (-p2^T * s_e2 * II * F * (p1*p1^T) * F^T * p2) /
					//        (p2^T * s_e2 * II * F * II * F^T * p2)
					Eigen::Matrix3d FT = F.transpose();
					Eigen::Matrix3d p1p1T = p1 * p1.transpose();
					Eigen::Matrix3d p2p2T = p2 * p2.transpose();

					// Compute step by step for clarity
					Eigen::Matrix3d s_e2_II = s_e2 * II;
					Eigen::Matrix3d s_e1_II = s_e1 * II;

					double num1 = -(p2.transpose() * s_e2_II * F * p1p1T * FT * p2)(0);
					double den1 = (p2.transpose() * s_e2_II * F * II * FT * p2)(0);

					// Bougnoux formula for f2^2
					// f2^2 = (-p1^T * s_e1 * II * F^T * (p2*p2^T) * F * p1) /
					//        (p1^T * s_e1 * II * F^T * II * F * p1)
					double num2 = -(p1.transpose() * s_e1_II * FT * p2p2T * F * p1)(0);
					double den2 = (p1.transpose() * s_e1_II * FT * II * F * p1)(0);

					bool f1_valid = false, f2_valid = false;
					const double eps = 1e-10;  // Slightly relaxed tolerance
					const double min_focal = 50.0;
					const double max_focal = 50000.0;

					// Compute f1
					if (std::abs(den1) > eps)
					{
						double f1_sq = num1 / den1;
						if (std::isfinite(f1_sq) && f1_sq > 0.0)
						{
							f1 = std::sqrt(f1_sq);
							if (f1 >= min_focal && f1 <= max_focal)
								f1_valid = true;
						}
					}

					// Compute f2
					if (std::abs(den2) > eps)
					{
						double f2_sq = num2 / den2;
						if (std::isfinite(f2_sq) && f2_sq > 0.0)
						{
							f2 = std::sqrt(f2_sq);
							if (f2 >= min_focal && f2 <= max_focal)
								f2_valid = true;
						}
					}

					// If both are valid, return success
					if (f1_valid && f2_valid)
						return true;

					// If fallback is allowed, use fallback for invalid focal lengths
					if (allowFallback)
					{
						if (!f1_valid)
							f1 = fallbackFocal;
						if (!f2_valid)
							f2 = fallbackFocal;
						return true;
					}

					return false;
				}

				// Solve cubic equation: x^3 + c2*x^2 + c1*x + c0 = 0
				FORCE_INLINE int solveCubicReal(double c2, double c1, double c0, double roots[3]) const
				{
					double a = c1 - c2 * c2 / 3.0;
					double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
					double c = b * b / 4.0 + a * a * a / 27.0;
					int n_roots;
					if (c > 0) {
						c = std::sqrt(c);
						b *= -0.5;
						roots[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
						n_roots = 1;
					} else {
						c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
						double d = 2.0 * std::sqrt(-a / 3.0);
						roots[0] = d * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
						roots[1] = d * std::cos(std::acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0;
						roots[2] = d * std::cos(std::acos(c) / 3.0 - 4.18879020478639052527114472468384) - c2 / 3.0;
						n_roots = 3;
					}

					// Newton iteration for refinement
					for (int i = 0; i < n_roots; ++i) {
						double x = roots[i];
						double x2 = x * x;
						double x3 = x * x2;
						double dx = -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
						roots[i] += dx;
					}
					return n_roots;
				}
			};

			FORCE_INLINE bool FocalFundamentalMatrixSevenPointSolver::estimateModel(
				const DataMatrix& kData_,
				const size_t *kSample_,
				const size_t kSampleNumber_,
				std::vector<models::Model> &models_,
				const double *kWeights_) const
			{
				try
				{
					if (kSampleNumber_ < 7)
						return false;

					// Compute a reasonable fallback focal length from point coordinates
					// Use max absolute coordinate as a proxy for image size, then estimate focal
					double maxCoord = 0.0;
					for (size_t i = 0; i < 7; ++i)
					{
						const size_t idx = kSample_ == nullptr ? i : kSample_[i];
						maxCoord = std::max(maxCoord, std::abs(kData_(idx, 0)));
						maxCoord = std::max(maxCoord, std::abs(kData_(idx, 1)));
						maxCoord = std::max(maxCoord, std::abs(kData_(idx, 2)));
						maxCoord = std::max(maxCoord, std::abs(kData_(idx, 3)));
					}
					// Typical focal length is roughly 1-2x image dimension
					double fallbackFocal = std::max(300.0, maxCoord * 1.5);

					// Build the coefficient matrix for the epipolar constraint
					Eigen::Matrix<double, 7, 9> coefficients;

					for (size_t i = 0; i < 7; ++i)
					{
						const size_t idx = kSample_ == nullptr ? i : kSample_[i];

						const double x0 = kData_(idx, 0);
						const double y0 = kData_(idx, 1);
						const double x1 = kData_(idx, 2);
						const double y1 = kData_(idx, 3);

						double weight = 1.0;
						if (kWeights_ != nullptr)
							weight = kWeights_[idx];

						coefficients(i, 0) = weight * x1 * x0;
						coefficients(i, 1) = weight * x1 * y0;
						coefficients(i, 2) = weight * x1;
						coefficients(i, 3) = weight * y1 * x0;
						coefficients(i, 4) = weight * y1 * y0;
						coefficients(i, 5) = weight * y1;
						coefficients(i, 6) = weight * x0;
						coefficients(i, 7) = weight * y0;
						coefficients(i, 8) = weight;
					}

					// Compute null space (2D)
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
					if (lu.dimensionOfKernel() != 2)
						return false;

					const Eigen::Matrix<double, 9, 2> N = lu.kernel();

					// Compute coefficients for det(F(x)) = 0 where F = x*N[:,0] + N[:,1]
					const double c3 = N(0, 0) * N(4, 0) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 0) - N(1, 0) * N(3, 0) * N(8, 0) +
									  N(1, 0) * N(5, 0) * N(6, 0) + N(2, 0) * N(3, 0) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 0);
					const double c2 = N(0, 0) * N(4, 0) * N(8, 1) + N(0, 0) * N(4, 1) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 1) -
									  N(0, 0) * N(5, 1) * N(7, 0) + N(0, 1) * N(4, 0) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 0) -
									  N(1, 0) * N(3, 0) * N(8, 1) - N(1, 0) * N(3, 1) * N(8, 0) + N(1, 0) * N(5, 0) * N(6, 1) +
									  N(1, 0) * N(5, 1) * N(6, 0) - N(1, 1) * N(3, 0) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 0) +
									  N(2, 0) * N(3, 0) * N(7, 1) + N(2, 0) * N(3, 1) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 1) -
									  N(2, 0) * N(4, 1) * N(6, 0) + N(2, 1) * N(3, 0) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 0);
					const double c1 = N(0, 0) * N(4, 1) * N(8, 1) - N(0, 0) * N(5, 1) * N(7, 1) + N(0, 1) * N(4, 0) * N(8, 1) +
									  N(0, 1) * N(4, 1) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 1) - N(0, 1) * N(5, 1) * N(7, 0) -
									  N(1, 0) * N(3, 1) * N(8, 1) + N(1, 0) * N(5, 1) * N(6, 1) - N(1, 1) * N(3, 0) * N(8, 1) -
									  N(1, 1) * N(3, 1) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 1) + N(1, 1) * N(5, 1) * N(6, 0) +
									  N(2, 0) * N(3, 1) * N(7, 1) - N(2, 0) * N(4, 1) * N(6, 1) + N(2, 1) * N(3, 0) * N(7, 1) +
									  N(2, 1) * N(3, 1) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 1) - N(2, 1) * N(4, 1) * N(6, 0);
					const double c0 = N(0, 1) * N(4, 1) * N(8, 1) - N(0, 1) * N(5, 1) * N(7, 1) - N(1, 1) * N(3, 1) * N(8, 1) +
									  N(1, 1) * N(5, 1) * N(6, 1) + N(2, 1) * N(3, 1) * N(7, 1) - N(2, 1) * N(4, 1) * N(6, 1);

					if (std::abs(c3) < 1e-10)
						return false;

					// Solve the cubic
					double inv_c3 = 1.0 / c3;
					double roots[3];
					int n_roots = solveCubicReal(c2 * inv_c3, c1 * inv_c3, c0 * inv_c3, roots);

					for (int i = 0; i < n_roots; ++i)
					{
						Eigen::Matrix<double, 9, 1> f = N.col(0) * roots[i] + N.col(1);
						f.normalize();

						Eigen::Matrix3d F;
						F << f[0], f[1], f[2],
							 f[3], f[4], f[5],
							 f[6], f[7], f[8];

						// Skip if F is not finite
						if (!F.allFinite())
							continue;

						// Extract focal lengths using Bougnoux formula
						// Always use fallback for invalid focals to avoid rejecting valid F matrices
						double f1 = fallbackFocal, f2 = fallbackFocal;
						extractFocalLengths(F, f1, f2, fallbackFocal, true);

						// Ensure focal lengths are valid (finite and positive)
						if (!std::isfinite(f1) || f1 <= 0)
							f1 = fallbackFocal;
						if (!std::isfinite(f2) || f2 <= 0)
							f2 = fallbackFocal;

						// Create model descriptor: store 3x3 F matrix + 2 focal lengths
						// We use a 3x4 matrix where last column stores [f1, f2, 0]
						Eigen::MatrixXd descriptor(3, 4);
						descriptor.block<3, 3>(0, 0) = F;
						descriptor(0, 3) = f1;
						descriptor(1, 3) = f2;
						descriptor(2, 3) = 0.0;

						models::Model model;
						model.setData(descriptor);
						models_.push_back(model);
					}

					return models_.size() > 0;
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
		}
	}
}
