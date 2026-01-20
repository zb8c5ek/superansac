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

#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

#include <Eigen/SVD>
#include <cmath>

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// Radial Fundamental Matrix Solver with 9-point minimal solver
			// Estimates F (fundamental matrix) plus lambda1, lambda2 (division distortion parameters)
			// Total unknowns: 10 (8 for F with F[2,2]=1, 2 for lambda1, lambda2)
			// Uses Levenberg-Marquardt numerical optimization
			class RadialFundamentalMatrixNinePointSolver : public AbstractSolver
			{
			public:
				RadialFundamentalMatrixNinePointSolver()
				{
				}

				~RadialFundamentalMatrixNinePointSolver()
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
					return 9;
				}

				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_,
					const size_t *kSample_,
					const size_t kSampleNumber_,
					std::vector<models::Model> &models_,
					const double *kWeights_ = nullptr) const override;

			protected:
				// Helper functions for the solver

				// Compute isotropic scaling factor for normalization
				FORCE_INLINE static double computeScale(
					const Eigen::MatrixXd& pts_centered)
				{
					double mean_r = 0.0;
					for (int i = 0; i < pts_centered.rows(); ++i)
					{
						double r = std::sqrt(pts_centered(i, 0) * pts_centered(i, 0) + 
											  pts_centered(i, 1) * pts_centered(i, 1));
						mean_r += r;
					}
					mean_r /= pts_centered.rows();
					if (mean_r < 1e-12)
						return 1.0;
					return std::sqrt(2.0) / mean_r;
				}

				// Rank-2 projection: set smallest singular value to 0
				FORCE_INLINE static Eigen::Matrix3d rank2Project(const Eigen::Matrix3d& F)
				{
					Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
					Eigen::Vector3d s = svd.singularValues();
					s(2) = 0.0;
					return svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
				}

				// Initialize F using 8-point algorithm (ignoring distortion)
				FORCE_INLINE static Eigen::Matrix3d eightPointFInit(
					const Eigen::MatrixXd& p1n,
					const Eigen::MatrixXd& p2n)
				{
					const int N = p1n.rows();
					Eigen::MatrixXd A(N, 9);
					for (int i = 0; i < N; ++i)
					{
						double x1 = p1n(i, 0), y1 = p1n(i, 1);
						double x2 = p2n(i, 0), y2 = p2n(i, 1);
						A(i, 0) = x2 * x1;
						A(i, 1) = x2 * y1;
						A(i, 2) = x2;
						A(i, 3) = y2 * x1;
						A(i, 4) = y2 * y1;
						A(i, 5) = y2;
						A(i, 6) = x1;
						A(i, 7) = y1;
						A(i, 8) = 1.0;
					}

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
					Eigen::VectorXd f = svd.matrixV().col(8);
					// Map using ROW-MAJOR because we filled A in row-major order
					Eigen::Matrix3d F = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(f.data());
					F = rank2Project(F);
					if (std::abs(F(2, 2)) > 1e-12)
						F = F / F(2, 2);
					return F;
				}

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

				// Compute residuals for the 10 equations
				FORCE_INLINE static Eigen::VectorXd residuals10(
					const Eigen::VectorXd& theta,
					const Eigen::MatrixXd& p1n,
					const Eigen::MatrixXd& p2n)
				{
					Eigen::Matrix3d F;
					F << theta(0), theta(1), theta(2),
						 theta(3), theta(4), theta(5),
						 theta(6), theta(7), 1.0;

					double lam1 = theta(8);
					double lam2 = theta(9);

					Eigen::VectorXd r(10);
					for (int i = 0; i < 9; ++i)
					{
						Eigen::Vector3d x1 = undistHDivision(p1n.row(i).transpose(), lam1);
						Eigen::Vector3d x2 = undistHDivision(p2n.row(i).transpose(), lam2);
						r(i) = x2.dot(F * x1);
					}
					r(9) = F.determinant();
					return r;
				}

				// Levenberg-Marquardt solver with numerical Jacobian
				FORCE_INLINE static bool lmSolve10Eq10Unk(
					Eigen::VectorXd& theta,
					const Eigen::MatrixXd& p1n,
					const Eigen::MatrixXd& p2n,
					int max_iters = 60,
					double mu0 = 1e-3,
					double eps_fd = 1e-6)
				{
					double mu = mu0;
					Eigen::VectorXd r = residuals10(theta, p1n, p2n);
					double cost = 0.5 * r.dot(r);

					for (int iter = 0; iter < max_iters; ++iter)
					{
						// Numerical Jacobian
						Eigen::MatrixXd J(10, 10);
						for (int k = 0; k < 10; ++k)
						{
							Eigen::VectorXd t2 = theta;
							double step = eps_fd * (1.0 + std::abs(theta(k)));
							t2(k) += step;
							Eigen::VectorXd r2 = residuals10(t2, p1n, p2n);
							J.col(k) = (r2 - r) / step;
						}

						Eigen::MatrixXd H = J.transpose() * J;
						H.diagonal() += Eigen::VectorXd::Constant(10, mu);
						Eigen::VectorXd g = J.transpose() * r;

						// Solve linear system
						Eigen::VectorXd delta;
						try
						{
							delta = H.colPivHouseholderQr().solve(-g);
						}
						catch (...)
						{
							return false;
						}

						if (!delta.allFinite())
							return false;

						Eigen::VectorXd t_new = theta + delta;
						Eigen::VectorXd r_new = residuals10(t_new, p1n, p2n);
						double c_new = 0.5 * r_new.dot(r_new);

						if (c_new < cost)
						{
							theta = t_new;
							r = r_new;
							cost = c_new;
							mu = std::max(1e-12, mu * 0.3);
							if (delta.norm() < 1e-10)
								break;
						}
						else
						{
							mu = std::min(1e12, mu * 10.0);
						}
					}

					return std::isfinite(cost);
				}

				// Normalize point coordinates (isotropic scaling only - NO centering)
				// The points are already centered relative to principal point from pybind
				FORCE_INLINE static void normalizePoints(
					const Eigen::MatrixXd& pts,
					Eigen::MatrixXd& pts_norm,
					double& scale)
				{
					// Compute scale for isotropic normalization
					scale = computeScale(pts);
					pts_norm = pts * scale;
				}
			};

			FORCE_INLINE bool RadialFundamentalMatrixNinePointSolver::estimateModel(
				const DataMatrix& kData_,
				const size_t *kSample_,
				const size_t kSampleNumber_,
				std::vector<models::Model> &models_,
				const double *kWeights_) const
			{
				try
				{
					// Only support exactly 9 points (minimal case)
					if (kSampleNumber_ != 9)
					{
						return false;
					}

					// Extract 9 point correspondences
					Eigen::MatrixXd pts1(9, 2), pts2(9, 2);
					for (size_t i = 0; i < 9; ++i)
					{
						size_t idx = kSample_[i];
						
						if (idx >= static_cast<size_t>(kData_.rows()))
						{
							return false;
						}
						
						pts1(i, 0) = kData_(idx, 0);
						pts1(i, 1) = kData_(idx, 1);
						pts2(i, 0) = kData_(idx, 2);
						pts2(i, 1) = kData_(idx, 3);
					}
					// Points are already normalized (centered + scaled) from pybind
					// No additional normalization needed
					Eigen::MatrixXd p1n = pts1;
					Eigen::MatrixXd p2n = pts2;
					double s1 = 1.0;
					double s2 = 1.0;

					// Initialize F using 8-point method
					Eigen::Matrix3d F0 = eightPointFInit(p1n, p2n);
					if (std::abs(F0(2, 2)) < 1e-8)
					{
						F0 = F0 / (F0.norm() + 1e-12);
						if (std::abs(F0(2, 2)) < 1e-8)
							F0(2, 2) = 1.0;
					}
					if (std::abs(F0(2, 2)) > 1e-12)
						F0 = F0 / F0(2, 2);

					// Set up initial parameter vector
					Eigen::VectorXd theta0(10);
					theta0 << F0(0, 0), F0(0, 1), F0(0, 2),
							  F0(1, 0), F0(1, 1), F0(1, 2),
							  F0(2, 0), F0(2, 1),
							  0.0, 0.0; // Initial lambda values

					if (!theta0.allFinite())
					{
						return false;
					}
					
					// Compute initial cost (with lambda=0)
					Eigen::VectorXd r_init = residuals10(theta0, p1n, p2n);
					double cost_init = 0.5 * r_init.dot(r_init);

					// Solve with Levenberg-Marquardt
					Eigen::VectorXd theta = theta0;
					if (!lmSolve10Eq10Unk(theta, p1n, p2n))
					{
						theta = theta0;  // Use initial solution
					}
					else
					{
						// Check if LM made things better
						Eigen::VectorXd r_final = residuals10(theta, p1n, p2n);
						double cost_final = 0.5 * r_final.dot(r_final);

						// If LM made things significantly worse, reject it
						if (cost_final > cost_init * 1.5 && cost_init < 1e-6)
						{
							theta = theta0;
						}
					}

					// Extract F and lambda values from solution
					Eigen::Matrix3d Fn;
					Fn << theta(0), theta(1), theta(2),
						  theta(3), theta(4), theta(5),
						  theta(6), theta(7), 1.0;

					Fn = rank2Project(Fn);
					if (std::abs(Fn(2, 2)) > 1e-12)
						Fn = Fn / Fn(2, 2);

					// Denormalize: F_c = D2^T F_n D1
					Eigen::Matrix3d D1, D2;
					D1 << s1, 0.0, 0.0,
						  0.0, s1, 0.0,
						  0.0, 0.0, 1.0;
					D2 << s2, 0.0, 0.0,
						  0.0, s2, 0.0,
						  0.0, 0.0, 1.0;
					Eigen::Matrix3d Fc = D2.transpose() * Fn * D1;

					// Denormalize lambda values
					double lam1 = theta(8) * (s1 * s1);
					double lam2 = theta(9) * (s2 * s2);

					// Create model descriptor: store 3x3 F matrix + 2 lambda values
					// We'll use a 3x4 matrix where last column stores [lam1, lam2, 0]
					Eigen::MatrixXd descriptor(3, 4);
					descriptor.block<3, 3>(0, 0) = Fc;
					descriptor(0, 3) = lam1;
					descriptor(1, 3) = lam2;
					descriptor(2, 3) = 0.0;

					if (!descriptor.allFinite())
						return false;

					// Create and add model
					models::Model model;
					model.setData(descriptor);
					models_.push_back(model);

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
		}
	}
}
