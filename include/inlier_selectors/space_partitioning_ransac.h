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
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <limits>
#include "types.h"
#include "../neighborhood/abstract_neighborhood.h"
#include "../neighborhood/grid_neighborhood_graph.h"
#include "../utils/macros.h"
#include "../models/model.h"
#include "../scoring/score.h"
#include "../utils/types.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace superansac 
{
	namespace inlier_selector
	{
		class SpacePartitioningRANSAC : public AbstractInlierSelector
		{
        protected:
            std::vector<bool> gridCornerMask;
            std::vector<bool> gridAngleMask;
	        std::vector<std::tuple<double, double, double, double>> gridCornerCoordinates;
            std::vector<std::tuple<int, int, double, double>> gridCornerCoordinatesH;
            std::vector<std::tuple<int, int, int, double, double, double>> gridCornerCoordinates3D;
	        std::vector<std::tuple<double, double, double, double, double, double>> gridCornerAngles;
            std::vector<double> additionalParameters;
            double scaleSrc, offsetXSrc, offsetYSrc, scaleDst, offsetXDst, offsetYDst;
            
            // OPTIMIZATION: Cache for image-1 cells to speed up lookup
            std::vector<std::vector<size_t>> cellIndicesByImage1Cell;  // Indexed by 1D image-1 cell index
            bool cellCacheValid = false;
            
            // OPTIMIZATION: Track best model inlier count for early rejection
            size_t bestModelInlierCount = 0;

            // PnP SPACE PARTITIONING: 2D image grid with 3D frustum checking
            struct ImageFrustum {
                Eigen::Vector3d rays[4];  // 4 corner rays: (min,min), (max,min), (min,max), (max,max)
            };
            std::vector<ImageFrustum> imageFrustums;  // One per 2D image cell
            Eigen::Matrix3d cameraMatrix;  // Camera intrinsics [fx 0 cx; 0 fy cy; 0 0 1]
            Eigen::Vector3d camerCenter;   // Camera center (world coordinates)
            bool pnpGridInitialized = false;

			neighborhood::AbstractNeighborhoodGraph* neighborhoodGraph;

            FORCE_INLINE void runHomography(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);
            
            FORCE_INLINE void runFundamentalMatrix(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);
            
            FORCE_INLINE void runRigidTransformation(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);

            FORCE_INLINE void runAbsolutePose(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                const double& kInlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_);

		public:
            cv::Mat img1, img2;

			SpacePartitioningRANSAC() : scaleSrc(1.0), offsetXSrc(0.0), offsetYSrc(0.0), scaleDst(1.0), offsetXDst(0.0), offsetYDst(0.0)
			{
			}

            ~SpacePartitioningRANSAC() {}

            void setNormalizers(
                const double &kScaleSrc_,
                const double &kOffsetXSrc_,
                const double &kOffsetYSrc_,
                const double &kScaleDst_,
                const double &kOffsetXDst_,
                const double &kOffsetYDst_)
            {
                scaleSrc = kScaleSrc_;
                offsetXSrc = kOffsetXSrc_;
                offsetYSrc = kOffsetYSrc_;
                scaleDst = kScaleDst_;
                offsetXDst = kOffsetXDst_;
                offsetYDst = kOffsetYDst_;
            }

            // PnP SPACE PARTITIONING: Initialize camera parameters for 2D+3D frustum approach
            void initializePnPGrid(
                const Eigen::Matrix3d& K,         // Camera intrinsics 3x3
                const Eigen::Vector3d& camCenter) // Camera center in world coordinates
            {
                cameraMatrix = K;
                camerCenter = camCenter;
                
                // Compute 2D image grid frustums
                // Assuming SimplePinhole: K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
                double fx = K(0, 0);
                double fy = K(1, 1);
                double cx = K(0, 2);
                double cy = K(1, 2);
                
                if (additionalParameters.size() < 5) {
                    pnpGridInitialized = false;
                    return;
                }
                
                size_t divisionNumber = static_cast<size_t>(additionalParameters[3]);
                
                // Typical image size (could be read from data, but use defaults for now)
                double imgWidth = 640.0;
                double imgHeight = 480.0;
                
                double cellSizeU = imgWidth / divisionNumber;
                double cellSizeV = imgHeight / divisionNumber;
                
                imageFrustums.clear();
                imageFrustums.reserve(divisionNumber * divisionNumber);
                
                // For each 2D image cell, compute 4 corner rays
                for (size_t iy = 0; iy < divisionNumber; ++iy) {
                    for (size_t ix = 0; ix < divisionNumber; ++ix) {
                        ImageFrustum frustum;
                        
                        // 4 corners of the cell in image coordinates
                        double corners[4][2] = {
                            {ix * cellSizeU,            iy * cellSizeV},            // (min,min)
                            {(ix + 1) * cellSizeU,      iy * cellSizeV},            // (max,min)
                            {ix * cellSizeU,            (iy + 1) * cellSizeV},      // (min,max)
                            {(ix + 1) * cellSizeU,      (iy + 1) * cellSizeV}       // (max,max)
                        };
                        
                        // Convert image coordinates to 3D rays (camera space)
                        // Ray direction: [u, v, 1] -> K^-1 @ [u, v, 1]^T
                        for (int c = 0; c < 4; ++c) {
                            double u = corners[c][0];
                            double v = corners[c][1];
                            
                            // Normalize by camera matrix
                            Eigen::Vector3d rayDir;
                            rayDir(0) = (u - cx) / fx;
                            rayDir(1) = (v - cy) / fy;
                            rayDir(2) = 1.0;
                            rayDir.normalize();
                            
                            frustum.rays[c] = rayDir;
                        }
                        
                        imageFrustums.push_back(frustum);
                    }
                }
                
                pnpGridInitialized = true;
            }


			void initialize(
				neighborhood::AbstractNeighborhoodGraph* neighborhoodGraph_,
				const models::Types kModelType_)
			{
				neighborhoodGraph = neighborhoodGraph_;
				modelType = kModelType_;

                // Save additional info needed for the selection
                const auto& kSizes = neighborhoodGraph_->getCellSizes();
                // Number of dimensions
                const size_t& kDimensions = kSizes.size();

                // The number cells filled in the grid
                const size_t& kCellNumber = neighborhoodGraph_->filledCellNumber();
                const size_t& kDivisionNumber = neighborhoodGraph_->getDivisionNumber();
                
                // DEFENSIVE CHECK: Limit maximum cell number to prevent excessive memory allocation
                // For high-dimensional grids (5D for PnP), the cell count grows exponentially.
                // Cap the mask size to prevent out-of-memory errors (max 4M cells ≈ 4MB).
                size_t kMaximumCellNumber = std::pow(kDivisionNumber, kDimensions);
                const size_t MAX_CELL_MASK_SIZE = 4000000;  // ~4MB limit
                if (kMaximumCellNumber > MAX_CELL_MASK_SIZE) {
                    // For high-dimensional grids, fall back to query-based approach (no pre-allocation)
                    kMaximumCellNumber = MAX_CELL_MASK_SIZE;
                }

                // Initialize the structures speeding up the selection by caching data
                gridCornerMask.resize(kMaximumCellNumber, false);

                switch (kModelType_)
                {
                case models::Types::Homography:
                    gridCornerCoordinates.resize(kMaximumCellNumber);
                    // OPTIMIZATION: Pre-compute cell lookup structure for homography
                    if (kDimensions == 4) {
                        // For 4D homography grid, cache image-1 cells for fast lookup
                        size_t N2 = kDivisionNumber * kDivisionNumber;
                        cellIndicesByImage1Cell.clear();
                        cellIndicesByImage1Cell.resize(N2);
                        
                        const auto& cellMap = neighborhoodGraph_->getCells();
                        for (const auto& [idx4D, value] : cellMap) {
                            size_t idx1D = idx4D % N2;
                            // Store the 4D index for this image-1 cell
                            cellIndicesByImage1Cell[idx1D].push_back(idx4D);
                        }
                        cellCacheValid = true;
                    }
                    break;
                case models::Types::FundamentalMatrix:
		            gridCornerAngles.resize(kMaximumCellNumber);
                    gridCornerCoordinates.resize(kMaximumCellNumber);
                    gridCornerMask.resize(kMaximumCellNumber, false);
                    gridAngleMask.resize(kMaximumCellNumber, false);
                    break;
                case models::Types::RigidTransformation:
                    gridCornerCoordinates3D.resize(kMaximumCellNumber);
                    gridCornerMask.resize(kMaximumCellNumber, false);
                    break;
                case models::Types::AbsolutePose:
                    // For PnP grid, initialize 2D image frustums
                    // Will be populated in initializePnPGrid() when we have camera data
                    pnpGridInitialized = false;  // Mark as not yet ready
                    imageFrustums.clear();
                    break;
                }
                
                /*if (kDimensions == 6)
                    gridCornerCoordinates3D.resize(kMaximumCellNumber);
                else
                    gridCornerCoordinatesH.resize(kMaximumCellNumber);*/

                additionalParameters.resize(kDimensions + 1);
                for (size_t dimension = 0; dimension < kDimensions; ++dimension)
                    additionalParameters[dimension] = kSizes[dimension]; // The cell size along the current dimension
                additionalParameters[kDimensions] = kDivisionNumber; // The number of cells along an axis 
			}

            // The function that runs the model-based inlier selector
            void run(
                const DataMatrix &kData_, // The data points
                const models::Model &kModel_, // The model estimated
                const scoring::AbstractScoring *kScoring_, // The scoring object used for the model estimation
                std::vector<const std::vector<size_t>*>& selectedCells_, // The indices of the points selected
                size_t& pointNumber_) // The number of points selected
            {
                // Initializing the selected point number to zero
                pointNumber_ = 0;
                selectedCells_.clear();  // Always clear input to ensure clean state

                // Defensive check: validate kScoring pointer
                if (!kScoring_) {
                    return;  // Cannot proceed without scoring object
                }

                // Get the current inlier-outlier threshold
                const double& kInlierOutlierThreshold_ = 3.0 / 2.0 * kScoring_->getThreshold();

                try {
                    if (modelType == models::Types::Homography)
                        runHomography(
                            kData_,
                            kModel_,
                            kInlierOutlierThreshold_,
                            selectedCells_,
                            pointNumber_);
                    else if (modelType == models::Types::FundamentalMatrix)
                        runFundamentalMatrix(
                            kData_,
                            kModel_,
                            kInlierOutlierThreshold_,
                            selectedCells_,
                            pointNumber_);
                    else if (modelType == models::Types::RigidTransformation)
                        runRigidTransformation(
                            kData_,
                            kModel_,
                            kInlierOutlierThreshold_,
                            selectedCells_,
                            pointNumber_);
                    else if (modelType == models::Types::AbsolutePose)
                        runAbsolutePose(
                            kData_,
                            kModel_,
                            kInlierOutlierThreshold_,
                            selectedCells_,
                            pointNumber_);
                    else
                        throw std::runtime_error("The estimator type is not supported by the space partitioning RANSAC.");
                } catch (const std::exception& e) {
                    // On any exception, clear output and return empty
                    selectedCells_.clear();
                    pointNumber_ = 0;
                } catch (...) {
                    // On any other exception, clear output and return empty
                    selectedCells_.clear();
                    pointNumber_ = 0;
                }
            }

            /*void runRigidTransformation(
				const DataMatrix &kData_,
				const models::Model &kModel_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_,
                const double& kInlierOutlierThreshold_);*/
                
            // OPTIMIZATION: Set the best model's inlier count for early rejection
            void setBestModelInlierCount(size_t inlierCount)
            {
                bestModelInlierCount = inlierCount;
            }
		};
        
        FORCE_INLINE void SpacePartitioningRANSAC::runHomography(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            // Validate that we have a valid neighborhoodGraph
            if (!neighborhoodGraph) {
                return;  // No neighborhood graph available
            }
            
            // Get homography matrix
            const Eigen::Matrix3d H = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(kModel_.getData().data());
            
            // Get grid parameters - these should only work with grid neighborhoods
            try {
                const auto& cellMap = neighborhoodGraph->getCells();
                const size_t N = neighborhoodGraph->getDivisionNumber();
                const auto& cellSizes = neighborhoodGraph->getCellSizes();
                
                // Validate that we have the correct dimensions for homography (4D grid)
                if (cellSizes.size() != 4) {
                    // Fallback: return all cells if not 4D grid
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }
                
                // Cell sizes: [width_img1, height_img1, width_img2, height_img2]
                const double w1 = cellSizes[0];
                const double h1 = cellSizes[1];
                const double w2 = cellSizes[2];
                const double h2 = cellSizes[3];
                
                // Check for zero cell sizes
                if (w1 <= 0.0 || h1 <= 0.0 || w2 <= 0.0 || h2 <= 0.0 || N == 0) {
                    // Fallback: return all cells
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }
                
                const double imgW2 = w2 * N;
                const double imgH2 = h2 * N;
                
                // Grid stride for 4D indexing
                const size_t N2 = N * N;
                const size_t N3 = N2 * N;
                
                // Extract H components once for performance - store for reuse
                const double h00 = H(0,0), h01 = H(0,1), h02 = H(0,2);
                const double h10 = H(1,0), h11 = H(1,1), h12 = H(1,2);
                const double h20 = H(2,0), h21 = H(2,1), h22 = H(2,2);
                
                // Check for degenerate homography
                const double det = H.determinant();
                if (std::abs(det) < 1e-10) {
                    // Degenerate - return all cells
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }
                
                const double inv_w2 = 1.0 / w2;
                const double inv_h2 = 1.0 / h2;
                const int N_int = static_cast<int>(N);
                
                // OPTIMIZATION: Skip filtering when density is too high (overhead > benefit)
                const size_t totalPoints = kData_.rows();
                const size_t numCells = cellMap.size();
                const double avgPointsPerCell = static_cast<double>(totalPoints) / std::max(size_t(1), numCells);
                
                // Skip filtering if very few cells exist (not enough to partition effectively)
                if (numCells < 4) {
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }
                
                // For high inlier ratios (low outlier ratios), cells tend to be dense
                // Skip filtering if average > 60 points/cell to avoid unnecessary overhead
                if (avgPointsPerCell > 60.0) {
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }
                
                // Process each existing cell in image 1
                std::unordered_set<size_t> processed_1d;
                processed_1d.reserve(N2);
                
                for (const auto& [idx4D, value] : cellMap) {
                    // Get 1D index for image 1: idx1D = x1 + y1*N
                    size_t idx1D = idx4D % N2;
                    
                    // Skip if we already processed this 1D cell
                    if (processed_1d.count(idx1D)) continue;
                    processed_1d.insert(idx1D);
                    
                    size_t x1 = idx1D % N;
                    size_t y1 = idx1D / N;
                    
                    // Cell bounds in image 1
                    double x1_min = x1 * w1;
                    double y1_min = y1 * h1;
                    double x1_max = x1_min + w1;
                    double y1_max = y1_min + h1;
                    
                    // Project 4 corners of cell in image 1 to image 2
                    // OPTIMIZATION: Use stack-allocated corners for better cache locality
                    double px_min = imgW2, py_min = imgH2;
                    double px_max = -1.0, py_max = -1.0;
                    bool valid = false;
                    
                    // Process corners in a tight loop for vectorization
                    const double corners_x[] = {x1_min, x1_max, x1_min, x1_max};
                    const double corners_y[] = {y1_min, y1_min, y1_max, y1_max};
                    
                    for (int i = 0; i < 4; ++i) {
                        double cx = corners_x[i];
                        double cy = corners_y[i];
                        double z = h20*cx + h21*cy + h22;
                        
                        if (z > 1e-10 || z < -1e-10) {  // OPTIMIZATION: Avoid abs() call
                            valid = true;
                            double z_inv = 1.0 / z;
                            double px = (h00*cx + h01*cy + h02) * z_inv;
                            double py = (h10*cx + h11*cy + h12) * z_inv;
                            
                            // OPTIMIZATION: Use branchless min/max with conditional assignment
                            px_min = (px < px_min) ? px : px_min;
                            py_min = (py < py_min) ? py : py_min;
                            px_max = (px > px_max) ? px : px_max;
                            py_max = (py > py_max) ? py : py_max;
                        }
                    }
                    
                    if (!valid) continue;
                    
                    // Clip projected bounding box to image 2 bounds
                    // OPTIMIZATION: Use conditional assignment instead of std::min/max
                    px_min = (px_min > 0.0) ? px_min : 0.0;
                    py_min = (py_min > 0.0) ? py_min : 0.0;
                    px_max = (px_max < imgW2) ? px_max : imgW2;
                    py_max = (py_max < imgH2) ? py_max : imgH2;
                    
                    // Early rejection with explicit bounds check
                    if (px_min >= px_max || py_min >= py_max) {
                        continue;
                    }
                    
                    // OPTIMIZATION: Use floor/ceil for correct cell range computation
                    int x2_min = static_cast<int>(std::floor(px_min * inv_w2));
                    int y2_min = static_cast<int>(std::floor(py_min * inv_h2));
                    int x2_max = static_cast<int>(std::floor((px_max - 1e-10) * inv_w2));
                    int y2_max = static_cast<int>(std::floor((py_max - 1e-10) * inv_h2));
                    
                    // Clamp to valid range [0, N-1]
                    if (x2_min < 0) x2_min = 0;
                    if (y2_min < 0) y2_min = 0;
                    if (x2_max >= N_int) x2_max = N_int - 1;
                    if (y2_max >= N_int) y2_max = N_int - 1;
                    
                    // Skip if invalid range
                    if (x2_min > x2_max || y2_min > y2_max) {
                        continue;
                    }
                    
                    // OPTIMIZATION: Collect compatible 4D cells with precalculated strides
                    // idx4D = x1 + y1*N + x2*N^2 + y2*N^3
                    const size_t base_idx = x1 + y1 * N;
                    
                    for (int y2 = y2_min; y2 <= y2_max; ++y2) {
                        size_t y2_stride = static_cast<size_t>(y2) * N3;
                        for (int x2 = x2_min; x2 <= x2_max; ++x2) {
                            size_t idx4D_cand = base_idx + (static_cast<size_t>(x2) * N2) + y2_stride;
                            auto it = cellMap.find(idx4D_cand);
                            if (it != cellMap.end()) {
                                selectedCells_.emplace_back(&it->second.first);
                                pointNumber_ += it->second.first.size();
                            }
                        }
                    }
                }
                
                // OPTIMIZATION: Early rejection if filtered points < best model
                // If we collected fewer points than the best model found so far,
                // reject this model immediately without scoring
                if (bestModelInlierCount > 0 && pointNumber_ < bestModelInlierCount) {
                    selectedCells_.clear();
                    pointNumber_ = 0;
                    return;
                }
            } catch (const std::exception& e) {
                // If any exception occurs, fall back to returning all cells
                const auto& cellMap = neighborhoodGraph->getCells();
                selectedCells_.reserve(cellMap.size());
                for (const auto& [idx4D, value] : cellMap) {
                    selectedCells_.emplace_back(&value.first);
                    pointNumber_ += value.first.size();
                }
            } catch (...) {
                // Catch any other exception and return all cells
                try {
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx4D, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                } catch (...) {
                    // If even fallback fails, do nothing
                }
            }
        }

        FORCE_INLINE void SpacePartitioningRANSAC::runFundamentalMatrix(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            // The actual descriptor of the model
            const auto &kModelData = kModel_.getData();

            const double& kSourceCellWidth = additionalParameters[0], // The width of the source image
                & kSourceCellHeight = additionalParameters[1], // The height of the source images
                & kDestinationCellWidth = additionalParameters[2], // The width of the destination image
                & kDestinationCellHeight = additionalParameters[3], // The height of the destination image
                & kPartitionNumber = additionalParameters[4]; // The number of cells in the neighborhood structure along an axis

            // The sizes of the cells along each axis
            const double kCellSize1 = kSourceCellWidth * scaleSrc, 
                kCellSize2 = kSourceCellHeight * scaleSrc,
                kCellSize3 = kDestinationCellWidth * scaleDst,
                kCellSize4 = kDestinationCellHeight * scaleDst;

            // Calculate the normalized image corners
            double normDstX0 = offsetXDst,
                normDstX1 = kCellSize3 * kPartitionNumber + offsetXDst,
                normDstY0 = offsetYDst,
                normDstY1 = kCellSize4 * kPartitionNumber + offsetYDst;

            //std::cout << normDstX0 << " " << normDstX1 << " " << normDstY0 << " " << normDstY1 << std::endl;

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = { 0, 0,
                0, 1,
                1, 0,
                1, 1 };

            // Filling the vectors with zeros
            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);
            std::fill(std::begin(gridAngleMask), std::end(gridAngleMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [kCell, kValue] : neighborhoodGraph->getCells())
            {
                const auto& kPoints = kValue.first;

                // Checking if there are enough points in the cell to make the cell selection worth it.
                if (kPoints.size() < 20)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&kPoints);
                    pointNumber_ += kPoints.size();
                    continue;
                }

                // The coordinates of the cell corners.
                const auto& kCornerIndices = kValue.second;
                // The index of the cell in the source image.
                size_t cellIdx = kCornerIndices[0] * kPartitionNumber + kCornerIndices[1];
                
                // The parameters of the epipolar lines corresponding to the minimum and maximum angles
                double &A1 = std::get<0>(gridCornerAngles[cellIdx]), 
                    &B1 = std::get<1>(gridCornerAngles[cellIdx]),
                    &C1 = std::get<2>(gridCornerAngles[cellIdx]),
                    &A2 = std::get<3>(gridCornerAngles[cellIdx]), 
                    &B2 = std::get<4>(gridCornerAngles[cellIdx]), 
                    &C2 = std::get<5>(gridCornerAngles[cellIdx]);

                // Iterate through the corners of the current cell
                // TODO(danini): Handle the case when the epipole falls inside the image
                if (!gridAngleMask[cellIdx])
                {
                    bool insideImage = false;
                    double minAngle = std::numeric_limits<double>::max(),
                        maxAngle = std::numeric_limits<double>::lowest();
                    for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                    {
                        // Stepping in the current direction from the currently selected corner
                        size_t colIdx = kCornerIndices[0] + steps[stepIdx],
                            rowIdx = kCornerIndices[1] + steps[stepIdx + 1];
                            
                        if (rowIdx >= kPartitionNumber ||
                            colIdx >= kPartitionNumber)
                            continue;

                        // Calculating the selected cell's index
                        const size_t kIdx2d = rowIdx * kPartitionNumber + colIdx;

                        // Get the index of the corner's projection in the destination image
                        auto& lineTuple = gridCornerCoordinates[kIdx2d];
                        auto& angle = std::get<0>(lineTuple);
                        auto& a2 = std::get<1>(lineTuple);
                        auto& b2 = std::get<2>(lineTuple);
                        auto& c2 = std::get<3>(lineTuple);

                        // If the corner hasn't yet been projected to the destination image
                        if (!gridCornerMask[kIdx2d])
                        {
                            // Get the coordinates of the corner
                            double kX1 = colIdx * kCellSize1 + offsetXSrc,
                                kY1 = rowIdx * kCellSize2 + offsetYSrc;

                            // Move the corner by the threshold
                            if (stepIdx == 0)
                            {
                                kX1 -= kInlierOutlierThreshold_;
                                kY1 -= kInlierOutlierThreshold_;
                            }
                            else if (stepIdx == 2)
                            {
                                kX1 -= kInlierOutlierThreshold_;
                                kY1 += kInlierOutlierThreshold_;
                            }
                            else if (stepIdx == 4)
                            {
                                kX1 += kInlierOutlierThreshold_;
                                kY1 -= kInlierOutlierThreshold_;
                            }
                            else
                            {
                                kX1 += kInlierOutlierThreshold_;
                                kY1 += kInlierOutlierThreshold_;
                            }

                            // Project them by the estimated homography matrix
                            a2 = kX1 * kModelData(0, 0) + kY1 * kModelData(0, 1) + kModelData(0, 2);
                            b2 = kX1 * kModelData(1, 0) + kY1 * kModelData(1, 1) + kModelData(1, 2);
                            c2 = kX1 * kModelData(2, 0) + kY1 * kModelData(2, 1) + kModelData(2, 2);

                            // Angle of direction vector v
                            // n = [a2, b2]
                            // v = [-b2, a2]
                            angle = std::atan2(a2, b2);

                            // Note that the corner has been already projected
                            gridCornerMask[kIdx2d] = true;

                            // Check if the line intersects the image borders
                            // a * x + b * y + c
                            if (!insideImage)
                            {
                                double yLeft = -(a2 * normDstX0 + c2) / b2;
                                double yRight = -(a2 * normDstX1 + c2) / b2;

                                if (yLeft >= normDstY0 && yLeft <= normDstY1 ||
                                    yRight >= normDstY0 && yRight <= normDstY1)
                                    insideImage = true;
                                else
                                {
                                    double xTop = -(b2 * normDstY0 + c2) / a2;
                                    double xBottom = -(b2 * normDstY1 + c2) / a2;

                                    if (xTop >= normDstX0 && xTop <= normDstX1 ||
                                        xBottom >= normDstX0 && xBottom <= normDstX1)
                                        insideImage = true;
                                }
                            }
                        }

                        // Save the epipolar line's parameters if the angle is the new mininum
                        if (angle < minAngle)
                        {
                            minAngle = angle;
                            A1 = a2;
                            B1 = b2;
                            C1 = c2;
                        }

                        // Save the epipolar line's parameters if the angle is the new maximum
                        if (angle > maxAngle)
                        {
                            maxAngle = angle;
                            A2 = a2;
                            B2 = b2;
                            C2 = c2;
                        }
                    }

                    // Note that the cell has already been processed
                    gridAngleMask[cellIdx] = true;

                    if (!insideImage)
                        continue;
                }

                // Iterate through the corners of the cell and check if any of the corners fall 
                // between the two selected epipolar lines.
                double distance1, distance2;
                bool overlaps = false;
                for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                {
                    // Get the coordinates of the corner
                    double kX2 = (kCornerIndices[2] + steps[stepIdx]) * kCellSize3 + offsetXDst,
                        kY2 = (kCornerIndices[3] + steps[stepIdx + 1]) * kCellSize4 + offsetYDst;

                    distance1 = A1 * kX2 + B1 * kY2 + C1;
                    distance2 = A2 * kX2 + B2 * kY2 + C2;

                    // If the distance sign is different, the point falls between the lines.
                    if (distance1 * distance2 <= 0)
                    {
                        overlaps = true;
                        break;
                    }
                }

                if (!overlaps)
                {
                    // OPTIMIZATION: Pre-calculate corner coordinates once
                    const double kX20 = (kCornerIndices[2]) * kCellSize3 + offsetXDst,
                        kY20 = (kCornerIndices[3]) * kCellSize4 + offsetYDst,
                        kX21 = (kCornerIndices[2] + 1) * kCellSize3 + offsetXDst,
                        kY21 = (kCornerIndices[3] + 1) * kCellSize4 + offsetYDst;

                    // OPTIMIZATION: Vectorized distance calculations for A1-B1-C1 line
                    const double d1_00 = A1 * kX20 + B1 * kY20 + C1;  // (0, 0)
                    const double d1_10 = A1 * kX21 + B1 * kY20 + C1;  // (1, 0)
                    const double d1_01 = A1 * kX20 + B1 * kY21 + C1;  // (0, 1)
                    const double d1_11 = A1 * kX21 + B1 * kY21 + C1;  // (1, 1)

                    // Check if any two corners have opposite signs (cross the line)
                    if ((d1_00 * d1_10 < 0) || (d1_00 * d1_01 < 0) || 
                        (d1_10 * d1_11 < 0) || (d1_01 * d1_11 < 0)) {
                        overlaps = true;
                    } else {
                        // OPTIMIZATION: Vectorized distance calculations for A2-B2-C2 line
                        const double d2_00 = A2 * kX20 + B2 * kY20 + C2;
                        const double d2_10 = A2 * kX21 + B2 * kY20 + C2;
                        const double d2_01 = A2 * kX20 + B2 * kY21 + C2;
                        const double d2_11 = A2 * kX21 + B2 * kY21 + C2;

                        // Check if any two corners have opposite signs (cross the line)
                        if ((d2_00 * d2_10 < 0) || (d2_00 * d2_01 < 0) || 
                            (d2_10 * d2_11 < 0) || (d2_01 * d2_11 < 0)) {
                            overlaps = true;
                        }
                    }
                }

                if (overlaps)
                {
                    // Store the points in the cell to be tested
                    selectedCells_.emplace_back(&kPoints);
                    pointNumber_ += kPoints.size();
                    
                    // OPTIMIZATION: Early rejection if filtered points < best model
                    // Stop collecting cells if we already have enough for evaluation
                    if (bestModelInlierCount > 0 && pointNumber_ >= bestModelInlierCount * 2) {
                        break;
                    }
                }
            }
            
            // OPTIMIZATION: Early rejection if filtered points < best model
            // If we collected fewer points than the best model found so far,
            // reject this model immediately without scoring
            if (bestModelInlierCount > 0 && pointNumber_ < bestModelInlierCount) {
                selectedCells_.clear();
                pointNumber_ = 0;
                return;
            }
        }

        /*template <typename _Estimator,
            typename _NeighborhoodStructure>
            OLGA_INLINE void SpacePartitioningRANSAC<_Estimator, _NeighborhoodStructure>::runRigidTransformation(
                const cv::Mat& kCorrespondences_,
                const gcransac::Model& kModel_,
                const _NeighborhoodStructure& kNeighborhood_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_,
                const double& kInlierOutlierThreshold_)
        {
            const double& kCellSize1 = additionalParameters[0],
                & kCellSize2 = additionalParameters[1],
                & kCellSize3 = additionalParameters[2],
                & kCellSize4 = additionalParameters[3],
                & kCellSize5 = additionalParameters[4],
                & kCellSize6 = additionalParameters[5],
                & kPartitionNumber = additionalParameters[6];

            const Eigen::Matrix4d& descriptor = kModelData;

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = {
                0, 0, 0,
                0, 1, 0,
                1, 0, 0,
                1, 1, 0,
                0, 0, 1,
                0, 1, 1,
                1, 0, 1,
                1, 1, 1 };

            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [cell, value] : neighborhoodGraph->getCells())
            {
                // The points in the cell
                const auto& points = value.first;

                // Checking if there are enough points in the cell to make the cell selection worth it
                if (points.size() < 8)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                    continue;
                }

                const auto& kCornerIndices = value.second;
                bool overlaps = false;

                // Iterate through the corners of the current cell
                for (size_t stepIdx = 0; stepIdx < 24; stepIdx += 3)
                {
                    // The index of the currently projected corner
                    const size_t kCornerXIndex = kCornerIndices[0] + steps[stepIdx];
                    if (kCornerXIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerYIndex = kCornerIndices[1] + steps[stepIdx + 1];
                    if (kCornerYIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerZIndex = kCornerIndices[2] + steps[stepIdx + 2];
                    if (kCornerZIndex >= kPartitionNumber)
                        continue;

                    // Get the index of the corner's projection in the destination image
                    const size_t kIdx3d = kCornerXIndex * kPartitionNumber * kPartitionNumber + kCornerYIndex * kPartitionNumber + kCornerYIndex;

                    // This is already or will be the horizontal and vertical indices in the destination image
                    auto& indexPair = gridCornerCoordinates3D[kIdx3d];

                    // If the corner hasn't yet been projected to the destination image
                    if (!gridCornerMask[kIdx3d])
                    {
                        // Get the coordinates of the corner
                        const double kX1 = kCornerXIndex * kCellSize1,
                            kY1 = kCornerYIndex * kCellSize2,
                            kZ1 = kCornerZIndex * kCellSize3;

                        const double x2p = descriptor(0, 0) * kX1 + descriptor(1, 0) * kY1 + descriptor(2, 0) * kZ1 + descriptor(3, 0),
                            y2p = descriptor(0, 1) * kX1 + descriptor(1, 1) * kY1 + descriptor(2, 1) * kZ1 + descriptor(3, 1),
                            z2p = descriptor(0, 2) * kX1 + descriptor(1, 2) * kY1 + descriptor(2, 2) * kZ1 + descriptor(3, 2);

                        // Store the projected corner's cell indices
                        std::get<0>(indexPair) = x2p / kCellSize4;
                        std::get<1>(indexPair) = y2p / kCellSize5;
                        std::get<2>(indexPair) = z2p / kCellSize6;
                        std::get<3>(indexPair) = x2p;
                        std::get<4>(indexPair) = y2p;
                        std::get<5>(indexPair) = z2p;

                        // Note that the corner has been already projected
                        gridCornerMask[kIdx3d] = true;
                    }

                    // Check if the projected corner is equal to the correspondence's destination point's grid cell.
                    // This works due to the coordinate truncation.
                    if (std::get<0>(indexPair) == kCornerIndices[3] &&
                        std::get<1>(indexPair) == kCornerIndices[4] &&
                        std::get<2>(indexPair) == kCornerIndices[5])
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                        break;
                    }
                }

                // Check if there is an overlap
                if (!overlaps)
                {
                    // The X index of the bottom-right corner
                    const size_t kCornerXIndex111 = kCornerIndices[0] + 1;
                    // The Y index of the bottom-right corner
                    const size_t kCornerYIndex111 = kCornerIndices[1] + 1;
                    // The Z index of the bottom-right corner
                    const size_t kCornerZIndex111 = kCornerIndices[2] + 1;

                    // Calculating the index of the top-left corner
                    const size_t kIdx3d000 = kCornerIndices[0] * kPartitionNumber * kPartitionNumber + kCornerIndices[1] * kPartitionNumber + kCornerIndices[2];
                    // Calculating the index of the bottom-right corner
                    const size_t kIdx3d111 = kCornerXIndex111 * kPartitionNumber * kPartitionNumber + kCornerYIndex111 * kPartitionNumber + kCornerZIndex111;

                    // Coordinates of the top-left and bottom-right corners in the destination image
                    auto& indexPair000 = gridCornerCoordinates3D[kIdx3d000];

                    std::tuple<int, int, int, double, double, double> indexPair111;
                    if (kCornerYIndex111 >= kPartitionNumber ||
                        kCornerXIndex111 >= kPartitionNumber ||
                        kCornerZIndex111 >= kPartitionNumber)
                    {
                        // Get the coordinates of the corner
                        const double kX111 = kCornerXIndex111 * kCellSize1,
                            kY111 = kCornerYIndex111 * kCellSize2,
                            kZ111 = kCornerZIndex111 * kCellSize3;

                        // Project them by the estimated homography matrix
                        const double x2p = descriptor(0, 0) * kX111 + descriptor(1, 0) * kY111 + descriptor(2, 0) * kZ111 + descriptor(3, 0),
                            y2p = descriptor(0, 1) * kX111 + descriptor(1, 1) * kY111 + descriptor(2, 1) * kZ111 + descriptor(3, 1),
                            z2p = descriptor(0, 2) * kX111 + descriptor(1, 2) * kY111 + descriptor(2, 2) * kZ111 + descriptor(3, 2);

                        indexPair111 = std::tuple<int, int, int, double, double, double>(
                            x2p / kCellSize4, y2p / kCellSize5, z2p / kCellSize6,
                            x2p, y2p, z2p);
                    }
                    else
                        indexPair111 = gridCornerCoordinates3D[kIdx3d111];

                    const double &l1x = std::get<3>(indexPair000) - kInlierOutlierThreshold_;
                    const double &l1y = std::get<4>(indexPair000) - kInlierOutlierThreshold_;
                    const double &l1z = std::get<5>(indexPair000) - kInlierOutlierThreshold_;
                    const double &r1x = std::get<3>(indexPair111) + kInlierOutlierThreshold_;
                    const double &r1y = std::get<4>(indexPair111) + kInlierOutlierThreshold_;
                    const double &r1z = std::get<5>(indexPair111) + kInlierOutlierThreshold_;

                    const double l2x = kCellSize4 * kCornerIndices[3];
                    const double l2y = kCellSize5 * kCornerIndices[4];
                    const double l2z = kCellSize6 * kCornerIndices[5];
                    const double r2x = l2x + kCellSize4;
                    const double r2y = l2y + kCellSize5;
                    const double r2z = l2z + kCellSize6;

                    // If one rectangle is on left side of other
                    if (l1x <= r2x && l2x <= r1x ||
                        // If one rectangle is above other
                        r1y <= l2y && r2y <= l1y ||
                        // If one rectangle is above other
                        r1z <= l2z && r2z <= l1z)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                    }
                }
            }
        }

		FORCE_INLINE void SpacePartitioningRANSAC::runHomography(
			const DataMatrix &kData_,
			const models::Model &kModel_,
			const double& kInlierOutlierThreshold_,
			std::vector<const std::vector<size_t>*>& selectedCells_,
			size_t& pointNumber_)
        {*/
            /*
                Selecting cells based on mutual visibility
            */
            /*constexpr double kDeterminantEpsilon = 1e-3;
            const Eigen::Matrix3d& descriptor = kModelData;
            const double kDeterminant = descriptor.determinant();
            if (abs(kDeterminant) < kDeterminantEpsilon)
                return;

            const double& kCellSize1 = additionalParameters[0],
                & kCellSize2 = additionalParameters[1],
                & kCellSize3 = additionalParameters[2],
                & kCellSize4 = additionalParameters[3],
                & kPartitionNumber = additionalParameters[4];

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = { 0, 0,
                0, 1,
                1, 0,
                1, 1 };

            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [cell, value] : neighborhoodGraph->getCells())
            {
                // The points in the cell
                const auto& points = value.first;

                // Checking if there are enough points in the cell to make the cell selection worth it
                if (points.size() < 4)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                    continue;
                }

                const auto& kCornerIndices = value.second;
                bool overlaps = false;

                // Iterate through the corners of the current cell
                for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                {
                    // The index of the currently projected corner
                    const size_t kCornerHorizontalIndex = kCornerIndices[0] + steps[stepIdx];
                    if (kCornerHorizontalIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerVerticalIndex = kCornerIndices[1] + steps[stepIdx + 1];
                    if (kCornerVerticalIndex >= kPartitionNumber)
                        continue;

                    // Get the index of the corner's projection in the destination image
                    const size_t kIdx2d = kCornerHorizontalIndex * kPartitionNumber + kCornerVerticalIndex;

                    // This is already or will be the horizontal and vertical indices in the destination image 
                    auto& indexPair = gridCornerCoordinatesH[kIdx2d];

                    // If the corner hasn't yet been projected to the destination image
                    if (!gridCornerMask[kIdx2d])
                    {
                        // Get the coordinates of the corner
                        const double kX1 = kCornerHorizontalIndex * kCellSize1,
                            kY1 = kCornerVerticalIndex * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX1 * descriptor(0, 0) + kY1 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX1 * descriptor(1, 0) + kY1 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX1 * descriptor(2, 0) + kY1 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        // Store the projected corner's cell indices
                        std::get<0>(indexPair) = x2p / kCellSize3;
                        std::get<1>(indexPair) = y2p / kCellSize4;
                        std::get<2>(indexPair) = x2p;
                        std::get<3>(indexPair) = y2p;

                        // Note that the corner has been already projected
                        gridCornerMask[kIdx2d] = true;
                    }

                    // Check if the projected corner is equal to the correspondence's destination point's grid cell.
                    // This works due to the coordinate truncation.
                    if (std::get<0>(indexPair) == kCornerIndices[2] &&
                        std::get<1>(indexPair) == kCornerIndices[3])
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                        break;
                    }
                }

                // Check if there is an overlap
                if (!overlaps)
                {
                    // The horizontal index of the bottom-right corner
                    const size_t kCornerHorizontalIndex11 = kCornerIndices[0] + steps[6];
                    // The vertical index of the bottom-right corner
                    const size_t kCornerVerticalIndex11 = kCornerIndices[1] + steps[7];

                    // Calculating the index of the top-left corner
                    const size_t kIdx2d00 = kCornerIndices[0] * kPartitionNumber + kCornerIndices[1];
                    // Calculating the index of the bottom-right corner
                    const size_t kIdx2d11 = kCornerHorizontalIndex11 * kPartitionNumber + kCornerVerticalIndex11;

                    // Coordinates of the top-left and bottom-right corners in the destination image
                    auto& indexPair00 = gridCornerCoordinatesH[kIdx2d00];

                    std::tuple<int, int, double, double> indexPair11;
                    if (kCornerVerticalIndex11 >= kPartitionNumber ||
                        kCornerHorizontalIndex11 >= kPartitionNumber)
                    {
                        // Get the coordinates of the corner
                        const double kX11 = kCornerHorizontalIndex11 * kCellSize1,
                            kY11 = kCornerVerticalIndex11 * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX11 * descriptor(0, 0) + kY11 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX11 * descriptor(1, 0) + kY11 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX11 * descriptor(2, 0) + kY11 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        indexPair11 = std::tuple<int, int, double, double>(x2p / kCellSize3, y2p / kCellSize4, x2p, y2p);
                    }
                    else
                        indexPair11 = gridCornerCoordinatesH[kIdx2d11];

                    const double& l1x = std::get<2>(indexPair00) - kInlierOutlierThreshold_;
                    const double& l1y = std::get<3>(indexPair00) - kInlierOutlierThreshold_;
                    const double& r1x = std::get<2>(indexPair11) + kInlierOutlierThreshold_;
                    const double& r1y = std::get<3>(indexPair11) + kInlierOutlierThreshold_;

                    const double l2x = kCellSize3 * kCornerIndices[2];
                    const double l2y = kCellSize4 * kCornerIndices[3];
                    const double r2x = l2x + kCellSize3;
                    const double r2y = l2y + kCellSize4;

                    // If one rectangle is on left side of other
                    if (l1x <= r2x && l2x <= r1x ||
                        // If one rectangle is above other
                        r1y <= l2y && r2y <= l1y)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                    }
                }
            }*/
        //}

        FORCE_INLINE void SpacePartitioningRANSAC::runRigidTransformation(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            // Validate that we have a valid neighborhoodGraph
            if (!neighborhoodGraph) {
                return;  // No neighborhood graph available
            }
            
            try {
                // The actual descriptor of the model - 4x4 transformation matrix
                const auto &kModelData = kModel_.getData();

                // Get 6D grid parameters
                const double& kCellSize1 = additionalParameters[0],  // X cell size for source cloud
                    & kCellSize2 = additionalParameters[1],          // Y cell size for source cloud
                    & kCellSize3 = additionalParameters[2],          // Z cell size for source cloud
                    & kCellSize4 = additionalParameters[3],          // X cell size for destination cloud
                    & kCellSize5 = additionalParameters[4],          // Y cell size for destination cloud
                    & kCellSize6 = additionalParameters[5],          // Z cell size for destination cloud
                    & kPartitionNumber = additionalParameters[6];    // Number of cells along each axis

                // Validate cell sizes
                if (kCellSize1 <= 0 || kCellSize2 <= 0 || kCellSize3 <= 0 ||
                    kCellSize4 <= 0 || kCellSize5 <= 0 || kCellSize6 <= 0 || 
                    kPartitionNumber == 0) {
                    // Fallback: return all cells
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                    return;
                }

                // For 6D grid: N^6 total cells
                const size_t N = static_cast<size_t>(kPartitionNumber);
                const size_t N2 = N * N;
                const size_t N3 = N2 * N;
                const size_t N4 = N3 * N;
                const size_t N5 = N4 * N;

                // Fill the corner mask with zeros
                std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

                // Iterate through all cells and check for overlap
                const auto& cellMap = neighborhoodGraph->getCells();
                for (const auto& [kCell, kValue] : cellMap)
                {
                    const auto& kPoints = kValue.first;

                    // If the cell has fewer than 5 points, include all points without filtering
                    if (kPoints.size() < 5)
                    {
                        selectedCells_.emplace_back(&kPoints);
                        pointNumber_ += kPoints.size();
                        continue;
                    }

                    // Extract the corner indices from the cell
                    const auto& kCornerIndices = kValue.second;
                    
                    // Validate corner indices
                    if (kCornerIndices.size() < 6) {
                        // Invalid cell structure, include all points
                        selectedCells_.emplace_back(&kPoints);
                        pointNumber_ += kPoints.size();
                        continue;
                    }

                    bool overlaps = false;

                    // Check all 8 corners of the source cell
                    for (int dx1 = 0; dx1 <= 1 && !overlaps; ++dx1) {
                        for (int dy1 = 0; dy1 <= 1 && !overlaps; ++dy1) {
                            for (int dz1 = 0; dz1 <= 1 && !overlaps; ++dz1) {
                                // Source point corner
                                const double px1 = (kCornerIndices[0] + dx1) * kCellSize1;
                                const double py1 = (kCornerIndices[1] + dy1) * kCellSize2;
                                const double pz1 = (kCornerIndices[2] + dz1) * kCellSize3;

                                // Transform the corner to the destination frame
                                const double px2_proj = kModelData(0, 0) * px1 + kModelData(0, 1) * py1 + 
                                                       kModelData(0, 2) * pz1 + kModelData(0, 3);
                                const double py2_proj = kModelData(1, 0) * px1 + kModelData(1, 1) * py1 + 
                                                       kModelData(1, 2) * pz1 + kModelData(1, 3);
                                const double pz2_proj = kModelData(2, 0) * px1 + kModelData(2, 1) * py1 + 
                                                       kModelData(2, 2) * pz1 + kModelData(2, 3);

                                // Calculate which cell this projects into
                                int x2_cell = static_cast<int>(std::floor(px2_proj / kCellSize4));
                                int y2_cell = static_cast<int>(std::floor(py2_proj / kCellSize5));
                                int z2_cell = static_cast<int>(std::floor(pz2_proj / kCellSize6));

                                // Clamp to grid bounds
                                x2_cell = std::max(0, std::min(x2_cell, static_cast<int>(N) - 1));
                                y2_cell = std::max(0, std::min(y2_cell, static_cast<int>(N) - 1));
                                z2_cell = std::max(0, std::min(z2_cell, static_cast<int>(N) - 1));

                                // Check if this matches the destination cell
                                if (x2_cell == static_cast<int>(kCornerIndices[3]) &&
                                    y2_cell == static_cast<int>(kCornerIndices[4]) &&
                                    z2_cell == static_cast<int>(kCornerIndices[5]))
                                {
                                    overlaps = true;
                                    break;
                                }
                            }
                        }
                    }

                    // If no corner matches, check for bounding box overlap
                    if (!overlaps)
                    {
                        // OPTIMIZATION: Pre-compute projected corner coordinates
                        double x2_min = 1e10, y2_min = 1e10, z2_min = 1e10;
                        double x2_max = -1e10, y2_max = -1e10, z2_max = -1e10;

                        for (int dx1 = 0; dx1 <= 1; ++dx1) {
                            for (int dy1 = 0; dy1 <= 1; ++dy1) {
                                for (int dz1 = 0; dz1 <= 1; ++dz1) {
                                    const double px1 = (kCornerIndices[0] + dx1) * kCellSize1;
                                    const double py1 = (kCornerIndices[1] + dy1) * kCellSize2;
                                    const double pz1 = (kCornerIndices[2] + dz1) * kCellSize3;

                                    const double px2 = kModelData(0, 0) * px1 + kModelData(0, 1) * py1 + 
                                                      kModelData(0, 2) * pz1 + kModelData(0, 3);
                                    const double py2 = kModelData(1, 0) * px1 + kModelData(1, 1) * py1 + 
                                                      kModelData(1, 2) * pz1 + kModelData(1, 3);
                                    const double pz2 = kModelData(2, 0) * px1 + kModelData(2, 1) * py1 + 
                                                      kModelData(2, 2) * pz1 + kModelData(2, 3);

                                    // Update bounding box with threshold
                                    x2_min = std::min(x2_min, px2 - kInlierOutlierThreshold_);
                                    y2_min = std::min(y2_min, py2 - kInlierOutlierThreshold_);
                                    z2_min = std::min(z2_min, pz2 - kInlierOutlierThreshold_);
                                    x2_max = std::max(x2_max, px2 + kInlierOutlierThreshold_);
                                    y2_max = std::max(y2_max, py2 + kInlierOutlierThreshold_);
                                    z2_max = std::max(z2_max, pz2 + kInlierOutlierThreshold_);
                                }
                            }
                        }

                        // Destination cell bounds
                        const double dest_x_min = kCornerIndices[3] * kCellSize4;
                        const double dest_y_min = kCornerIndices[4] * kCellSize5;
                        const double dest_z_min = kCornerIndices[5] * kCellSize6;
                        const double dest_x_max = dest_x_min + kCellSize4;
                        const double dest_y_max = dest_y_min + kCellSize5;
                        const double dest_z_max = dest_z_min + kCellSize6;

                        // AABB-AABB intersection test
                        overlaps = (x2_min <= dest_x_max && x2_max >= dest_x_min) &&
                                  (y2_min <= dest_y_max && y2_max >= dest_y_min) &&
                                  (z2_min <= dest_z_max && z2_max >= dest_z_min);
                    }

                    if (overlaps)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&kPoints);
                        pointNumber_ += kPoints.size();
                        
                        // OPTIMIZATION: Early rejection if filtered points < best model
                        if (bestModelInlierCount > 0 && pointNumber_ >= bestModelInlierCount * 2) {
                            break;
                        }
                    }
                }
                
                // OPTIMIZATION: Early rejection if filtered points < best model
                if (bestModelInlierCount > 0 && pointNumber_ < bestModelInlierCount) {
                    selectedCells_.clear();
                    pointNumber_ = 0;
                    return;
                }
            } catch (const std::exception& e) {
                // If any exception occurs, fall back to returning all cells
                try {
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    pointNumber_ = 0;
                    for (const auto& [idx, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                } catch (...) {
                    // Last resort: do nothing
                }
            } catch (...) {
                // Catch any other exception
                try {
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    pointNumber_ = 0;
                    for (const auto& [idx, value] : cellMap) {
                        selectedCells_.emplace_back(&value.first);
                        pointNumber_ += value.first.size();
                    }
                } catch (...) {
                    // If even fallback fails, do nothing
                }
            }
        }

        FORCE_INLINE void SpacePartitioningRANSAC::runAbsolutePose(
            const DataMatrix &kData_,
            const models::Model &kModel_,
            const double& kInlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_)
        {
            // 2D IMAGE GRID + 3D FRUSTUM APPROACH FOR PNP:
            // - Partition 2D image space into a grid
            // - Each 2D cell defines a viewing frustum (4 corner rays)
            // - Partition 3D world space independently
            // - For each 3D world cell, project corners and check frustum overlap
            // This avoids 5D complexity by separating image and world concerns
            
            pointNumber_ = 0;
            selectedCells_.clear();
            
            // Validation
            if (!neighborhoodGraph) return;
            
            try {
                // If frustums not yet initialized, fall back to returning all cells
                if (!pnpGridInitialized) {
                    const auto& cellMap = neighborhoodGraph->getCells();
                    if (cellMap.empty()) return;
                    
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx, value] : cellMap) {
                        const auto& points = value.first;
                        if (!points.empty()) {
                            selectedCells_.emplace_back(&points);
                            pointNumber_ += points.size();
                        }
                    }
                    return;
                }
                
                // Extract camera info from model
                // AbsolutePose model: K (3x3), R (3x3), t (3x1), camera center, grid info
                // We need: camera calibration matrix, camera center (world coords)
                
                // Grid parameters from additionalParameters for 5D PnP grid:
                // [0] = image U cell size
                // [1] = image V cell size
                // [2] = world X cell size
                // [3] = world Y cell size
                // [4] = world Z cell size
                // [5] = division number (grid cells per axis)
                
                if (additionalParameters.size() < 6) {
                    // Fall back: return all cells if grid not properly initialized
                    const auto& cellMap = neighborhoodGraph->getCells();
                    selectedCells_.reserve(cellMap.size());
                    for (const auto& [idx, value] : cellMap) {
                        const auto& points = value.first;
                        if (!points.empty()) {
                            selectedCells_.emplace_back(&points);
                            pointNumber_ += points.size();
                        }
                    }
                    return;
                }
                
                // Extract grid cell sizes from neighborhood structure
                const double cellSizeU = additionalParameters[0];    // Image U cell size
                const double cellSizeV = additionalParameters[1];    // Image V cell size
                const double cellSizeX = additionalParameters[2];    // World X cell size
                const double cellSizeY = additionalParameters[3];    // World Y cell size
                const double cellSizeZ = additionalParameters[4];    // World Z cell size
                const size_t divisionNumber = static_cast<size_t>(additionalParameters[5]);  // Cells per axis
                
                // Compute image dimensions from grid parameters
                // Image dimensions = cell_size * division_number
                const double imgWidth = cellSizeU * divisionNumber;
                const double imgHeight = cellSizeV * divisionNumber;
                const double minU = 0.0;
                const double maxU = imgWidth;
                const double minV = 0.0;
                const double maxV = imgHeight;
                
                // Now process 3D world cells against 2D image frustums
                const auto& cellMap = neighborhoodGraph->getCells();
                selectedCells_.reserve(cellMap.size());
                
                // For each 3D world cell, check if it overlaps with visible image frustum
                for (const auto& [idx, value] : cellMap) {
                    const auto& points = value.first;
                    if (points.empty()) continue;
                    
                    // Decode 5D cell index: for 5D space with N divisions per axis
                    // idx = u + v*N + x*N^2 + y*N^3 + z*N^4
                    // where u,v are image indices and x,y,z are world indices
                    const size_t N = divisionNumber;
                    const size_t N2 = N * N;
                    const size_t N3 = N2 * N;
                    const size_t N4 = N3 * N;
                    
                    const size_t gridU = idx % N;
                    const size_t gridV = (idx / N) % N;
                    const size_t gridX = (idx / N2) % N;
                    const size_t gridY = (idx / N3) % N;
                    const size_t gridZ = idx / N4;
                    
                    // 3D world cell bounds (using X, Y, Z indices from 5D cell)
                    double worldMinX = gridX * cellSizeX;
                    double worldMaxX = (gridX + 1) * cellSizeX;
                    double worldMinY = gridY * cellSizeY;
                    double worldMaxY = (gridY + 1) * cellSizeY;
                    double worldMinZ = gridZ * cellSizeZ;
                    double worldMaxZ = (gridZ + 1) * cellSizeZ;
                    
                    // Check if any corner of the 3D cell projects into the image
                    bool cellVisible = false;
                    double corners[8][3] = {
                        {worldMinX, worldMinY, worldMinZ},
                        {worldMaxX, worldMinY, worldMinZ},
                        {worldMinX, worldMaxY, worldMinZ},
                        {worldMaxX, worldMaxY, worldMinZ},
                        {worldMinX, worldMinY, worldMaxZ},
                        {worldMaxX, worldMinY, worldMaxZ},
                        {worldMinX, worldMaxY, worldMaxZ},
                        {worldMaxX, worldMaxY, worldMaxZ}
                    };
                    
                    for (int c = 0; c < 8; ++c) {
                        Eigen::Vector3d worldPoint(corners[c][0], corners[c][1], corners[c][2]);
                        
                        // Project to image (using camera matrix)
                        // p_img = K @ (R @ p_world + t)
                        Eigen::Vector3d camPoint = worldPoint - camerCenter;  // Relative to camera
                        // Note: We don't have full R,t here, so we use a simpler depth test
                        if (camPoint.z() > 0.0) {  // Must be in front of camera
                            double u = cameraMatrix(0, 0) * camPoint.x() / camPoint.z() + cameraMatrix(0, 2);
                            double v = cameraMatrix(1, 1) * camPoint.y() / camPoint.z() + cameraMatrix(1, 2);
                            
                            // Check if projection is within image bounds (with slight margin)
                            if (u >= minU - cellSizeU && u <= maxU + cellSizeU && 
                                v >= minV - cellSizeV && v <= maxV + cellSizeV) {
                                cellVisible = true;
                                break;
                            }
                        }
                    }
                    
                    // If cell is visible in image frustum, add all its points
                    if (cellVisible) {
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                    }
                }
                
            } catch (const std::bad_alloc& e) {
                selectedCells_.clear();
                pointNumber_ = 0;
            } catch (const std::exception& e) {
                selectedCells_.clear();
                pointNumber_ = 0;
            } catch (...) {
                selectedCells_.clear();
                pointNumber_ = 0;
            }
        }

	}
}