#include <Eigen/Dense>

#include "superansac.h"

#include "estimators/solver_homography_four_point.h"
#include "estimators/estimator_homography.h"

#include "estimators/solver_essential_matrix_five_point_nister.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/estimator_essential_matrix.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_fundamental_matrix_bundle_adjustment.h"
#include "estimators/estimator_fundamental_matrix.h"

#include "estimators/solver_radial_fundamental_matrix_nine_point.h"
#include "estimators/estimator_radial_fundamental_matrix.h"

#include "estimators/solver_focal_fundamental_matrix_seven_point.h"
#include "estimators/solver_focal_fundamental_matrix_rt_lm.h"
#include "estimators/estimator_focal_fundamental_matrix.h"

#include "estimators/numerical_optimizer/essential.h"

#include "estimators/solver_rigid_transform_proscrutes.h"
#include "estimators/estimator_rigid_transformation.h"

#include "estimators/solver_p3p_lambda_twist.h"
#include "estimators/solver_pnp_bundle_adjustment.h"
#include "estimators/estimator_absolute_pose.h"

#include "estimators/numerical_optimizer/types.h"
#include "estimators/numerical_optimizer/bundle.h"
#include "estimators/numerical_optimizer/jacobian_impl.h"

#include "superansac.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "neighborhood/types.h"
#include "inlier_selectors/types.h"
#include "utils/types.h"
#include "utils/utils_point_correspondence.h"
#include "camera/types.h"

// Function to initialize the neighborhood graph
template <size_t _DimensionNumber>
void initializeNeighborhood(
    const DataMatrix& kCorrespondences_, // The point correspondences
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> &neighborhoodGraph_, // The neighborhood graph
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const superansac::RANSACSettings &kSettings_) // The RANSAC settings
{
    // Create the neighborhood graph
    neighborhoodGraph_ = superansac::neighborhood::createNeighborhoodGraph<_DimensionNumber>(kNeighborhoodType_);
    // Initialize the neighborhood graph if the neighborhood is grid
    if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::Grid) 
    {
        // Check if the image sizes have the correct number of elements
        if (kImageSizes_.size() != _DimensionNumber)
            throw std::invalid_argument("The image sizes must have " + std::to_string(_DimensionNumber) + " elements.");

        // Cast the neighborhood graph to the grid neighborhood graph
        auto gridNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<_DimensionNumber> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        const auto &kCellNumber = kSettings_.neighborhoodSettings.neighborhoodGridDensity;
        std::vector<double> kCellSizes(_DimensionNumber);
        for (size_t i = 0; i < _DimensionNumber; i++)
        {
            kCellSizes[i] = kImageSizes_[i] / kCellNumber;
            if (kCellSizes[i] < 1.0)
                throw std::invalid_argument("The cell size is too small (< 1px). Try setting a smaller neighborhood size (in grid it acts as the number of cells along an axis).");
        }

        gridNeighborhoodGraph->setCellSizes(
            kCellSizes, // The sizes of the cells in each dimension
            kCellNumber); // The number of cells in each dimension
    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_KNN)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 0> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setNearestNeighborNumber(kSettings_.neighborhoodSettings.nearestNeighborNumber); 

    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_Radius)
    {
        // Cast the neighborhood graph to the FLANN neighborhood graph
        auto flannNeighborhoodGraph = 
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 1> *>(neighborhoodGraph_.get());
        // Initialize the neighborhood graph
        flannNeighborhoodGraph->setRadius(kSettings_.neighborhoodSettings.neighborhoodSize); 

    }
    neighborhoodGraph_->initialize(&kCorrespondences_);
}

template <size_t _DimensionNumber>
void initializeLocalOptimizer(
    const DataMatrix& kCorrespondences_, // The point correspondences
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> &localOptimizer_,
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> &neighborhoodGraph_, // The neighborhood graph
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_, // The type of the neighborhood
    const superansac::local_optimization::LocalOptimizationType kLocalOptimizationType_, // The type of the neighborhood
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    const superansac::RANSACSettings &kSettings_, // The RANSAC settings
    const superansac::LocalOptimizationSettings &kLOSettings_, // The RANSAC settings
    const superansac::models::Types &kModelType_,
    const bool kFinalOptimization_ = false) 
{
    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::None)
        return;

    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::GCRANSAC)
    {
        // Initialize the neighborhood graph if needed
        if (neighborhoodGraph_ == nullptr)
            initializeNeighborhood<_DimensionNumber>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph_, // The neighborhood graph
                kNeighborhoodType_, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                kSettings_); // The RANSAC settings
        // Set the neighborhood graph to the local optimizer
        auto gcransacLocalOptimizer = dynamic_cast<superansac::local_optimization::GraphCutRANSACOptimizer *>(localOptimizer_.get());
        gcransacLocalOptimizer->setNeighborhood(neighborhoodGraph_.get());
        gcransacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);
        gcransacLocalOptimizer->setGraphCutNumber(kLOSettings_.graphCutNumber);
        gcransacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
        gcransacLocalOptimizer->setSpatialCoherenceWeight(kLOSettings_.spatialCoherenceWeight);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IRLS)
    {
        // Set the neighborhood graph to the local optimizer
        auto irlsLocalOptimizer = dynamic_cast<superansac::local_optimization::IRLSOptimizer *>(localOptimizer_.get());
        irlsLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);       
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            irlsLocalOptimizer->setUseInliers(true); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::LSQ)
    {
        // Set the neighborhood graph to the local optimizer
        auto lsqLocalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            lsqLocalOptimizer->setUseInliers(true);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::NestedRANSAC)
    {
        // Set the neighborhood graph to the local optimizer
        auto nestedRansacLocalOptimizer = dynamic_cast<superansac::local_optimization::NestedRANSACOptimizer *>(localOptimizer_.get());
        nestedRansacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations); 
        nestedRansacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IteratedLMEDS)
    {
        auto iteratedLMEDSLocalOptimizer = dynamic_cast<superansac::local_optimization::IteratedLMEDSOptimizer *>(localOptimizer_.get());
        iteratedLMEDSLocalOptimizer->setModelType(kModelType_); 
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::CrossValidation)
    {
        auto crossValidationLocalOptimizer = dynamic_cast<superansac::local_optimization::CrossValidationOptimizer *>(localOptimizer_.get());
        if (kFinalOptimization_ || 
            kModelType_ == superansac::models::Types::Homography || 
            kModelType_ == superansac::models::Types::RigidTransformation || 
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            crossValidationLocalOptimizer->setUseInliers(true);
        
        //crossValidationLocalOptimizer->setRepetitions(kLOSettings_.maxIterations);
        crossValidationLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
    } 
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<size_t>, double, size_t> estimateAbsolutePose(
    const DataMatrix& kCorrespondences_, // The 2D-3D point correspondences
    const superansac::camera::CameraType &kCameraType_, // The type of the camera 
    const std::vector<double>& kCameraParams_, // The camera parameters
    const std::vector<double>& kBoundingBox_, // The bounding box dimensions (image width, image height, X, Y, Z)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 5) 
        throw std::invalid_argument("The input matrix must have 5 columns (x1, y1, x2, y2, z2).");
    if (kCorrespondences_.rows() < 3) 
        throw std::invalid_argument("The input matrix must have at least 3 rows.");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    std::unique_ptr<superansac::camera::AbstractCamera> currentCamera =
        superansac::camera::createCamera(kCameraType_, kCameraParams_);
    std::unique_ptr<superansac::camera::AbstractCamera> onlyScaledCamera =
        superansac::camera::createIdentityCamera(kCameraType_);
    onlyScaledCamera->rescale(currentCamera->focalLength());

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    currentCamera->fromPixelToImageCoordinates(kCorrespondences_, normalizedCorrespondences);
        
    // Normalize the threshold
    settings_.inlierThreshold = 
        currentCamera->normalizeThreshold(settings_.inlierThreshold);

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::AbsolutePoseEstimator> estimator = 
        std::unique_ptr<superansac::estimator::AbsolutePoseEstimator>(new superansac::estimator::AbsolutePoseEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::P3PLambdaTwistSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::PnPBundleAdjustmentSolver());
    dynamic_cast<superansac::estimator::solver::PnPBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->setCamera(onlyScaledCamera.get());
    auto &solverOptions = dynamic_cast<superansac::estimator::solver::PnPBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver())->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<5>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<5> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kBoundingBox_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<5>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kBoundingBox_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<5>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    } else if (kScoring == superansac::scoring::ScoringType::ACRANSAC) // Initialize the scoring object if the scoring is ACRANSAC
        throw std::invalid_argument("The ACRANSAC scoring is not implemented for the absolute pose estimation.");

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector (including SpacePartitioning with 2D+3D frustum support for AbsolutePose)
        inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
        
        // For AbsolutePose with SpacePartitioning, initialize the 2D image grid + 3D frustum approach
        if (settings_.inlierSelector == superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC)
        {
            // Extract camera matrix from camera object
            // For SimplePinhole: camera parameters are [f, cx, cy]
            // We'll set up camera matrix properly during frustum initialization
            
            // Initialize the SpacePartitioningRANSAC with 2D+3D frustum support
            // This happens in the inlier selector's initialize() method, which gets camera data
            // from the neighborhood graph that we'll set up next
        }
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<5>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kBoundingBox_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::AbsolutePose,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<5>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kBoundingBox_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::AbsolutePose,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // OPTIMIZATION: Set minimum inlier threshold for early rejection
    if (inlierSelector) {
        auto *spSelector = dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        if (spSelector) {
            spSelector->setBestModelInlierCount(estimator->sampleSize());
        }
    }
    
    // Set the inlier selector if created
    if (inlierSelector)
        robustEstimator.setInlierSelector(inlierSelector.get());
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d rotation = robustEstimator.getBestModel().getData().block<3, 3>(0, 0).eval();
    Eigen::Vector3d translation = robustEstimator.getBestModel().getData().block<3, 1>(0, 3).eval();

    // Return the best model with the inliers and the score
    return std::make_tuple(rotation, 
        translation,
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

std::tuple<Eigen::Matrix4d, std::vector<size_t>, double, size_t> estimateRigidTransform(
    const DataMatrix& kCorrespondences_, // The 3D-3D point correspondences
    const std::vector<double>& kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 6) 
        throw std::invalid_argument("The input matrix must have 6 columns (x1, y1, z1, x2, y2, z2).");
    if (kCorrespondences_.rows() < 3) 
        throw std::invalid_argument("The input matrix must have at least 3 rows.");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    // Normalize the point correspondences
    /*DataMatrix normalizedCorrespondences;
    Eigen::Matrix4d normalizingTransformSource, normalizingTransformDestination;
    normalize3D3DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   

    std::cout << normalizingTransformSource << std::endl;
    std::cout << normalizingTransformDestination << std::endl;
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;*/

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::RigidTransformationEstimator> estimator = 
        std::unique_ptr<superansac::estimator::RigidTransformationEstimator>(new superansac::estimator::RigidTransformationEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<6>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<6> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kBoundingBoxSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<6>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<6>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<6> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<6>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kBoundingBoxSizes_, // Bounding box sizes (x1, y1, z1, x2, y2, z2)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<6> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
    } else if (kScoring == superansac::scoring::ScoringType::ACRANSAC) // Initialize the scoring object if the scoring is ACRANSAC
        throw std::invalid_argument("The scoring type is not supported.");

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
        
        // If space partitioning is selected, initialize it for rigid transformation
        if (settings_.inlierSelector == superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC)
        {
            if (kNeighborhood == superansac::neighborhood::NeighborhoodType::Grid && neighborhoodGraph)
            {
                superansac::inlier_selector::SpacePartitioningRANSAC *spacePartitioningRANSAC = 
                    reinterpret_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
                spacePartitioningRANSAC->initialize(
                    neighborhoodGraph.get(), 
                    superansac::models::Types::RigidTransformation);
            }
        }
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<6>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kBoundingBoxSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::RigidTransformation,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<6>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kBoundingBoxSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::RigidTransformation,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // OPTIMIZATION: Set minimum inlier threshold for early rejection
    if (inlierSelector) {
        auto *spSelector = dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        if (spSelector) {
            spSelector->setBestModelInlierCount(estimator->sampleSize());
        }
    }
    
    // Set the inlier selector if created
    if (inlierSelector)
        robustEstimator.setInlierSelector(inlierSelector.get());
    
    // Run the robust estimator
    robustEstimator.run(kCorrespondences_);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix4d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix4d rigidTransform = robustEstimator.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    //rigidTransform = normalizingTransformDestination.transpose() * rigidTransform * normalizingTransformSource;

    // Return the best model with the inliers and the score
    return std::make_tuple(rigidTransform, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());    
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateFundamentalMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 8) 
        throw std::invalid_argument("The input matrix must have at least 8 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");
    if (kPointProbabilities_.size() > 0 && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have either the same number of elements as the number of correspondences or none.");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalize2D2DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator> estimator = 
        std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator>(new superansac::estimator::FundamentalMatrixEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::FundamentalMatrixSevenPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver());
    superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver * nonminimalSolver = 
        dynamic_cast<superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    if (kPointProbabilities_.size() > 0)
        nonminimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Create the space partitioning inlier selector object
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    { 
        // Check whether the scoring is RANSAC, MSAC, or MAGSAC++
        if (kScoring != superansac::scoring::ScoringType::RANSAC &&
            kScoring != superansac::scoring::ScoringType::MSAC &&
            kScoring != superansac::scoring::ScoringType::MAGSAC)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with RANSAC, MSAC, or MAGSAC++ scoring.");

        // Check if the neighborhood is grid
        if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid)
            throw std::invalid_argument("The space partitioning inlier selector can only be used with grid neighborhood.");

        // Initialize the neighborhood graph if needed
        if (neighborhoodGraph == nullptr)
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
                
        // Create the inlier selector
        inlierSelector = 
            superansac::inlier_selector::createInlierSelector(superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC);
        // Initialize the inlier selector
        superansac::inlier_selector::SpacePartitioningRANSAC *spacePartitioningRANSAC = 
            reinterpret_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        spacePartitioningRANSAC->initialize(
            neighborhoodGraph.get(), 
            superansac::models::Types::FundamentalMatrix);
        spacePartitioningRANSAC->setNormalizers(
            normalizingTransformSource(0, 0), normalizingTransformSource(0, 2), normalizingTransformSource(1, 2),
            normalizingTransformDestination(0, 0), normalizingTransformDestination(0, 2), normalizingTransformDestination(1, 2));
        // Set the inlier selector to the robust estimator
        robustEstimator.setInlierSelector(inlierSelector.get());
    }

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // OPTIMIZATION: Set minimum inlier threshold for early rejection
    if (inlierSelector) {
        auto *spSelector = dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        if (spSelector) {
            spSelector->setBestModelInlierCount(estimator->sampleSize());
        }
    }
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d fundamentalMatrix = robustEstimator.getBestModel().getData();

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Return the best model with the inliers and the score
    return std::make_tuple(fundamentalMatrix, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

// Function to estimate radial fundamental matrix with division distortion
std::tuple<Eigen::Matrix3d, std::vector<double>, std::vector<size_t>, double, size_t> estimateRadialFundamentalMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    std::cerr.flush();

    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 9) 
        throw std::invalid_argument("The input matrix must have at least 9 rows for radial fundamental matrix estimation.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalize2D2DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);   
        
    const double kScale = 
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the estimator (only uses minimal solver with 9-point)
    std::unique_ptr<superansac::estimator::RadialFundamentalMatrixEstimator> estimator = 
        std::unique_ptr<superansac::estimator::RadialFundamentalMatrixEstimator>(new superansac::estimator::RadialFundamentalMatrixEstimator());
    
    // Create the sampler (9 points for radial fundamental matrix solver)
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<9>(kSampler);
    
    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // Note: For 9-point solver, skip type-specific operations to avoid issues
    // The sampler is already created with proper template parameters
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
    {
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    }
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        // auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<9> *>(sampler.get());
        // pNapsacSampler->setSampleSize(estimator->sampleSize());
        // pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
        //     kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<9>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object (for 4D point correspondence data)
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion =
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;

    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set up local optimization if requested
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        localOptimizer = superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // For LSQ optimization, enable using inliers for nonminimal estimation
        if (kLocalOptimization == superansac::local_optimization::LocalOptimizationType::LSQ)
        {
            auto lsqLocalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer *>(localOptimizer.get());
            lsqLocalOptimizer->setUseInliers(true);
        }
        else if (kLocalOptimization == superansac::local_optimization::LocalOptimizationType::IRLS)
        {
            auto irlsLocalOptimizer = dynamic_cast<superansac::local_optimization::IRLSOptimizer *>(localOptimizer.get());
            irlsLocalOptimizer->setUseInliers(true);
        }

        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set up final optimization if requested
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        finalOptimizer = superansac::local_optimization::createLocalOptimizer(kFinalOptimization);

        // For LSQ optimization, enable using inliers for nonminimal estimation
        if (kFinalOptimization == superansac::local_optimization::LocalOptimizationType::LSQ)
        {
            auto lsqFinalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer *>(finalOptimizer.get());
            lsqFinalOptimizer->setUseInliers(true);
        }
        else if (kFinalOptimization == superansac::local_optimization::LocalOptimizationType::IRLS)
        {
            auto irlsFinalOptimizer = dynamic_cast<superansac::local_optimization::IRLSOptimizer *>(finalOptimizer.get());
            irlsFinalOptimizer->setUseInliers(true);
        }

        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    robustEstimator.setSettings(settings_);
    
    size_t sampleSize;
    try {
        sampleSize = estimator->sampleSize();
    } catch (const std::exception& e) {
        throw;
    }
    
    // Run the robust estimator
    try {
        robustEstimator.run(normalizedCorrespondences);
    } catch (const std::exception& e) {
        throw;
    } catch (...) {
        throw;
    }

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<double>(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix and distortion parameters
    Eigen::MatrixXd modelDescriptor = robustEstimator.getBestModel().getData();
    Eigen::Matrix3d fundamentalMatrix = modelDescriptor.block<3, 3>(0, 0);
    double lam1 = modelDescriptor(0, 3);
    double lam2 = modelDescriptor(1, 3);

    // Transform the estimated fundamental matrix back to the not normalized space
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Scale the lambda parameters back
    // If p_normalized = scale * p_centered, then lambda_centered = lambda_normalized * scale^2
    double scale1 = normalizingTransformSource(0, 0);
    double scale2 = normalizingTransformDestination(0, 0);
    lam1 = lam1 * (scale1 * scale1);
    lam2 = lam2 * (scale2 * scale2);

    std::vector<double> distortionParams = {lam1, lam2};

    // Return the best model with the inliers, distortion parameters, and the score
    return std::make_tuple(fundamentalMatrix, 
        distortionParams,
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

// Function to refine radial fundamental matrix using LM optimization on inlier points
std::tuple<Eigen::Matrix3d, std::vector<double>, double> refineRadialFundamentalMatrixLM(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const Eigen::Matrix3d& kInitialF_, // Initial fundamental matrix
    const std::vector<double>& kInitialDistortion_, // Initial distortion [lam1, lam2]
    const std::vector<size_t>& kInlierIndices_) // Indices of inlier points
{
    // Extract inlier points
    std::vector<Eigen::Vector2d> x1, x2;
    for (const size_t idx : kInlierIndices_)
    {
        x1.push_back(Eigen::Vector2d(kCorrespondences_(idx, 0), kCorrespondences_(idx, 1)));
        x2.push_back(Eigen::Vector2d(kCorrespondences_(idx, 2), kCorrespondences_(idx, 3)));
    }

    // Initialize parameters
    poselib::RadialFundamentalMatrixParams params(
        kInitialF_,
        kInitialDistortion_.size() > 0 ? kInitialDistortion_[0] : 0.0,
        kInitialDistortion_.size() > 1 ? kInitialDistortion_[1] : 0.0);

    // Configure LM options
    poselib::BundleOptions lm_options;
    lm_options.max_iterations = 100;
    lm_options.loss_type = poselib::BundleOptions::LossType::CAUCHY;
    lm_options.loss_scale = 0.5;

    // Run Levenberg-Marquardt optimization
    poselib::BundleStats stats = poselib::refine_radial_fundamental(x1, x2, &params, lm_options, std::vector<double>());

    // Return refined parameters
    std::vector<double> distortion{params.lam1, params.lam2};
    return std::make_tuple(params.F, distortion, stats.cost);
}

// Helper function to compute the Essential matrix from F and focal lengths
// E = diag(f2, f2, 1) * F * diag(f1, f1, 1)
inline Eigen::Matrix3d computeEssentialFromF(const Eigen::Matrix3d& F, double f1, double f2)
{
    Eigen::Matrix3d E = F;
    // Multiply rows 0,1 by f2
    E.row(0) *= f2;
    E.row(1) *= f2;
    // Multiply cols 0,1 by f1
    E.col(0) *= f1;
    E.col(1) *= f1;
    return E;
}

// Compute the essential matrix constraint error: |sigma1 - sigma2| / sigma1
// For a valid essential matrix, the two largest singular values should be equal
inline double essentialConstraintError(const Eigen::Matrix3d& F, double f1, double f2)
{
    Eigen::Matrix3d E = computeEssentialFromF(F, f1, f2);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
    Eigen::Vector3d s = svd.singularValues();
    if (s(0) < 1e-10) return 1e10;
    return std::abs(s(0) - s(1)) / s(0);
}

// Extract focal lengths from F by minimizing the essential matrix constraint
// Uses golden section search for efficiency
std::pair<double, double> extractFocalLengthsFromEssentialConstraint(
    const Eigen::Matrix3d& F, double f_init = 500.0)
{
    const double min_f = 50.0;
    const double max_f = 5000.0;
    const int grid_size = 20;
    const double tol = 1.0;  // 1 pixel tolerance

    // Phase 1: Coarse grid search
    double best_f1 = f_init, best_f2 = f_init;
    double best_err = essentialConstraintError(F, f_init, f_init);

    for (int i = 0; i < grid_size; ++i) {
        double f1 = min_f + (max_f - min_f) * i / (grid_size - 1);
        for (int j = 0; j < grid_size; ++j) {
            double f2 = min_f + (max_f - min_f) * j / (grid_size - 1);
            double err = essentialConstraintError(F, f1, f2);
            if (err < best_err) {
                best_err = err;
                best_f1 = f1;
                best_f2 = f2;
            }
        }
    }

    // Phase 2: Local refinement using coordinate descent
    double step = (max_f - min_f) / grid_size / 2;
    for (int iter = 0; iter < 10 && step > tol; ++iter) {
        // Refine f1
        for (double df = -step; df <= step; df += step/2) {
            double f1_new = std::max(min_f, std::min(max_f, best_f1 + df));
            double err = essentialConstraintError(F, f1_new, best_f2);
            if (err < best_err) {
                best_err = err;
                best_f1 = f1_new;
            }
        }
        // Refine f2
        for (double df = -step; df <= step; df += step/2) {
            double f2_new = std::max(min_f, std::min(max_f, best_f2 + df));
            double err = essentialConstraintError(F, best_f1, f2_new);
            if (err < best_err) {
                best_err = err;
                best_f2 = f2_new;
            }
        }
        step /= 2;
    }

    return {best_f1, best_f2};
}

// Function to extract focal lengths from a fundamental matrix
// Uses the essential matrix constraint: E = K2^T * F * K1 should have two equal singular values
std::tuple<Eigen::Matrix3d, std::vector<double>, double> refineFocalFundamentalMatrixLM(
    const DataMatrix& kCorrespondences_, // The point correspondences (centered at principal point)
    const Eigen::Matrix3d& kInitialF_, // Initial fundamental matrix
    const std::vector<double>& kInitialFocals_, // Initial focal lengths [f1, f2]
    const std::vector<size_t>& kInlierIndices_) // Indices of inlier points
{
    // NOTE: We do NOT refine F with poselib::refine_fundamental here because
    // that refinement optimizes pixel-space Sampson error which changes F's
    // structure in a way that makes focal length extraction unreliable.
    //
    // Instead, we work directly with the RANSAC-estimated F, which preserves
    // the epipolar geometry needed for focal length extraction.

    // Normalize F so that ||F|| = 1
    Eigen::Matrix3d F_normalized = kInitialF_ / kInitialF_.norm();

    // Extract focal lengths using essential matrix constraint
    // The constraint is that E = K2^T * F * K1 should have two equal singular values
    // We use optimization to find f1, f2 that minimize |sigma1 - sigma2|
    double f_init = kInitialFocals_.size() > 0 ? kInitialFocals_[0] : 500.0;
    auto [f1_est, f2_est] = extractFocalLengthsFromEssentialConstraint(F_normalized, f_init);

    // Compute the essential matrix error for reporting
    double essential_err = essentialConstraintError(F_normalized, f1_est, f2_est);

    // Return the normalized F with estimated focal lengths
    std::vector<double> focals{f1_est, f2_est};
    return std::make_tuple(F_normalized, focals, essential_err);
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateEssentialMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const Eigen::Matrix3d &kIntrinsicsSource_, // The intrinsic matrix of the source camera
    const Eigen::Matrix3d &kIntrinsicsDestination_, // The intrinsic matrix of the destination camera
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 6) 
        throw std::invalid_argument("The input matrix must have at least 6 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");
    if (kPointProbabilities_.size() > 0 && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have either the same number of elements as the number of correspondences or none.");

    // Normalize the point correspondences
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalizePointsByIntrinsics(
        kCorrespondences_,
        kIntrinsicsSource_,
        kIntrinsicsDestination_,
        normalizedCorrespondences);
        
    const double kScale = 
        0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));
    settings_.inlierThreshold /= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::EssentialMatrixEstimator> estimator = 
        std::unique_ptr<superansac::estimator::EssentialMatrixEstimator>(new superansac::estimator::EssentialMatrixEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::EssentialMatrixFivePointNisterSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver());
    superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver * nonminimalSolver = 
        dynamic_cast<superansac::estimator::solver::EssentialMatrixBundleAdjustmentSolver *>(estimator->getMutableNonMinimalSolver());
    if (kPointProbabilities_.size() > 0)
        nonminimalSolver->setWeights(&kPointProbabilities_);
    auto &solverOptions = nonminimalSolver->getMutableOptions();
    solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
    solverOptions.loss_scale = settings_.inlierThreshold;
    solverOptions.max_iterations = 25;

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        inlierSelector = 
            superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::EssentialMatrix,
            false);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::EssentialMatrix,
            true);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // OPTIMIZATION: Set minimum inlier threshold for early rejection
    if (inlierSelector) {
        auto *spSelector = dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        if (spSelector) {
            spSelector->setBestModelInlierCount(estimator->sampleSize());
        }
    }
    
    // Run the robust estimator
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the normalized fundamental matrix
    Eigen::Matrix3d essentialMatrix = robustEstimator.getBestModel().getData();

    // Return the best model with the inliers and the score
    return std::make_tuple(essentialMatrix, 
        robustEstimator.getInliers(), 
        robustEstimator.getBestScore().getValue(), 
        robustEstimator.getIterationNumber());
}

std::tuple<Eigen::Matrix3d, std::vector<size_t>, double, size_t> estimateHomography(
    const DataMatrix& kCorrespondences_, // The point correspondences
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4) 
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 4) 
        throw std::invalid_argument("The input matrix must have at least 4 rows.");
    if (kImageSizes_.size() != 4) 
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");
    if (settings_.sampler == superansac::samplers::SamplerType::ImportanceSampler && 
        kPointProbabilities_.size() != kCorrespondences_.rows())
        throw std::invalid_argument("The point probabilities must have the same number of elements as the number of correspondences when using the ImportanceSampler or the ARSampler.");

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Create the solvers and the estimator
    std::unique_ptr<superansac::estimator::HomographyEstimator> estimator = 
        std::unique_ptr<superansac::estimator::HomographyEstimator>(new superansac::estimator::HomographyEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::HomographyFourPointSolver());

    // Create the sampler
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler = 
        superansac::samplers::createSampler<4>(kSampler);    

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // If the sampler is PROSAC, set the sample size
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, 
            kImageSizes_);
    } else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_, // The point correspondences
            neighborhoodGraph, // The neighborhood graph
            kNeighborhood, // The type of the neighborhood
            kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
            settings_); // The RANSAC settings
        // Set the neighborhood graph to the sampler
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    } else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_); // Set the probabilities to the sampler
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_); // Initialize the sampler

    // Create the scoring object
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer = 
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold); // Set the threshold

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::GRID) // Set the neighborhood structure if the scoring is GRID
    {
        // Check whether the neighborhood graph is already initialized
        superansac::neighborhood::GridNeighborhoodGraph<4> *gridNeighborhoodGraph;
        if (neighborhoodGraph == nullptr)
            // Initialize the neighborhood graph
            initializeNeighborhood<4>(
                kCorrespondences_, // The point correspondences
                neighborhoodGraph, // The neighborhood graph
                kNeighborhood, // The type of the neighborhood
                kImageSizes_, // Image sizes (height source, width source, height destination, width destination)
                settings_); // The RANSAC settings
        else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) // Check whether the provided neighborhood type is grid
            throw std::invalid_argument("The neighborhood graph is already initialized, but the neighborhood type is not grid.");
        // Set the neighborhood graph
        dynamic_cast<superansac::scoring::GridScoring<4> *>(scorer.get())->setNeighborhood(gridNeighborhoodGraph);
    } else if (kScoring == superansac::scoring::ScoringType::MAGSAC) // Initialize the scoring object if the scoring is MAGSAC
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
    }    

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion = 
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create inlier selector object if needed
    std::unique_ptr<superansac::inlier_selector::AbstractInlierSelector> inlierSelector;
    if (settings_.inlierSelector != superansac::inlier_selector::InlierSelectorType::None)
    {
        // Create the inlier selector
        inlierSelector = superansac::inlier_selector::createInlierSelector(settings_.inlierSelector);
        
        // Space partitioning requires a grid neighborhood - initialize it if needed
        if (settings_.inlierSelector == superansac::inlier_selector::InlierSelectorType::SpacePartitioningRANSAC)
        {
            // Check if neighborhood is already initialized
            if (neighborhoodGraph == nullptr) {
                // Initialize the neighborhood graph as GRID for space partitioning
                initializeNeighborhood<4>(
                    kCorrespondences_, // The point correspondences
                    neighborhoodGraph, // The neighborhood graph
                    superansac::neighborhood::NeighborhoodType::Grid, // Force GRID neighborhood
                    kImageSizes_, // Image sizes
                    settings_); // The RANSAC settings
            } else if (kNeighborhood != superansac::neighborhood::NeighborhoodType::Grid) {
                throw std::invalid_argument("Space partitioning inlier selector requires Grid neighborhood type.");
            }
            
            // Initialize the inlier selector with the neighborhood graph
            dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get())->initialize(
                neighborhoodGraph.get(),
                superansac::models::Types::Homography);
        }
    }

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get()); // Set the estimator
    robustEstimator.setSampler(sampler.get()); // Set the sampler
    robustEstimator.setScoring(scorer.get()); // Set the scoring method
    robustEstimator.setTerminationCriterion(terminationCriterion.get()); // Set the termination criterion
    
    // Set the inlier selector if created
    if (inlierSelector)
        robustEstimator.setInlierSelector(inlierSelector.get());

    // Set the local optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the local optimizer
        localOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        // Initialize the local optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::Homography);
            
        // Set the local optimizer
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set the final optimization object
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        // Create the final optimizer
        finalOptimizer = 
            superansac::local_optimization::createLocalOptimizer(kFinalOptimization);
            
        // Initialize the final optimizer
        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::Homography);

        // Set the final optimizer
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    // Set the settings
    robustEstimator.setSettings(settings_);
    
    // OPTIMIZATION: Set minimum inlier threshold for early rejection
    // Models with fewer inliers than the estimator's sample size won't be useful anyway
    // This helps space-partitioning skip unpromising models early
    if (inlierSelector) {
        auto *spSelector = dynamic_cast<superansac::inlier_selector::SpacePartitioningRANSAC *>(inlierSelector.get());
        if (spSelector) {
            // Initialize with minimum sample size as threshold
            // This ensures early rejection kicks in once we find any valid model
            spSelector->setBestModelInlierCount(estimator->sampleSize());
        }
    }
    
    // Run the robust estimator
    robustEstimator.run(kCorrespondences_);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Return the best model with the inliers and the score
    return std::make_tuple(robustEstimator.getBestModel().getData(),
        robustEstimator.getInliers(),
        robustEstimator.getBestScore().getValue(),
        robustEstimator.getIterationNumber());
}

// Function to refine F and focal lengths using R+t parameterization
// This ensures F always corresponds to a valid essential matrix
std::tuple<Eigen::Matrix3d, std::vector<double>, double> refineFocalFundamentalMatrixRTLM(
    const DataMatrix& kCorrespondences_, // The point correspondences (centered at principal point)
    const Eigen::Matrix3d& kInitialF_, // Initial fundamental matrix
    const std::vector<double>& kInitialFocals_, // Initial focal lengths [f1, f2]
    const std::vector<size_t>& kInlierIndices_) // Indices of inlier points
{
    if (kInitialFocals_.size() != 2)
        throw std::invalid_argument("Initial focal lengths must have 2 elements [f1, f2].");
    if (kInlierIndices_.empty())
        throw std::invalid_argument("Inlier indices cannot be empty.");

    double f1_init = kInitialFocals_[0];
    double f2_init = kInitialFocals_[1];

    // Compute E from F and initial focal lengths
    // E = K2^T * F * K1
    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
    K1(0, 0) = f1_init;
    K1(1, 1) = f1_init;
    Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
    K2(0, 0) = f2_init;
    K2(1, 1) = f2_init;

    Eigen::Matrix3d E_init = K2.transpose() * kInitialF_ * K1;

    // Normalize E via SVD to ensure valid essential matrix
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double avg_sv = (svd.singularValues()(0) + svd.singularValues()(1)) / 2.0;
    Eigen::Matrix3d E_valid = svd.matrixU() * Eigen::Vector3d(avg_sv, avg_sv, 0.0).asDiagonal() * svd.matrixV().transpose();

    // Prepare normalized points for pose recovery
    std::vector<Eigen::Vector3d> x1_normalized, x2_normalized;
    x1_normalized.reserve(kInlierIndices_.size());
    x2_normalized.reserve(kInlierIndices_.size());

    for (size_t idx : kInlierIndices_)
    {
        if (idx >= static_cast<size_t>(kCorrespondences_.rows()))
            continue;
        x1_normalized.emplace_back(kCorrespondences_(idx, 0) / f1_init, kCorrespondences_(idx, 1) / f1_init, 1.0);
        x2_normalized.emplace_back(kCorrespondences_(idx, 2) / f2_init, kCorrespondences_(idx, 3) / f2_init, 1.0);
    }

    // Decompose E into R, t
    poselib::CameraPoseVector relative_poses;
    poselib::motion_from_essential(E_valid, x1_normalized, x2_normalized, &relative_poses);

    if (relative_poses.empty())
    {
        // Return original values if decomposition fails
        std::vector<double> focals{f1_init, f2_init};
        return std::make_tuple(kInitialF_, focals, -1.0);
    }

    poselib::CameraPose best_pose = relative_poses[0];

    // Prepare points for LM refinement
    std::vector<poselib::Point2D> points1, points2;
    points1.reserve(kInlierIndices_.size());
    points2.reserve(kInlierIndices_.size());

    for (size_t idx : kInlierIndices_)
    {
        if (idx >= static_cast<size_t>(kCorrespondences_.rows()))
            continue;
        points1.emplace_back(kCorrespondences_(idx, 0), kCorrespondences_(idx, 1));
        points2.emplace_back(kCorrespondences_(idx, 2), kCorrespondences_(idx, 3));
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
    poselib::BundleStats stats = poselib::refine_focal_relpose(
        points1, points2, &params, opts, std::vector<double>());

    // Extract refined F from params
    Eigen::Matrix3d F_refined = params.F();
    F_refined.normalize();

    std::vector<double> focals{params.f1, params.f2};
    return std::make_tuple(F_refined, focals, stats.cost);
}

// Function to estimate focal fundamental matrix using RANSAC
// Uses 7-point + Bougnoux minimal solver for focal length extraction
// Uses R+t parameterized LM non-minimal solver
std::tuple<Eigen::Matrix3d, std::vector<double>, std::vector<size_t>, double, size_t> estimateFocalFundamentalMatrix(
    const DataMatrix& kCorrespondences_, // The point correspondences (centered at principal point)
    const std::vector<double>& kPointProbabilities_, // The probabilities of the points being inliers
    const std::vector<double>& kImageSizes_, // Image sizes (width source, height source, width destination, height destination)
    superansac::RANSACSettings &settings_) // The RANSAC settings
{
    // Check if the input matrix has the correct dimensions
    if (kCorrespondences_.cols() != 4)
        throw std::invalid_argument("The input matrix must have 4 columns (x1, y1, x2, y2).");
    if (kCorrespondences_.rows() < 7)
        throw std::invalid_argument("The input matrix must have at least 7 rows for focal fundamental matrix estimation.");
    if (kImageSizes_.size() != 4)
        throw std::invalid_argument("The image sizes must have 4 elements (height source, width source, height destination, width destination).");

    // Normalize the point correspondences for numerical stability
    DataMatrix normalizedCorrespondences;
    Eigen::Matrix3d normalizingTransformSource, normalizingTransformDestination;
    normalize2D2DPointCorrespondences(
        kCorrespondences_,
        normalizedCorrespondences,
        normalizingTransformSource,
        normalizingTransformDestination);

    const double kScale =
        0.5 * (normalizingTransformSource(0, 0) + normalizingTransformDestination(0, 0));
    settings_.inlierThreshold *= kScale;

    // Get the values from the settings
    const superansac::scoring::ScoringType kScoring = settings_.scoring;
    const superansac::samplers::SamplerType kSampler = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLocalOptimization = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFinalOptimization = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerminationCriterion = settings_.terminationCriterion;

    // Use the standard FundamentalMatrixEstimator to ensure identical results
    std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator> estimator =
        std::unique_ptr<superansac::estimator::FundamentalMatrixEstimator>(new superansac::estimator::FundamentalMatrixEstimator());
    estimator->setMinimalSolver(new superansac::estimator::solver::FundamentalMatrixSevenPointSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver());

    // Configure the BA solver with truncated loss (same as standard F)
    superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver* nonminimalSolver =
        dynamic_cast<superansac::estimator::solver::FundamentalMatrixBundleAdjustmentSolver*>(estimator->getMutableNonMinimalSolver());
    if (nonminimalSolver != nullptr)
    {
        if (kPointProbabilities_.size() > 0)
            nonminimalSolver->setWeights(&kPointProbabilities_);
        auto& solverOptions = nonminimalSolver->getMutableOptions();
        solverOptions.loss_type = poselib::BundleOptions::LossType::TRUNCATED;
        solverOptions.loss_scale = settings_.inlierThreshold;
        solverOptions.max_iterations = 25;
    }

    // Create the sampler (4D data: x1, y1, x2, y2; sample size 7 is set separately)
    std::unique_ptr<superansac::samplers::AbstractSampler> sampler =
        superansac::samplers::createSampler<4>(kSampler);

    // Create the neighborhood object (if needed)
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    // Configure sampler based on type
    if (kSampler == superansac::samplers::SamplerType::PROSAC)
    {
        dynamic_cast<superansac::samplers::PROSACSampler *>(sampler.get())->setSampleSize(estimator->sampleSize());
    }
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto pNapsacSampler = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4> *>(sampler.get());
        pNapsacSampler->setSampleSize(estimator->sampleSize());
        pNapsacSampler->setLayerData({ 16, 8, 4, 2 }, kImageSizes_);
    }
    else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        // Initialize the neighborhood graph
        initializeNeighborhood<4>(
            kCorrespondences_,
            neighborhoodGraph,
            kNeighborhood,
            kImageSizes_,
            settings_);
        dynamic_cast<superansac::samplers::NAPSACSampler *>(sampler.get())->setNeighborhood(neighborhoodGraph.get());
    }
    else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler *>(sampler.get())->setProbabilities(kPointProbabilities_);
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler *>(sampler.get())->initialize(
            &kCorrespondences_,
            kPointProbabilities_,
            settings_.arSamplerSettings.estimatorVariance,
            settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_);

    // Create the scoring object (for 4D point correspondence data)
    std::unique_ptr<superansac::scoring::AbstractScoring> scorer =
        superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold);

    // Set the image sizes if the scoring is ACRANSAC
    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
       scorer->setImageSize(kImageSizes_[0], kImageSizes_[1], kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::MAGSAC)
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring *>(scorer.get())->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring *>(scorer.get())->initialize(estimator.get());
        // Update solver options for MAGSAC scoring (same as standard F)
        if (nonminimalSolver != nullptr)
        {
            auto& solverOptions = nonminimalSolver->getMutableOptions();
            solverOptions.loss_type = poselib::BundleOptions::LossType::MAGSACPlusPlus;
        }
    }

    // Create termination criterion object
    std::unique_ptr<superansac::termination::AbstractCriterion> terminationCriterion =
        superansac::termination::createTerminationCriterion(kTerminationCriterion);

    if (kTerminationCriterion == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion *>(terminationCriterion.get())->setConfidence(settings_.confidence);

    // Create the RANSAC object
    superansac::SupeRansac robustEstimator;

    robustEstimator.setEstimator(estimator.get());
    robustEstimator.setSampler(sampler.get());
    robustEstimator.setScoring(scorer.get());
    robustEstimator.setTerminationCriterion(terminationCriterion.get());

    // Set up local optimization if requested (same initialization as standard F)
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLocalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        localOptimizer = superansac::local_optimization::createLocalOptimizer(kLocalOptimization);

        initializeLocalOptimizer<4>(
            kCorrespondences_,
            localOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kLocalOptimization,
            kImageSizes_,
            settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            false);

        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Set up final optimization if requested (same initialization as standard F)
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFinalOptimization != superansac::local_optimization::LocalOptimizationType::None)
    {
        finalOptimizer = superansac::local_optimization::createLocalOptimizer(kFinalOptimization);

        initializeLocalOptimizer<4>(
            kCorrespondences_,
            finalOptimizer,
            neighborhoodGraph,
            kNeighborhood,
            kFinalOptimization,
            kImageSizes_,
            settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::FundamentalMatrix,
            true);

        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    robustEstimator.setSettings(settings_);

    // Run the robust estimator on normalized correspondences
    robustEstimator.run(normalizedCorrespondences);

    // Check if the model is valid
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(Eigen::Matrix3d::Identity(), std::vector<double>(), std::vector<size_t>(), 0.0, robustEstimator.getIterationNumber());

    // Get the fundamental matrix and focal lengths
    // Descriptor is 3x4: [F (3x3) | f1, f2, 0]
    Eigen::MatrixXd modelDescriptor = robustEstimator.getBestModel().getData();
    Eigen::Matrix3d fundamentalMatrix = modelDescriptor.block<3, 3>(0, 0);

    // De-normalize the fundamental matrix: F_orig = T2^T * F_norm * T1
    fundamentalMatrix = normalizingTransformDestination.transpose() * fundamentalMatrix * normalizingTransformSource;
    fundamentalMatrix.normalize();

    // Extract focal lengths from de-normalized F using Bougnoux formula
    // (The focal lengths from normalized F are not valid for original coordinates)
    double f1, f2;
    double maxCoord = kCorrespondences_.array().abs().maxCoeff();
    double fallbackFocal = std::max(300.0, maxCoord * 1.5);
    superansac::estimator::solver::FocalFundamentalMatrixSevenPointSolver::extractFocalLengths(
        fundamentalMatrix, f1, f2, fallbackFocal, true);

    std::vector<double> focalLengths = {f1, f2};

    // Return the best model with the inliers, focal lengths, and the score
    return std::make_tuple(fundamentalMatrix,
        focalLengths,
        robustEstimator.getInliers(),
        robustEstimator.getBestScore().getValue(),
        robustEstimator.getIterationNumber());
}