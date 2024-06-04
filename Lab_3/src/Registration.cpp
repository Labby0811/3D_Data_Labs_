#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  PointDistance(Eigen::Vector3d source_point, Eigen::Vector3d target_point) : target_point(target_point), source_point(source_point){}

  template<typename T> bool operator()(const T *const transf, T *residuals) const 
  {
    Eigen::Matrix<T, 3, 1> old_source = {T(source_point[0]), T(source_point[1]), T(source_point[2])};
    Eigen::Matrix<T, 3, 1> new_source;

    //apply rotation
    ceres::AngleAxisRotatePoint(transf, old_source.data(), new_source.data());

    //apply translation
    for(int i = 0; i < 3; i++)
      new_source[i] += transf[i + 3];  

    //compute residuals
    for(int i = 0; i < 3; i++)
      residuals[i] = new_source[i] - T(target_point[i]);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d source_point, const Eigen::Vector3d target_point) {
        return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(
                new PointDistance(source_point, target_point)));
    }
  
  Eigen::Vector3d source_point, target_point;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(mode != "svd" && mode != "lm")
  {
    std::cout << "Invalid mode" << std::endl;   //shouldn't arrive there but a double check isn't a problem
    return;
  }

  Eigen::Matrix4d new_transformation = Eigen::Matrix4d::Identity(4,4);
  double prev_rmse = 0.0;

  for(int iteration = 0; iteration < max_iteration; iteration++)
  {
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> closest_pt = find_closest_point(threshold);
    std::vector<size_t> source_indices = std::get<0>(closest_pt);
    std::vector<size_t> target_indices = std::get<1>(closest_pt);
    double current_rmse = std::get<2>(closest_pt);

    //check if the abs difference between current rmse and prev rmse is less than relative rmse, a terminationn criterion
    if(std::abs(current_rmse - prev_rmse) < relative_rmse)
    {
      std::cout << "Converged at iteration: " << iteration << std::endl;
      return;
    }

    prev_rmse = current_rmse;

    if(mode == "svd")
    {
      new_transformation = get_svd_icp_transformation(source_indices, target_indices);
    }
    else  //mode == "lm"
    {
      new_transformation = get_lm_icp_registration(source_indices, target_indices);
    }

    Eigen::Matrix4d prev_transf = get_transformation(); 
    Eigen::Matrix4d current_transformation = Eigen::Matrix4d::Identity(4,4);
    current_transformation.block<3,3>(0,0) = new_transformation.block<3,3>(0,0) * prev_transf.block<3,3>(0,0);
    current_transformation.block<3,1>(0,3) = new_transformation.block<3,3>(0,0) * prev_transf.block<3,1>(0,3) + new_transformation.block<3,1>(0,3);
    set_transformation(current_transformation);

    source_for_icp_.Transform(new_transformation);
    std::cout << "Transformation matrix: " << std::endl;

  }

  std::cout <<  "Diverged: MAX_ITERATION surpassed." << std::endl;

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double mse, rmse;
  std::vector<int> target_idx(1);
  std::vector<double> dist2(1);
  
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_for_icp_;
  
  int num_source_points  = source_clone.points_.size();

  for(size_t source_idx = 0; source_idx < num_source_points; source_idx++)
  {
    
    source_point = source_clone.points_[source_idx];
    target_kd_tree.SearchKNN(source_point, 1, target_idx, dist2);
    
    //save iff distance is smaller than threshold   -   else discarrd the index
    if(sqrt(dist2[0]) <= threshold)
    {
      target_indices.push_back(target_idx[0]);
      source_indices.push_back(source_idx);

      //mse update
      mse = mse * source_idx/(source_idx + 1) + dist2[0]/(source_idx + 1);
    }

  }

  rmse = sqrt(mse);
  std::cout << "RMSE: " << rmse << std::endl;
  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  //We use source_for_icp to use the transformed source cloud directly
  open3d::geometry::PointCloud source_clone = source_for_icp_; 
  open3d::geometry::PointCloud target_clone = target_;

  //Compute centroids
  Eigen::Vector3d source_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_centroid = Eigen::Vector3d::Zero();
  int source_size = source_clone.points_.size();
  int target_size = target_clone.points_.size();

  //dc = (dc + di)/N
  for(int i = 0; i < source_size; i++)
    source_centroid += source_clone.points_[i];
  
  source_centroid /= source_size;

  //mc = (mc + mi)/N
  for(int i = 0; i < target_size; i++)
    target_centroid += target_clone.points_[i];
  
  target_centroid /= target_size;


  //Now that we've found the centroids we can subtract them from their respective clouds points and create the 3X3 matrix W
  //Creating the subtracted source and target matrices di' = (di - dc) and mi' = (mi - mc)
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

  //filling the subtracted source and target matrices
  for (size_t i = 0; i < source_indices.size(); i++)  //source_indices.size() == target_indices.size() because they are coupled
  {
    //saving current source and target points
    Eigen::Vector3d source_point = source_clone.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_clone.points_[target_indices[i]];

    //subtracting the centroids
    source_point = source_point - source_centroid;
    target_point = target_point - target_centroid;
    W = W + target_point * source_point.transpose();  //W = W + (mi - mc) * (di - dc)^T
  }

  //We now have to find R and t
  //SVD computation
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

  //We have to check if the determinant of the matrix is negative
  //Note that det(U*V^T) = det(U) * det(V^T) = det(U) * det(V),
  //so for semplification we can check the product of the determinants
  if((svd.matrixU().determinant() * svd.matrixV().determinant()) == -1) //Special reflection case(if corrupted data is present)
  {
    Eigen::Matrix3d diag = Eigen::Matrix3d::Identity();
    diag(2,2) = -1;
    R = svd.matrixU() * diag * svd.matrixV().transpose();  //R = U * diag(1, 1, -1) * V^T
  }
  
  else //Standard case
    R = svd.matrixU() * svd.matrixV().transpose();  //R = U * V^T

  //Now we have to find the translation vector
  t = target_centroid - R * source_centroid;   //t = mc - R * dc

  //FINAL TRANSFORMATION MATRIX
  transformation.block<3,3>(0,0) = R;
  transformation.block<3,1>(0,3) = t;
  
  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

  //saving the source points in a clone
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  ceres::Problem problem;
  ceres::Solver::Summary summary;

  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    ceres::CostFunction* cost_function =PointDistance::Create(source_clone.points_[source_indices[i]], target_.points_[target_indices[i]]);
    problem.AddResidualBlock(cost_function,
                           nullptr /* squared loss */,
                           transformation_arr.data());
  }

  ceres::Solve(options, &problem, &summary);

  //Now we have to convert the euler angles in a rotation matrix
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  
  R = (Eigen::AngleAxisd(transformation_arr[0], Eigen::Vector3d::UnitX()) * 
  Eigen::AngleAxisd(transformation_arr[1], Eigen::Vector3d::UnitY()) * 
  Eigen::AngleAxisd(transformation_arr[2], Eigen::Vector3d::UnitZ())).toRotationMatrix();

  //Now we have to find the translation vector
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  t << transformation_arr[3], transformation_arr[4], transformation_arr[5];

  //Saving the transformation matrix
  transformation.block<3,3>(0,0) = R;
  transformation.block<3,1>(0,3) = t;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


