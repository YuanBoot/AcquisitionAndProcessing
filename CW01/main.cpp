

#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>

#include <Eigen/Geometry>

#include <iostream>

#include <igl/readOBJ.h>

#include "nanoflann/include/nanoflann.hpp"

using namespace Eigen;
using namespace std;
using namespace nanoflann;

//
typedef std::pair<Eigen::MatrixXd, Eigen::MatrixXi> Mesh;

struct model{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

model concat_meshes(Eigen::MatrixXd VA, Eigen::MatrixXi FA,
                    Eigen::MatrixXd VB, Eigen::MatrixXi FB) {


    //Found this way of concatenating meshes in the libigl github comments
    // Concatenate (Target_V,Target_F) and (Model_V,Model_F) into (V,F)
    Eigen::MatrixXd V(VA.rows() + VB.rows(), VA.cols());
    V << VA, VB;

    Eigen::MatrixXi F(FA.rows() + FB.rows(), FA.cols());
    F << FA, (FB.array() + VA.rows());

    struct model result;
    result.V = V;
    result.F = F;

    return result;
}


void compute_closest_points(Eigen::MatrixXd target, Eigen::MatrixXd measurement){

    //bulid kd tree
    int dim = 3;
    KDTreeEigenMatrixAdaptor< Eigen::MatrixXd > kd_tree_index(target, 10);
    kd_tree_index.index -> buildIndex();


    int number_point = measurement.rows();

    std::cout << number_point << std::endl;

    for (int i=0; i < number_point; i++) {
        std::vector<double> query_point = {target(i, 0),
                                           target(i, 1),
                                           target(i, 2)};
    }

    // do a knn search
    const size_t num_results = 10;
    vector<size_t> ret_indexes(num_results);
    vector<double> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<double> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
    //kd_tree_index.index->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));

    std::cout << "knnSearch(nn="<<num_results<<"): \n";
    for (size_t i=0; i<num_results; i++)
        std::cout << "ret_index["<<i<<"]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;

}

void build_kdtree(Eigen::MatrixXd target){

    int dim = 3;
    //Eigen::MatrixXd  mat(target.rows(), dim);
    //mat = target;

    std::vector<double> query_point(dim);

    typedef KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>  kd_tree_t;

    kd_tree_t kd_tree_index(target, 10);
    kd_tree_index.index -> buildIndex();

    // do a knn search
    const size_t num_results = 10;
    vector<size_t>   ret_indexes(num_results);
    vector<double> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<double> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
    kd_tree_index.index->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));

    std::cout << "knnSearch(nn="<<num_results<<"): \n";
    for (size_t i=0; i<num_results; i++)
        std::cout << "ret_index["<<i<<"]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;
}

int main(int argc, char *argv[])
{
    struct model target;
    struct model shift;

    igl::readOBJ("../bunny/data/bun000yuan02.obj", target.V, target.F);
    igl::readOBJ("../bunny/data/bun045yuan01.obj", shift.V, shift.F);

    // Compute the optimal transformation of the data
    Eigen::AngleAxisd rotationVector(-M_PI/4, Eigen::Vector3d(0,0,1));
    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
    rotationMatrix = rotationVector.toRotationMatrix();

    // Transform the data mesh
    target.V = target.V * rotationMatrix.transpose();

    Eigen::Vector3d translation;
    translation = Eigen::Vector3d(0.1,0,0);
    target.V = target.V + translation.transpose().replicate(target.V.rows(), 1);

    struct model result = concat_meshes(target.V, target.F, shift.V, shift.F);

    Mesh mesh = std::make_pair(result.V,result.F);

    igl::viewer::Viewer viewer;


    //viewer.data.add_points(target.V,Eigen::RowVector3d(1,0,0));
    viewer.data.set_points(target.V,Eigen::RowVector3d(1,1,0));
    viewer.data.add_points(shift.V,Eigen::RowVector3d(0,1,1));
    viewer.core.align_camera_center(target.V,target.F);
    viewer.core.point_size = 0.5;
    //viewer.data.set_mesh(target.V,target.F);
    viewer.launch();

    //compute_closest_points(target.V,target.V);

    return 0;
}

