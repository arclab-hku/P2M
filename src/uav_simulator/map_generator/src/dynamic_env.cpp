#include <ros/ros.h>
#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Eigen>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/impl/search.hpp>

#include <sensor_msgs/PointCloud2.h>


#ifndef PCL_NO_PRECOMPILE
  #include <pcl/impl/instantiate.hpp>
  #include <pcl/point_types.h>
  PCL_INSTANTIATE(Search, PCL_POINT_TYPES)
#endif // PCL_NO_PRECOMPILE

using namespace std;

pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeLocalMap;

random_device rd;
default_random_engine eng;
std::uniform_real_distribution<double> rand_x;
std::uniform_real_distribution<double> rand_y;
std::uniform_real_distribution<double> rand_sobs_w;
std::uniform_real_distribution<double> rand_dobs_w;
std::uniform_real_distribution<double> rand_h;
std::uniform_real_distribution<double> rand_inf;

ros::Publisher _all_map_pub;
ros::Publisher _all_map_static_obs_pub;
ros::Publisher _all_map_wall_pub;
ros::Publisher _obj_cloud_pub;

int _obs_num, _moving_obs_num, seed_;
double _x_size, _y_size, _z_size;
double _x_l, _x_h, _y_l, _y_h, _sobs_w_l, _sobs_w_h, _dobs_w_l, _dobs_w_h, _h_l, _h_h;
double _resolution, _pub_rate;
double _min_dist, obs_vel_l, obs_vel_h;

bool _map_ok = false;

double radius_l_, radius_h_;
std::uniform_real_distribution<double> rand_radius_;

sensor_msgs::PointCloud2 globalMap_pcd;
sensor_msgs::PointCloud2 globalMap_static_obs_pcd;
sensor_msgs::PointCloud2 globalMap_wall_pcd;
pcl::PointCloud<pcl::PointXYZ> cloudMap;
pcl::PointCloud<pcl::PointXYZ> cloudMap_static_obs;
pcl::PointCloud<pcl::PointXYZ> cloudMap_wall;

std::vector<Eigen::Vector2d> obj_pos;
std::vector<double> obj_vel_per_x;
std::vector<double> obj_vel_amp_x;
std::vector<double> obj_vel_per_y;
std::vector<double> obj_vel_amp_y;

std::uniform_real_distribution<double> rand_vel_per_x_;
std::uniform_real_distribution<double> rand_vel_per_y_;
std::uniform_real_distribution<double> rand_vel_amp_x_;
std::uniform_real_distribution<double> rand_vel_amp_y_;

std::vector<pcl::PointCloud<pcl::PointXYZ>> obj_clusters;

void ObjUpdate(double t) {
    for (int i = 0; i < _moving_obs_num; i++) {
        float vx = obj_vel_amp_x[i] * sin(obj_vel_per_x[i] * t) + 1.1 * obj_vel_amp_x[i];
        float vy = obj_vel_amp_y[i] * sin(obj_vel_per_y[i] * t) + 1.1 * obj_vel_amp_y[i];

        obj_pos[i](0) += vx * 0.02;
        obj_pos[i](1) += vy * 0.02;
        
        for (int j = 0; j < obj_clusters[i].points.size(); ++j) {
            obj_clusters[i].points[j].x += vx * 0.02;
            obj_clusters[i].points[j].y += vy * 0.02;
        }
        
        if (obj_pos[i](0) < _x_l || obj_pos[i](0) > _x_h || obj_pos[i](1) < _y_l || obj_pos[i](1) > _y_h) {
            obj_vel_amp_x[i] = -obj_vel_amp_x[i];
            obj_vel_amp_y[i] = -obj_vel_amp_y[i];
        }
    }

    pcl::PointCloud<pcl::PointXYZ> obj_points;
    for (int i = 0; i < obj_clusters.size(); i++) {
        obj_points += obj_clusters[i];
    }

    sensor_msgs::PointCloud2 objCloud_pcd;
    pcl::toROSMsg(obj_points, objCloud_pcd);
    objCloud_pcd.header.frame_id = "world";
    objCloud_pcd.header.stamp = ros::Time::now();
    _obj_cloud_pub.publish(objCloud_pcd);
}

void RandomObjGenerate() {
    pcl::PointXYZ pt_random;

    for (int i = 0; i < _moving_obs_num; i++) {
        pcl::PointCloud<pcl::PointXYZ> cluster;

        rand_vel_per_x_ = std::uniform_real_distribution<double>(0.0, 2.0);
        rand_vel_per_y_ = std::uniform_real_distribution<double>(0.0, 2.0);
        rand_vel_amp_x_ = std::uniform_real_distribution<double>(obs_vel_l, obs_vel_h);
        rand_vel_amp_y_ = std::uniform_real_distribution<double>(obs_vel_l, obs_vel_h);
        rand_dobs_w = std::uniform_real_distribution<double>(_dobs_w_l, _dobs_w_h);

        double x, y, w, h;
        x = rand_x(eng);
        y = rand_y(eng);
        w = rand_dobs_w(eng);
        h = rand_h(eng);

        double vel_per_x = rand_vel_per_x_(eng);
        double vel_per_y = rand_vel_per_y_(eng);
        double vel_amp_x = rand_vel_amp_x_(eng);
        double vel_amp_y = rand_vel_amp_y_(eng);
        
        Eigen::Vector2d pos(x, y);
        x = floor(x / _resolution) * _resolution + _resolution / 2.0;
        y = floor(y / _resolution) * _resolution + _resolution / 2.0;
        
        obj_pos.push_back(pos);
        obj_vel_per_x.push_back(vel_per_x);
        obj_vel_amp_x.push_back(vel_amp_x);
        obj_vel_per_y.push_back(vel_per_y);
        obj_vel_amp_y.push_back(vel_amp_y);
        
        int widNum = ceil(w / _resolution);
        double radius = w;
        
        for (int r = -widNum / 2.0; r < widNum / 2.0; r++) {
            for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
                int heiNum = ceil(h / _resolution);
                for (int t = -10; t < heiNum; t++) {
                    if((r <= 0.5 - (widNum / 2.0)) || (r >= -1 + (widNum / 2.0)) || 
                       (s <= 0.5 - (widNum / 2.0)) || (s >= -1 + (widNum / 2.0))) {
                        double temp_x = x + (r + 0.5) * _resolution + 1e-2;
                        double temp_y = y + (s + 0.5) * _resolution + 1e-2;
                        double temp_z = (t + 0.5) * _resolution + 1e-2;
                        
                        if ((Eigen::Vector2d(temp_x, temp_y) - Eigen::Vector2d(x, y)).norm() <= radius) {
                            pt_random.x = temp_x;
                            pt_random.y = temp_y;
                            pt_random.z = temp_z;
                            cluster.push_back(pt_random);
                        }
                    }
                }
            }
        }
        obj_clusters.push_back(cluster);
    }
}

void CorridorGenerate() {
    pcl::PointXYZ pt_random;
    double wall_resolution = _resolution * 2;
    double _x_size_wall = _x_size + 3;
    double _y_size_wall = _y_size + 10;
    
    for (int i = 0; i < _x_size_wall / wall_resolution; ++i) {
        for (int j = 0; j < _z_size / wall_resolution; ++j) {
            pt_random.x = -(_x_size_wall / 2) + wall_resolution * i;
            pt_random.y = -(_y_size_wall / 2);
            pt_random.z = -1.0 + wall_resolution * j;
            cloudMap.points.push_back(pt_random);
            cloudMap_wall.push_back(pt_random);
        }
    }

    for (int i = 0; i < _x_size_wall / wall_resolution; ++i) {
        for (int j = 0; j < _z_size / wall_resolution; ++j) {
            pt_random.x = -(_x_size_wall / 2) + wall_resolution * i;
            pt_random.y = _y_size_wall / 2;
            pt_random.z = -1.0 + wall_resolution * j;
            cloudMap.points.push_back(pt_random);
            cloudMap_wall.push_back(pt_random);
        }
    }
    
    for (int i = 0; i < _y_size_wall / wall_resolution; ++i) {
        for (int j = 0; j < _z_size / wall_resolution; ++j) {
            pt_random.y = -(_y_size_wall / 2) + wall_resolution * i;
            pt_random.x = _x_size_wall / 2;
            pt_random.z = -1.0 + wall_resolution * j;
            cloudMap.points.push_back(pt_random);
            cloudMap_wall.push_back(pt_random);
        }
    }
    
    for (int i = 0; i < _y_size_wall / wall_resolution; ++i) {
        for (int j = 0; j < _z_size / wall_resolution; ++j) {
            pt_random.y = -(_y_size_wall / 2) + wall_resolution * i;
            pt_random.x = -(_x_size_wall / 2);
            pt_random.z = -1.0 + wall_resolution * j;
            cloudMap.points.push_back(pt_random);
            cloudMap_wall.push_back(pt_random);
        }
    }
    
    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;

    cloudMap_static_obs.width = cloudMap_static_obs.points.size();
    cloudMap_static_obs.height = 1;
    cloudMap_static_obs.is_dense = true;

    cloudMap_wall.width = cloudMap_wall.points.size();
    cloudMap_wall.height = 1;
    cloudMap_wall.is_dense = true;
}

void RandomMapGenerateCylinder() {
    pcl::PointXYZ pt_random;
    std::vector<Eigen::Vector2d> obs_position;

    rand_x = std::uniform_real_distribution<double>(_x_l, _x_h);
    rand_y = std::uniform_real_distribution<double>(_y_l, _y_h);
    rand_sobs_w = std::uniform_real_distribution<double>(_sobs_w_l, _sobs_w_h);
    rand_h = std::uniform_real_distribution<double>(_h_l, _h_h);
    rand_inf = std::uniform_real_distribution<double>(0.5, 1.5);
    rand_radius_ = std::uniform_real_distribution<double>(radius_l_, radius_h_);

    for (int i = 0; i < _obs_num && ros::ok(); i++) {
        double x = rand_x(eng);
        double y = rand_y(eng);
        double w = rand_sobs_w(eng);
        double inf = rand_inf(eng);

        bool flag_continue = false;
        for (auto p : obs_position) {
            if ((Eigen::Vector2d(x, y) - p).norm() < _min_dist) {
                i--;
                flag_continue = true;
                break;
            }
        }
        if (flag_continue)
            continue;

        obs_position.push_back(Eigen::Vector2d(x, y));

        x = floor(x / _resolution) * _resolution + _resolution / 2.0;
        y = floor(y / _resolution) * _resolution + _resolution / 2.0;

        int widNum = ceil((w * inf) / _resolution);
        double radius = (w * inf) / 2;

        for (int r = -widNum / 2.0; r < widNum / 2.0; r++) {
            for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
                double h = rand_h(eng);
                int heiNum = ceil(h / _resolution);
                for (int t = -10; t < heiNum; t++) {
                    double temp_x = x + (r + 0.5) * _resolution + 1e-2;
                    double temp_y = y + (s + 0.5) * _resolution + 1e-2;
                    double temp_z = (t + 0.5) * _resolution + 1e-2;
                    float point_rad = (Eigen::Vector2d(temp_x, temp_y) - Eigen::Vector2d(x, y)).norm();
                    
                    if (point_rad <= radius && point_rad >= radius - (_resolution * 1.2)) {
                        pt_random.x = temp_x;
                        pt_random.y = temp_y;
                        pt_random.z = temp_z;
                        cloudMap.points.push_back(pt_random);
                        cloudMap_static_obs.push_back(pt_random);
                    }
                }
            }
        }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;

    kdtreeLocalMap.setInputCloud(cloudMap.makeShared());
    _map_ok = true;
}

void pubPoints() {
    while (ros::ok()) {
        ros::spinOnce();
        if (_map_ok)
            break;
    }

    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "world";
    _all_map_pub.publish(globalMap_pcd);

    pcl::toROSMsg(cloudMap_static_obs, globalMap_static_obs_pcd);
    globalMap_static_obs_pcd.header.frame_id = "world";
    _all_map_static_obs_pub.publish(globalMap_static_obs_pcd);

    pcl::toROSMsg(cloudMap_wall, globalMap_wall_pcd);
    globalMap_wall_pcd.header.frame_id = "world";
    _all_map_wall_pub.publish(globalMap_wall_pcd);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "random_map_sensing");
    ros::NodeHandle n("~");

    _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 1);
    _all_map_static_obs_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/static_obs_cloud", 1);
    _all_map_wall_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/wall_cloud", 1);
    _obj_cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/obj_cloud", 1);

    n.param("map/x_size", _x_size, 50.0);
    n.param("map/y_size", _y_size, 50.0);
    n.param("map/z_size", _z_size, 5.0);
    n.param("map/obs_num", _obs_num, 20);
    n.param("map/resolution", _resolution, 0.2);
    n.param("map/moving_obs_num", _moving_obs_num, 40);
    n.param("map/obs_vel_l", obs_vel_l, 0.2);
    n.param("map/obs_vel_h", obs_vel_h, 2.0);
    
    n.param("ObstacleShape/sobs_lower_width", _sobs_w_l, 0.0);
    n.param("ObstacleShape/sobs_upper_width", _sobs_w_h, 0.0);
    n.param("ObstacleShape/dobs_lower_width", _dobs_w_l, 0.0);
    n.param("ObstacleShape/dobs_upper_width", _dobs_w_h, 0.0);
    n.param("ObstacleShape/lower_hei", _h_l, 3.0);
    n.param("ObstacleShape/upper_hei", _h_h, 7.0);
    n.param("ObstacleShape/radius_l", radius_l_, 7.0);
    n.param("ObstacleShape/radius_h", radius_h_, 7.0);
    n.param("ObstacleShape/seed", seed_, 0);
    n.param("pub_rate", _pub_rate, 50.0);
    n.param("min_distance", _min_dist, 1.0);

    _x_l = -_x_size / 2.0;
    _x_h = +_x_size / 2.0;
    _y_l = -_y_size / 2.0;
    _y_h = +_y_size / 2.0;
    _obs_num = std::min(_obs_num, (int)_x_size * 10);

    ros::Duration(0.5).sleep();

    unsigned int seed = static_cast<unsigned int>(seed_);
    eng.seed(seed);

    RandomMapGenerateCylinder();
    CorridorGenerate();
    RandomObjGenerate();

    ros::Rate loop_rate(_pub_rate);
    ros::Time start_time = ros::Time::now();

    while (ros::ok()) {
        ros::Time cur_time = ros::Time::now();
        ObjUpdate((cur_time - start_time).toSec());
        pubPoints();
        ros::spinOnce();
        loop_rate.sleep();
    }
}






