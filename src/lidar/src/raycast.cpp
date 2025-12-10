#include <ros/ros.h>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float32MultiArray.h>

#define PI 3.1415926

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> Points;
typedef pcl::PointCloud<PointType>::Ptr PointsPtr;

using namespace std;

geometry_msgs::Vector3 uav_position;
PointsPtr dynamic_cloud(new Points);

ros::Publisher hits_pub;
ros::Publisher rayhits2array_pub;
ros::Publisher _depth_map_pub;

float lidar_hfov;
int lidar_h_res;
int lidar_h_sample;
std::vector<double> lidar_vfov;
int lidar_v_res;
int lidar_v_sample;
float lidar_range;
float bound_h;
float h_res;
float v_res;

void position_callback(const nav_msgs::Odometry::ConstPtr& pos_data) {
    uav_position.x = pos_data->pose.pose.position.x;
    uav_position.y = pos_data->pose.pose.position.y;
    uav_position.z = pos_data->pose.pose.position.z;
}

void dynamic_callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    PointsPtr latest_cloud(new Points);
    pcl::fromROSMsg(*msg, *latest_cloud);
    *dynamic_cloud = *latest_cloud;
}

void publishRayHitsAsPointCloud2(const std::vector<Eigen::Vector3f>& ray_hits, ros::Publisher& cloud_pub) {
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = "world";
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.height = 1;
    cloud_msg.width = ray_hits.size();
    cloud_msg.is_dense = false;
    cloud_msg.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(ray_hits.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    
    std_msgs::Float32MultiArray array_msg;
    
    for (const auto& point : ray_hits) {
        array_msg.data.push_back(std::isnan(point.x()) ? std::numeric_limits<float>::quiet_NaN() : point.x());
        array_msg.data.push_back(std::isnan(point.y()) ? std::numeric_limits<float>::quiet_NaN() : point.y());
        array_msg.data.push_back(std::isnan(point.z()) ? std::numeric_limits<float>::quiet_NaN() : point.z());

        if (std::isnan(point.x()) || std::isnan(point.y()) || std::isnan(point.z())) {
            *iter_x = std::numeric_limits<float>::quiet_NaN();
            *iter_y = std::numeric_limits<float>::quiet_NaN();
            *iter_z = std::numeric_limits<float>::quiet_NaN();
        } else {
            *iter_x = point.x();
            *iter_y = point.y();
            *iter_z = point.z();
        }
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }
    
    hits_pub.publish(cloud_msg);
    rayhits2array_pub.publish(array_msg);
}

void pcl_callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    PointsPtr latest_cloud(new Points);
    pcl::fromROSMsg(*msg, *latest_cloud);
    
    *latest_cloud = *latest_cloud + *dynamic_cloud;
    
    sensor_msgs::PointCloud2 cloudMap_depth_pcd;
    pcl::toROSMsg(*latest_cloud, cloudMap_depth_pcd);
    cloudMap_depth_pcd.header.frame_id = "world";
    _depth_map_pub.publish(cloudMap_depth_pcd);

    PointsPtr cloud_filtered(new Points);
    
    for(int i = 0; i < latest_cloud->size(); i++) {
        Eigen::Vector3d Point_Orig;
        Point_Orig[0] = latest_cloud->points[i].x;
        Point_Orig[1] = latest_cloud->points[i].y;
        Point_Orig[2] = latest_cloud->points[i].z;
        
        PointType Point_World;
        Point_World.x = Point_Orig[0];
        Point_World.y = Point_Orig[1];
        Point_World.z = Point_Orig[2];
        cloud_filtered->push_back(Point_World);
    }

    int num = lidar_h_res * lidar_v_sample * lidar_h_sample * lidar_v_res;
    vector<float> distences(num, lidar_range + 1);
    std::vector<Eigen::Vector3f> rayhits(num);
    
    for (int i = 0; i < num; ++i) {
        rayhits[i] = Eigen::Vector3f(std::numeric_limits<float>::infinity(), 
                                      std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::infinity());
    }
   
    Eigen::Vector3f origin(uav_position.x, uav_position.y, uav_position.z);
    float v_low = lidar_vfov[0];
    float v_high = lidar_vfov[1];
    
    for(int i = 0; i < cloud_filtered->size(); i++) {
        float x = cloud_filtered->points[i].x;
        float y = cloud_filtered->points[i].y;
        float z = cloud_filtered->points[i].z;
        Eigen::Vector3f point_vec(x, y, z);

        float dx = x - uav_position.x;
        float dy = y - uav_position.y;
        float dz = z - uav_position.z;
        float dis = (point_vec - origin).norm();
        
        if(dis < lidar_range && (dx != 0 || dy != 0)) {
            float angle = atan2(dy, dx);
            float denominator = sqrt(dx * dx + dy * dy);
            float vertical_angle = atan2(dz, denominator);
            float deg_z = vertical_angle * 180 / PI;
            
            if (deg_z > v_low && deg_z < v_high) {
                if (angle < 0) {
                    angle += 2 * M_PI;
                }
                
                float deg = angle * 180 / PI;
                int idx = floor(deg / h_res);
                
                if (idx >= lidar_h_res * lidar_h_sample) {
                    idx = lidar_h_res * lidar_h_sample - 1;
                }
                
                int idy = floor((deg_z - lidar_vfov[0]) / v_res);
                
                if (idy >= lidar_v_res * lidar_v_sample) {
                    idy = lidar_v_res * lidar_v_sample - 1;
                }
                
                int id = idx * (lidar_v_res * lidar_v_sample) + idy;
                
                if (id < 0 || id >= num) {
                    continue;
                }
                
                if(dis < distences[id] && dis < lidar_range) {
                    distences[id] = dis;
                    rayhits[id].x() = point_vec.x();
                    rayhits[id].y() = point_vec.y();
                    rayhits[id].z() = point_vec.z();
                }
            }
        }
    }

    publishRayHitsAsPointCloud2(rayhits, hits_pub);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "raycast");
    ros::NodeHandle nh;

    nh.getParam("lidar_range", lidar_range);
    nh.getParam("lidar_h_res", lidar_h_res);
    nh.getParam("lidar_v_res", lidar_v_res);
    nh.getParam("lidar_h_sample", lidar_h_sample);
    nh.getParam("lidar_v_sample", lidar_v_sample);
    nh.getParam("lidar_hfov", lidar_hfov);
    nh.getParam("lidar_vfov", lidar_vfov);
    nh.getParam("bound_h", bound_h);

    h_res = 360.0f / (lidar_h_res * lidar_h_sample);
    v_res = 1.0f * (lidar_vfov[1] - lidar_vfov[0]) / (lidar_v_res * lidar_v_sample);

    hits_pub = nh.advertise<sensor_msgs::PointCloud2>("ray_hits", 10);
    rayhits2array_pub = nh.advertise<std_msgs::Float32MultiArray>("ray2array_hits", 10);
    
    ros::Subscriber odom_sub = nh.subscribe("/sim/odom", 1000, position_callback);
    ros::Subscriber static_sub = nh.subscribe<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 10, pcl_callback);
    ros::Subscriber dynamic_sub = nh.subscribe<sensor_msgs::PointCloud2>("/map_generator/obj_cloud", 10, dynamic_callback);

    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();
    
    return 0;
}