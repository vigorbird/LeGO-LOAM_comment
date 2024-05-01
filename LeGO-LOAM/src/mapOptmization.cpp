// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class mapOptimization{

private:

    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;

    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;//每个关键帧的尖锐点数据
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

    deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    vector<int> surroundingExistingKeyPosesID;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;
    
    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;//最近关键帧的位姿

    // PointType(pcl::PointXYZI)的XYZI分别保存3个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    //PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    
    // 结尾有DS代表是downsize,进行过下采样
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;//typedef pcl::PointXYZI  PointType;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;//上个节点laser_cloud_corner_last发送来的消息
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;//上个节点laser_cloud_surf_last发送来的消息
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;//上个节点发送来的角点数据 降采样结果
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;//上个节点发送来的平面数据 降采样结果

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;//上个节点发送来的
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;//上个节点发送来的outlier数据降采样结果

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;//laserCloudSurfTotalLastDS = (输入的平面点降采样+输入的outlier降采样)再进行一次降采样

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;//最新关键帧相关的地图，存储的是各个关键帧的角点
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;//最新关键帧相关的地图，存储的是各个关键帧的平面点和outlier
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;//laserCloudCornerFromMap的降采样结果
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;//laserCloudSurfFromMap降采样的结果
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;//搜索范围是最近关键帧周围地图的降采样角点
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;//搜索范围是最近关键帧周围地图的降采样平面点和outlier

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;//闭环帧前后25帧没有降采样的点云
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;//闭环帧前后25帧进行0.4降采样之后的点云

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;//最近关键帧的点云
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;

    double timeLaserCloudCornerLast;
    double timeLaserCloudSurfLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;

    float transformLast[6];

    /*************高频转换量**************/
    // odometry计算得到的到世界坐标系下的转移矩阵
    float transformSum[6];//序号0对应theta_x,1对应theta_y,2对应theta_z,剩下的对应tx ty tz，从0坐标系移动到当前帧坐标系发生的位姿变换
    // 转移增量，只使用了后三个平移增量
    float transformIncre[6];

    /*************低频转换量*************/
    // 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
    float transformTobeMapped[6];//当前帧的经过修正的位姿，详见transformAssociateToMap函数
    // 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
    float transformBefMapped[6];//用于存放里程计发送来的当前帧的位姿
    float transformAftMapped[6]; // 存放mapping之后的经过mapping微调之后的位姿


    int imuPointerFront;
    int imuPointerLast;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing;

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    bool isDegenerate;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;

    bool aLoopIsClosed;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;//当前帧经过最近关键帧修正的在世界坐标系下的位姿，即世界坐标系到当前帧坐标的位姿变化
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

public:

    

    mapOptimization():
        nh("~")
    {
        // 用于闭环图优化的参数设置，使用gtsam库
    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);//更新了transformSum
        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

        // 设置滤波时创建的体素大小为0.2m/0.4m立方体,下面的单位为m
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";

        allocateMemory();
    }

    void allocateMemory(){

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());        

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        
        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        // matA1为边缘特征的协方差矩阵
        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征值
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征向量，对应于matD1存储
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    // 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
    //使用transformSum，transformAftMapped和transformBefMapped更新transformTobeMapped
    //我们已经知道了上一个关键帧地图优化之后的位姿，和其里程计的位姿，又知道了当前帧里程计的位姿
    //得到当前帧修正后的位姿=T_上一个关键帧地图优化之后的位姿*(T_当前帧里程计的位姿.invserse()*T_上一个关键帧里程计的位姿)
    void transformAssociateToMap()
    {   
    	//将上个里程计在世界坐标系下的位置变换到当前帧的坐标系下
    	//R_z.inverse()*R_x.inverse()*R_y.inverse()(t-t_sum)
	    //因为transformSum是从世界坐标系移动到当前帧坐标系的位姿，transformBefMapped[3-5]上一针的里程计在世界坐标系下的位置
	    //为了得到上一帧的里程计在当前帧的位置则 transformIncre = T_sum.inverse()*p_befmapped
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;//上一帧的里程计在当前帧坐标系下的位置
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                       crycrx / cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3]- (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

	//使用transformSum更新transformBefMapped，使用transformTobeMapped更新transformAftMapped
    void transformUpdate()
    {
		if (imuPointerLast >= 0) //不使用imu不进入这个条件
		{
		    float imuRollLast = 0, imuPitchLast = 0;
		    while (imuPointerFront != imuPointerLast) 
			{
		        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
		            break;
		        }
		        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
		    }

		    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		        imuRollLast = imuRoll[imuPointerFront];
		        imuPitchLast = imuPitch[imuPointerFront];
		    } else {
		        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
		                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
		                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

		        imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		    }

		    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		for (int i = 0; i < 6; i++) 
		{
		    transformBefMapped[i] = transformSum[i];//这个cpp文件中就这里更新了transformBefMapped的值
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

	//
    void updatePointAssociateToMapSinCos()
	{
        // 先提前求好roll,pitch,yaw的sin和cos值
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }
    //R_y*R_x*R_z*(x,y,z)+t
    //将当前帧的坐标点变换到世界坐标系下
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        // 进行6自由度的变换，先进行旋转，然后再平移
        // 主要进行坐标变换，将局部坐标转换到全局坐标中去	

        // 先绕z轴旋转
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        // [x1,y1,z1]^T=Rz*[pi->x,pi->y,pi->z]
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        // 最后再绕Y轴旋转，然后加上平移
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void updateTransformPointCloudSinCos(PointTypePose *tIn){

        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

	 //R_y*R_x*R_z*p+t
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn)
    {

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i)
		{

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

	//R_y*R_x*R_z*p+t,将k时刻的点云变换到地图坐标系下
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
	{

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

		// 坐标系变换，旋转rpy角
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

     //更新了transformSum
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
    {
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }

	//
    void publishTF()
	{
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];//这里作者非常省事没有自己定义一个数据类型而是直接使用的ros现有的里程计数据类型来保存两个位姿
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);//发布一个topic = aft_mapped_to_init

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);//发布一个tf树 camera_init=frame_id, child_frame_id=aft_mapped
    }

	//
    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0)//topic name = key_pose_origin,默认不进入这个条件
		{
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0)//topic name = recent_cloud,默认不进入这个条件
		{
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }
    }

    void visualizeGlobalMapThread(){
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }
    }

    void publishGlobalMap(){

        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 通过KDTree进行最近邻搜索
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
          globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // 对globalMapKeyPoses进行下采样
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i)
		{
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // 对globalMapKeyFrames进行下采样
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
 
        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        globalMapKeyFramesDS->clear();     
    }

    void loopClosureThread(){

        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(1);// 1秒钟执行一此回环检测
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    }


	//这个函数主要是得到当前帧在世界坐标系下的点云latestSurfKeyFrameCloud，和回环帧前后25帧在世界坐标系下的降采样点云nearHistorySurfKeyFrameCloudDS
	//并粗略判断是否发生了回环，发生回环的条件是存在距离当前帧小于50米的其他关键帧，并且这个关键帧的时间戳与里程计的时间戳大于30秒
    bool detectLoopClosure(){

        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();

        // 资源分配时初始化
        // 在互斥量被析构前不解锁
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
		//在最新的关键的范围中搜索距离当前帧小于50的集合
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        closestHistoryFrameID = -1;
		//不在附近中帧搜索回环帧
        for (int i = 0; i < pointSearchIndLoop.size(); ++i)
		{
            int id = pointSearchIndLoop[i];
            // 两个时间差值大于30秒
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0)
			{
                closestHistoryFrameID = id;
                break;
            }
        }
        if (closestHistoryFrameID == -1){
            // 找到的点和当前时间上没有超过30秒的
            return false;
        }

        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        // 将最近关键帧的角点和平面点变换到地图坐标系下
        *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        
        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());//存储的是最近关键帧的平面和角点点云在世界坐标系下的坐标
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
		{
            // intensity不小于0的点放进hahaCloud队列
            // 初始化时intensity是-1，滤掉那些点
            if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0)
			{
                hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
            }
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud   = *hahaCloud;

        // 将回环帧的前后25个帧的点云变换到地图坐标系下得到nearHistorySurfKeyFrameCloud
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
		{
		    // 要求closestHistoryFrameID + j在0到cloudKeyPoses3D->points.size()-1之间,不能超过索引
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
        }

        // 对回环帧的点云数据进行降采样得到nearHistorySurfKeyFrameCloudDS
        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);//进行0.4的降采样滤波
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

        if (pubHistoryKeyFrames.getNumSubscribers() != 0)//默认不进入这个条件
		{
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }

        return true;
    }

   //
    void performLoopClosure()
   {

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (potentialLoopFlag == false)
		{

            if (detectLoopClosure() == true)//非常重要的函数!!!!!!!!!!! 待测试???????????????????????????????????
			{
                potentialLoopFlag = true;
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false)
                return;
        }

        potentialLoopFlag = false;
        //当前帧的点云和回环帧的降采样点云进行icp匹配 
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);// 设置RANSAC运行次数
        icp.setInputSource(latestSurfKeyFrameCloud);//最近关键帧的点云
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);//闭环帧的降采样点云
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 为什么匹配分数高直接返回???分数高代表噪声太多
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)// historyKeyframeFitnessScore = 0.3;
            return;

        
        if (pubIcpKeyFrames.getNumSubscribers() != 0)//默认不进入这个条件
		{
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
			// icp.getFinalTransformation()的返回值是Eigen::Matrix<Scalar, 4, 4>
            pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }   
        // 以下在点云icp收敛并且噪声量在一定范围内进行
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation();//得到当前帧和闭环帧的相对位姿
		// 得到平移和旋转的角度
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);//最近帧的位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;//根据相对位姿和闭环帧的位姿得到最近帧的修正结果
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));//得到修正之后的最近关键帧的gtsam位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);//得到闭环帧的gtsam位姿
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        //使用gtsam做pgo
        std::lock_guard<std::mutex> lock(mtx);
        gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        gtSAMgraph.resize(0);

        aLoopIsClosed = true;
    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){
    	return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                           Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint)
	{
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    //更新邻近关键帧的信息并根据临近关键帧更新邻近的点云数据:laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
    void extractSurroundingKeyFrames()
    {

        if (cloudKeyPoses3D->points.empty() == true)
            return;	

       
		if (loopClosureEnableFlag == true)//默认不进入这个条件
		{
            /*这里进行注释方便阅读代码
            // recentCornerCloudKeyFrames保存的点云数量太少，则清空后重新塞入新的点云直至数量够
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum)//surroundingKeyframeSearchNum = 50
			{
                recentCornerCloudKeyFrames. clear();
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size();
                for (int i = numPoses-1; i >= 0; --i)
				{
                    // cloudKeyPoses3D的intensity中存的是索引值?
                    // 保存的索引值从1开始编号；
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);//更新全局变量
                    // 依据上面得到的变换thisTransformation，对cornerCloudKeyFrames，surfCloudKeyFrames，surfCloudKeyFrames
                    // 进行坐标变换
                    recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)//surroundingKeyframeSearchNum = 50
                        break;
                }


            }else{
                /
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1)//表示关键帧进行了更新
				{

                    recentCornerCloudKeyFrames. pop_front();
                    recentSurfCloudKeyFrames.   pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
                   
                    latestFrameID = cloudKeyPoses3D->points.size() - 1;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
                    recentSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                }
            }

            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i)
			{
                // 两个pcl::PointXYZI相加?
                // 注意这里把recentOutlierCloudKeyFrames也加入到了laserCloudSurfFromMap
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
            }
            */
		}else{

            // 下面这部分是没有闭环的代码，默认是没有回环的代码
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
            // 进行半径surroundingKeyframeSearchRadius内的邻域搜索=50，
            //在所有关键帧的位姿集合中搜索距离当前帧小于50米的集合，并将这些位姿集合保存到surroundingKeyPoses，而surroundingKeyPosesDS是surroundingKeyPoses的降采样
			kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
			kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
			for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
			downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);//进行1.0的降采样
			downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();

			//如果有新的关键帧插入，那么我们就需要更新一下最新关键帧周围的地图信息和相连帧信息了
			//基本思路如下:在之前的相连帧中搜索是否有距离最新帧50米的关键帧，如果没有则将之前的相连帧中的这个帧删除
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
			{
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j)
				{
                    // 双重循环，不断对比surroundingExistingKeyPosesID[i]和surroundingKeyPosesDS的点的index
                    // 如果能够找到一样的，说明存在相同的关键点(因为surroundingKeyPosesDS从cloudKeyPoses3D中筛选而来)
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
					{
                        existingFlag = true;
                        break;
                    }
                }
				
                if (existingFlag == false)
				{
                    // 如果surroundingExistingKeyPosesID[i]对比了一轮的已经存在的关键位姿的索引后（intensity保存的就是size()）
                    // 没有找到相同的关键点，那么把这个点从当前队列中删除
                    // 否则的话，existingFlag为true，该关键点就将它留在队列中
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }

          
            //在距离最新关键帧50米的其他关键帧中搜索是否有之前相连帧的信息，如果没有则向相连帧中添加信息
            for (int i = 0; i < numSurroundingPosesDS; ++i) 
			{
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter)
				{
                    // 把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                    // 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点没有对应的，也就是新的点
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity)
					{
                        existingFlag = true;
                        break;
                    }
                }
				
                if (existingFlag == true)
				{
                    continue;
                }else
                {
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));//某个关键帧对应的特征点经过变换压入到周围的环境中去
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) 
			{
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];//主线程每次都会将laserCloudCornerFromMap laserCloudSurfFromMap归零
				
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
		}//else结束

        // 进行两次下采样
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();//0.2的下采样

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }//extractSurroundingKeyFrames函数结束

	//对输入的平面点，角点和outlier进行降采样
    void downsampleCurrentScan(){

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);//上个节点发送来的
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();//进行0.2的降采样

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);//上个节点发送来的
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();//进行0.4的降采样

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);//上个节点发送来的
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();//进行0.4的降采样

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);//进行0.4的降采样，laserCloudSurfTotalLastDS = (输入的平面点降采样+输入的outlier降采样)再进行一次降采样
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

    //将当前帧的点与地图匹配的点做匹配，然后计算距离和垂线的方向
    //输入的迭代次数没有用!!!!!!!!!!!!
    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();
		//遍历输入的降采样的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) //这个函数就是一个大的for循环 
		{
            pointOri = laserCloudCornerLastDS->points[i];//当前帧的降采样之后的角点数据
            
            // 将当前帧的角点数据变换到世界坐标系下
            pointAssociateToMap(&pointOri, &pointSel);

            // 从最近关键帧周围地图的降采样角点中搜索与当前帧输入的角点最近的那5个点
            //kdtreeCornerFromMap的输入点云是laserCloudCornerFromMapDS=与地图匹配的角点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            // 只有当最远的那个邻域点的距离pointSearchSqDis[4]小于1m时才进行下面的计算
            // 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
            if (pointSearchSqDis[4] < 1.0) 
			{
                // 先求5个样本的平均值
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) 
				{
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 下面在求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
                // 更准确地说应该是在求协方差matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) 
				{
                    // ax代表的是x-cx,表示均值与每个实际值的差值，求取5个之后再次取平均，得到matA1
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中
                cv::eigen(matA1, matD1, matV1);

                // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
                // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) //最大的那个特征值大于三倍的第二大的特征值，表示直线表征明显
				{

					//求x0到x1和x2构建的直线上的距离
                    float x0 = pointSel.x;//当前帧上的角点
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
					//这里要非常注意了，我们在与地图匹配的角点中找到了5个点与当前阵的五个点最近，
					//相比于loam这里作者做了个优化，没有计算最近的两个点而是得到五个最近点，然后用这五个点拟合一条直线然后再算最近的点到这个直线的距离
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);//一定要注意这里使用的是特征值进行的向量的构建
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // l12表示的是0.2*(||V1[0]||)
                    // 也就是平行四边形一条底的长度
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // LLL=[la,lb,lc]是0.2*V1[0]这条高上的单位法向量。||LLL||=1；
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                    
                    // 计算点pointSel到直线的距离
                    // 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                    float ld2 = a012 / l12;

                    // 如果在最理想的状态的话，ld2应该为0，表示点在直线上
                    // 最理想状态s=1；
                    float s = 1 - 0.9 * fabs(ld2);
                    
                    // coeff代表系数的意思
                    // coff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;

                    // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;
                    
                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

	//输入的迭代次数没有用!!!!!!!!!!!!!
	//计算点到平面的距离和平面法向量
    void surfOptimization(int iterCount)
	{
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++)//这个函数就是这个大的for循环
		{
            pointOri = laserCloudSurfTotalLastDS->points[i];//当前帧的(输入的平面点降采样+输入的outlier降采样)再进行一次降采样，我们这里称为地图匹配的平面点
            pointAssociateToMap(&pointOri, &pointSel); 
			//从与地图匹配的点中搜索5个点与当前帧的点最近
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0)//最远的点要小于1米
			{
                for (int j = 0; j < 5; j++) 
				{
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // matB0是一个5x1的矩阵
                // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                // matX0是3x1的矩阵
                // 求解方程matA0*matX0=matB0
                // 公式其实是在求由matA0中的点构成的平面的法向量matX0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                // [pa,pb,pc,pd]=[matX0,pd]
                // 正常情况下（见后面planeValid判断条件），应该是
                // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                // 所以pd设置为1
                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                // 对[pa,pb,pc,pd]进行单位化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 求解后再次检查平面是否是有效平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) 
				{
					//代入平面的方程Ax+By+Cz+D = 0
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)//X*Y = |X|*|Y|costheta
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) 
				{
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;//点到平面的距离

                    // 后面部分相除求的是[pa,pb,pc,pd]与pointSel的夹角余弦值(两个sqrt，其实并不是余弦值)
                    // 这个夹角余弦值越小越好，越小证明所求的[pa,pb,pc,pd]与平面越垂直
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 判断是否是合格平面，是就加入laserCloudOri
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // 这部分的代码是基于高斯牛顿法的优化，不是zhang ji论文中提到的基于L-M的优化方法
    // 这部分的代码使用旋转矩阵对欧拉角求导，优化欧拉角，不是zhang ji论文中提到的使用angle-axis的优化
    bool LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);//当前帧的经过修正的位姿，详见transformAssociateToMap函数
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();//匹配上的点的数量
        // laser cloud original 点云太少，就跳过这次循环
        if (laserCloudSelNum < 50) 
		{
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        for (int i = 0; i < laserCloudSelNum; i++)//遍历匹配上的点
		{
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
            // 更详细的数学推导参看wykxwyc.github.io
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            // 同上，求解的是对pitch的偏导量
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;


            /*
            在求点到直线的距离时，coeff表示的是如下内容
            [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            在求点到平面的距离时，coeff表示的是
            [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */
            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;

            // 这部分是雅克比矩阵中距离对平移的偏导
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;

            // 残差项
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // 将矩阵由matA转置生成matAt
        // 先进行计算，以便于后边调用 cv::solve求解
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // 利用高斯牛顿法进行求解，
        // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J*f(x)
        // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
        // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) 
		{
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 对近似的Hessian矩阵求特征值和特征向量，
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) 
		{
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization()
	{
        //与地图匹配的角点降采样之后数量要大于10，与地图匹配的平面点降采样之后数量要大于100
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) 
		{

            
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);//最近关键帧周围地图的角点
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);//最近关键帧对周围的平面点和outlier

            for (int iterCount = 0; iterCount < 10; iterCount++) 
			{
                // 用for循环控制迭代次数，最多迭代10次
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);//将当前帧的点与地图匹配的点做匹配，然后计算距离和垂线的方向.注意输入的迭代次数没有用!!!!!!!!!!!!!!
                surfOptimization(iterCount);//计算点到平面的距离和平面法向量

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // 迭代结束更新相关的转移矩阵
            transformUpdate();//使用transformSum更新transformBefMapped，使用transformTobeMapped更新transformAftMapped
        }
    }

    //判断当前帧是不是关键帧，根据移动的距离进行判断
    //如果是关键帧将上一个关键帧的位姿和当前关键帧的位姿加入到pgo中，并进行一次优化，
    //将优化后的关键帧的位姿更新transformAftMapped
    void saveKeyFramesAndFactor(){

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3)//机器人移动了0.3米才会产生新的关键帧
        {
            saveThisKeyFrame = false;
        }

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;

        if (cloudKeyPoses3D->points.empty())//如果是第一个关键帧
		{
            // NonlinearFactorGraph增加一个PriorFactor因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transformLast[i] = transformTobeMapped[i];
        }
        else
		{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));//应该是上一个关键帧的位姿
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));//当前帧的关键帧
			
            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
																		 Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));//将当前帧的初值插入gtsam状态中
        }

        // initialEstimate是加到系统中的新变量的初始点
        isam->update(gtSAMgraph, initialEstimate);
        // update 函数为什么需要调用两次？
        isam->update();

		// 删除内容?
        gtSAMgraph.resize(0);
		initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);//得到最新一帧的位姿

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);//更新关键帧的位姿

        if (cloudKeyPoses3D->points.size() > 1)
		{
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i)
			{
            	transformLast[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        // PCL::copyPointCloud(const pcl::PCLPointCloud2 &cloud_in,pcl::PCLPointCloud2 &cloud_out )   
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);//将每个关键帧的尖锐点保存
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    }

	//如果发生了回环，则遍历所有的关键帧，并将pgo优化的结果更新所有的关键帧
    void correctPoses()
    {
	    if (aLoopIsClosed == true)//发生了回环
		{
	            recentCornerCloudKeyFrames. clear();
	            recentSurfCloudKeyFrames.   clear();
	            recentOutlierCloudKeyFrames.clear();

	            int numPoses = isamCurrentEstimate.size();
				for (int i = 0; i < numPoses; ++i)//遍历所有的关键帧，并将pgo优化的结果更新所有的关键帧
				{
					cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
					cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
					cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

					cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        	        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                    cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
             
            		cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            		cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
            		cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
				}
		    	aLoopIsClosed = false;
		  }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run()
	{

        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {

            newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx);

            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval)//默认是0.3
	    	{

                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();//当前帧修正后的位姿=T_上一个关键帧地图优化之后的位姿*(T_当前帧里程计的位姿.invserse()*T_上一个关键帧里程计的位姿)

                extractSurroundingKeyFrames();//得到与地图匹配的尖锐的点和平的点:laserCloudCornerFromMapDS和laserCloudSurfFromMapDS

                downsampleCurrentScan();//对输入的平面点，角点和outlier进行降采样

                // 当前扫描进行边缘优化，图优化以及进行LM优化的过程
                //最最重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                scan2MapOptimization();
				
                //判断当前帧是不是关键帧，根据移动的距离进行判断
    			//如果是关键帧将上一个关键帧的位姿和当前关键帧的位姿加入到pgo中，并进行一次优化，
    			//将优化后的关键帧的位姿更新transformAftMapped
                saveKeyFramesAndFactor();

                correctPoses();//如果发生了回环，则遍历所有的关键帧，并将pgo优化的结果更新所有的关键帧

                publishTF();//将优化的当前位姿通过tpopic 发布出去，tf树发布出去

                publishKeyPosesAndFrames();//这个函数什么都没做

                clearCloud();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

    // 进行闭环检测与闭环的功能
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
	
    // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();//最最重要的函数!!!!!!!!!!!!!!!!

        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}



