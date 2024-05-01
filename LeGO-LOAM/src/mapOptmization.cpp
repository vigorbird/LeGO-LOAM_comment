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

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;//ÿ���ؼ�֡�ļ��������
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
    PointType currentRobotPosPoint;//����ؼ�֡��λ��

    // PointType(pcl::PointXYZI)��XYZI�ֱ𱣴�3�������ϵ�ƽ�ƺ�һ������(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    //PointTypePose��XYZI�����cloudKeyPoses3Dһ�������ݣ����⻹����RPY�Ƕ��Լ�һ��ʱ��ֵtimeLaserOdometry
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    
    // ��β��DS������downsize,���й��²���
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;//typedef pcl::PointXYZI  PointType;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;//�ϸ��ڵ�laser_cloud_corner_last����������Ϣ
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;//�ϸ��ڵ�laser_cloud_surf_last����������Ϣ
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;//�ϸ��ڵ㷢�����Ľǵ����� ���������
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;//�ϸ��ڵ㷢������ƽ������ ���������

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;//�ϸ��ڵ㷢������
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;//�ϸ��ڵ㷢������outlier���ݽ��������

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;//laserCloudSurfTotalLastDS = (�����ƽ��㽵����+�����outlier������)�ٽ���һ�ν�����

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;//���¹ؼ�֡��صĵ�ͼ���洢���Ǹ����ؼ�֡�Ľǵ�
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;//���¹ؼ�֡��صĵ�ͼ���洢���Ǹ����ؼ�֡��ƽ����outlier
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;//laserCloudCornerFromMap�Ľ��������
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;//laserCloudSurfFromMap�������Ľ��
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;//������Χ������ؼ�֡��Χ��ͼ�Ľ������ǵ�
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;//������Χ������ؼ�֡��Χ��ͼ�Ľ�����ƽ����outlier

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;//�ջ�֡ǰ��25֡û�н������ĵ���
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;//�ջ�֡ǰ��25֡����0.4������֮��ĵ���

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;//����ؼ�֡�ĵ���
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

    /*************��Ƶת����**************/
    // odometry����õ��ĵ���������ϵ�µ�ת�ƾ���
    float transformSum[6];//���0��Ӧtheta_x,1��Ӧtheta_y,2��Ӧtheta_z,ʣ�µĶ�Ӧtx ty tz����0����ϵ�ƶ�����ǰ֡����ϵ������λ�˱任
    // ת��������ֻʹ���˺�����ƽ������
    float transformIncre[6];

    /*************��Ƶת����*************/
    // ����ʼλ��Ϊԭ�����������ϵ�µ�ת�����󣨲²�������Ķ���
    float transformTobeMapped[6];//��ǰ֡�ľ���������λ�ˣ����transformAssociateToMap����
    // ���mapping֮ǰ��Odometry�������������ϵ��ת������ע����Ƶ������һ����transformSumһ����
    float transformBefMapped[6];//���ڴ����̼Ʒ������ĵ�ǰ֡��λ��
    float transformAftMapped[6]; // ���mapping֮��ľ���mapping΢��֮���λ��


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

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;//��ǰ֡��������ؼ�֡����������������ϵ�µ�λ�ˣ�����������ϵ����ǰ֡�����λ�˱仯
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

public:

    

    mapOptimization():
        nh("~")
    {
        // ���ڱջ�ͼ�Ż��Ĳ������ã�ʹ��gtsam��
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
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);//������transformSum
        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

        // �����˲�ʱ���������ش�СΪ0.2m/0.4m������,����ĵ�λΪm
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

        // matA1Ϊ��Ե������Э�������
        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        // matA1������ֵ
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        // matA1��������������Ӧ��matD1�洢
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

    // ������ת�Ƶ���������ϵ��,�õ������ڽ�ͼ��Lidar���꣬���޸���transformTobeMapped��ֵ
    //ʹ��transformSum��transformAftMapped��transformBefMapped����transformTobeMapped
    //�����Ѿ�֪������һ���ؼ�֡��ͼ�Ż�֮���λ�ˣ�������̼Ƶ�λ�ˣ���֪���˵�ǰ֡��̼Ƶ�λ��
    //�õ���ǰ֡�������λ��=T_��һ���ؼ�֡��ͼ�Ż�֮���λ��*(T_��ǰ֡��̼Ƶ�λ��.invserse()*T_��һ���ؼ�֡��̼Ƶ�λ��)
    void transformAssociateToMap()
    {   
    	//���ϸ���̼�����������ϵ�µ�λ�ñ任����ǰ֡������ϵ��
    	//R_z.inverse()*R_x.inverse()*R_y.inverse()(t-t_sum)
	    //��ΪtransformSum�Ǵ���������ϵ�ƶ�����ǰ֡����ϵ��λ�ˣ�transformBefMapped[3-5]��һ�����̼�����������ϵ�µ�λ��
	    //Ϊ�˵õ���һ֡����̼��ڵ�ǰ֡��λ���� transformIncre = T_sum.inverse()*p_befmapped
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;//��һ֡����̼��ڵ�ǰ֡����ϵ�µ�λ��
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

	//ʹ��transformSum����transformBefMapped��ʹ��transformTobeMapped����transformAftMapped
    void transformUpdate()
    {
		if (imuPointerLast >= 0) //��ʹ��imu�������������
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
		    transformBefMapped[i] = transformSum[i];//���cpp�ļ��о����������transformBefMapped��ֵ
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

	//
    void updatePointAssociateToMapSinCos()
	{
        // ����ǰ���roll,pitch,yaw��sin��cosֵ
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
    //����ǰ֡�������任����������ϵ��
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        // ����6���ɶȵı任���Ƚ�����ת��Ȼ����ƽ��
        // ��Ҫ��������任�����ֲ�����ת����ȫ��������ȥ	

        // ����z����ת
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

        // �������Y����ת��Ȼ�����ƽ��
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

	//R_y*R_x*R_z*p+t,��kʱ�̵ĵ��Ʊ任����ͼ����ϵ��
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
	{

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

		// ����ϵ�任����תrpy��
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

     //������transformSum
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
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];//�������߷ǳ�ʡ��û���Լ�����һ���������Ͷ���ֱ��ʹ�õ�ros���е���̼�������������������λ��
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);//����һ��topic = aft_mapped_to_init

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);//����һ��tf�� camera_init=frame_id, child_frame_id=aft_mapped
    }

	//
    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0)//topic name = key_pose_origin,Ĭ�ϲ������������
		{
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0)//topic name = recent_cloud,Ĭ�ϲ������������
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
        // ͨ��KDTree�������������
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
          globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // ��globalMapKeyPoses�����²���
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i)
		{
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // ��globalMapKeyFrames�����²���
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

        ros::Rate rate(1);// 1����ִ��һ�˻ػ����
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    }


	//���������Ҫ�ǵõ���ǰ֡����������ϵ�µĵ���latestSurfKeyFrameCloud���ͻػ�֡ǰ��25֡����������ϵ�µĽ���������nearHistorySurfKeyFrameCloudDS
	//�������ж��Ƿ����˻ػ��������ػ��������Ǵ��ھ��뵱ǰ֡С��50�׵������ؼ�֡����������ؼ�֡��ʱ�������̼Ƶ�ʱ�������30��
    bool detectLoopClosure(){

        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();

        // ��Դ����ʱ��ʼ��
        // �ڻ�����������ǰ������
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
		//�����µĹؼ��ķ�Χ���������뵱ǰ֡С��50�ļ���
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        closestHistoryFrameID = -1;
		//���ڸ�����֡�����ػ�֡
        for (int i = 0; i < pointSearchIndLoop.size(); ++i)
		{
            int id = pointSearchIndLoop[i];
            // ����ʱ���ֵ����30��
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0)
			{
                closestHistoryFrameID = id;
                break;
            }
        }
        if (closestHistoryFrameID == -1){
            // �ҵ��ĵ�͵�ǰʱ����û�г���30���
            return false;
        }

        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        // ������ؼ�֡�Ľǵ��ƽ���任����ͼ����ϵ��
        *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        
        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());//�洢��������ؼ�֡��ƽ��ͽǵ��������������ϵ�µ�����
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
		{
            // intensity��С��0�ĵ�Ž�hahaCloud����
            // ��ʼ��ʱintensity��-1���˵���Щ��
            if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0)
			{
                hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
            }
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud   = *hahaCloud;

        // ���ػ�֡��ǰ��25��֡�ĵ��Ʊ任����ͼ����ϵ�µõ�nearHistorySurfKeyFrameCloud
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
		{
		    // Ҫ��closestHistoryFrameID + j��0��cloudKeyPoses3D->points.size()-1֮��,���ܳ�������
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
        }

        // �Իػ�֡�ĵ������ݽ��н������õ�nearHistorySurfKeyFrameCloudDS
        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);//����0.4�Ľ������˲�
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

        if (pubHistoryKeyFrames.getNumSubscribers() != 0)//Ĭ�ϲ������������
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

            if (detectLoopClosure() == true)//�ǳ���Ҫ�ĺ���!!!!!!!!!!! ������???????????????????????????????????
			{
                potentialLoopFlag = true;
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false)
                return;
        }

        potentialLoopFlag = false;
        //��ǰ֡�ĵ��ƺͻػ�֡�Ľ��������ƽ���icpƥ�� 
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);// ����RANSAC���д���
        icp.setInputSource(latestSurfKeyFrameCloud);//����ؼ�֡�ĵ���
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);//�ջ�֡�Ľ���������
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // Ϊʲôƥ�������ֱ�ӷ���???�����ߴ�������̫��
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)// historyKeyframeFitnessScore = 0.3;
            return;

        
        if (pubIcpKeyFrames.getNumSubscribers() != 0)//Ĭ�ϲ������������
		{
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
			// icp.getFinalTransformation()�ķ���ֵ��Eigen::Matrix<Scalar, 4, 4>
            pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }   
        // �����ڵ���icp����������������һ����Χ�ڽ���
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation();//�õ���ǰ֡�ͱջ�֡�����λ��
		// �õ�ƽ�ƺ���ת�ĽǶ�
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);//���֡��λ��
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;//�������λ�˺ͱջ�֡��λ�˵õ����֡���������
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));//�õ�����֮�������ؼ�֡��gtsamλ��
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);//�õ��ջ�֡��gtsamλ��
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        //ʹ��gtsam��pgo
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

    //�����ڽ��ؼ�֡����Ϣ�������ٽ��ؼ�֡�����ڽ��ĵ�������:laserCloudCornerFromMapDS��laserCloudSurfFromMapDS
    void extractSurroundingKeyFrames()
    {

        if (cloudKeyPoses3D->points.empty() == true)
            return;	

       
		if (loopClosureEnableFlag == true)//Ĭ�ϲ������������
		{
            /*�������ע�ͷ����Ķ�����
            // recentCornerCloudKeyFrames����ĵ�������̫�٣�����պ����������µĵ���ֱ��������
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum)//surroundingKeyframeSearchNum = 50
			{
                recentCornerCloudKeyFrames. clear();
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size();
                for (int i = numPoses-1; i >= 0; --i)
				{
                    // cloudKeyPoses3D��intensity�д��������ֵ?
                    // ���������ֵ��1��ʼ��ţ�
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);//����ȫ�ֱ���
                    // ��������õ��ı任thisTransformation����cornerCloudKeyFrames��surfCloudKeyFrames��surfCloudKeyFrames
                    // ��������任
                    recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)//surroundingKeyframeSearchNum = 50
                        break;
                }


            }else{
                /
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1)//��ʾ�ؼ�֡�����˸���
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
                // ����pcl::PointXYZI���?
                // ע�������recentOutlierCloudKeyFramesҲ���뵽��laserCloudSurfFromMap
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
            }
            */
		}else{

            // �����ⲿ����û�бջ��Ĵ��룬Ĭ����û�лػ��Ĵ���
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
            // ���а뾶surroundingKeyframeSearchRadius�ڵ���������=50��
            //�����йؼ�֡��λ�˼������������뵱ǰ֡С��50�׵ļ��ϣ�������Щλ�˼��ϱ��浽surroundingKeyPoses����surroundingKeyPosesDS��surroundingKeyPoses�Ľ�����
			kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
			kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
			for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
			downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);//����1.0�Ľ�����
			downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();

			//������µĹؼ�֡���룬��ô���Ǿ���Ҫ����һ�����¹ؼ�֡��Χ�ĵ�ͼ��Ϣ������֡��Ϣ��
			//����˼·����:��֮ǰ������֡�������Ƿ��о�������֡50�׵Ĺؼ�֡�����û����֮ǰ������֡�е����֡ɾ��
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
			{
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j)
				{
                    // ˫��ѭ�������϶Ա�surroundingExistingKeyPosesID[i]��surroundingKeyPosesDS�ĵ��index
                    // ����ܹ��ҵ�һ���ģ�˵��������ͬ�Ĺؼ���(��ΪsurroundingKeyPosesDS��cloudKeyPoses3D��ɸѡ����)
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
					{
                        existingFlag = true;
                        break;
                    }
                }
				
                if (existingFlag == false)
				{
                    // ���surroundingExistingKeyPosesID[i]�Ա���һ�ֵ��Ѿ����ڵĹؼ�λ�˵�������intensity����ľ���size()��
                    // û���ҵ���ͬ�Ĺؼ��㣬��ô�������ӵ�ǰ������ɾ��
                    // ����Ļ���existingFlagΪtrue���ùؼ���ͽ������ڶ�����
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }

          
            //�ھ������¹ؼ�֡50�׵������ؼ�֡�������Ƿ���֮ǰ����֡����Ϣ�����û����������֡�������Ϣ
            for (int i = 0; i < numSurroundingPosesDS; ++i) 
			{
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter)
				{
                    // ��surroundingExistingKeyPosesID��û�ж�Ӧ�ĵ�Ž�һ��������
                    // �������ר�Ŵ����Χ���ڵĹؼ�֡�����Ǻ�surroundingExistingKeyPosesID�ĵ�û�ж�Ӧ�ģ�Ҳ�����µĵ�
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
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));//ĳ���ؼ�֡��Ӧ�������㾭���任ѹ�뵽��Χ�Ļ�����ȥ
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) 
			{
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];//���߳�ÿ�ζ��ὫlaserCloudCornerFromMap laserCloudSurfFromMap����
				
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
		}//else����

        // ���������²���
        // ������������laserCloudCornerFromMapDS��laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();//0.2���²���

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }//extractSurroundingKeyFrames��������

	//�������ƽ��㣬�ǵ��outlier���н�����
    void downsampleCurrentScan(){

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);//�ϸ��ڵ㷢������
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();//����0.2�Ľ�����

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);//�ϸ��ڵ㷢������
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();//����0.4�Ľ�����

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);//�ϸ��ڵ㷢������
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();//����0.4�Ľ�����

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);//����0.4�Ľ�������laserCloudSurfTotalLastDS = (�����ƽ��㽵����+�����outlier������)�ٽ���һ�ν�����
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

    //����ǰ֡�ĵ����ͼƥ��ĵ���ƥ�䣬Ȼ��������ʹ��ߵķ���
    //����ĵ�������û����!!!!!!!!!!!!
    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();
		//��������Ľ������Ľǵ�
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) //�����������һ�����forѭ�� 
		{
            pointOri = laserCloudCornerLastDS->points[i];//��ǰ֡�Ľ�����֮��Ľǵ�����
            
            // ����ǰ֡�Ľǵ����ݱ任����������ϵ��
            pointAssociateToMap(&pointOri, &pointSel);

            // ������ؼ�֡��Χ��ͼ�Ľ������ǵ��������뵱ǰ֡����Ľǵ��������5����
            //kdtreeCornerFromMap�����������laserCloudCornerFromMapDS=���ͼƥ��Ľǵ�
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            // ֻ�е���Զ���Ǹ������ľ���pointSearchSqDis[4]С��1mʱ�Ž�������ļ���
            // ���²��ֵļ������ڼ���㼯��Э�������Zhang Ji�����������ᵽ�ⲿ��
            if (pointSearchSqDis[4] < 1.0) 
			{
                // ����5��������ƽ��ֵ
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) 
				{
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // �����������matA1=[ax,ay,az]^t*[ax,ay,az]
                // ��׼ȷ��˵Ӧ��������Э����matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) 
				{
                    // ax�������x-cx,��ʾ��ֵ��ÿ��ʵ��ֵ�Ĳ�ֵ����ȡ5��֮���ٴ�ȡƽ�����õ�matA1
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

                // �������������ֵ����������
                // ����ֵ��matD1������������matV1��
                cv::eigen(matA1, matD1, matV1);

                // ��Ե����ϴ�����ֵ���Ӧ���������������Ե�ߵķ���һ����С������
                // ������һ������ڼ���㵽��Ե�ľ��룬���ͨ��ϵ��s���ж��Ƿ����ܽ�
                // �������ܽ�����Ϊ������ڱ�Ե�ϣ���Ҫ�ŵ�laserCloudOri��
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) //�����Ǹ�����ֵ���������ĵڶ��������ֵ����ʾֱ�߱�������
				{

					//��x0��x1��x2������ֱ���ϵľ���
                    float x0 = pointSel.x;//��ǰ֡�ϵĽǵ�
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
					//����Ҫ�ǳ�ע���ˣ����������ͼƥ��Ľǵ����ҵ���5�����뵱ǰ�������������
					//�����loam�����������˸��Ż���û�м����������������ǵõ��������㣬Ȼ��������������һ��ֱ��Ȼ����������ĵ㵽���ֱ�ߵľ���
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);//һ��Ҫע������ʹ�õ�������ֵ���е������Ĺ���
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // ���������[(x0-x1),(y0-y1),(z0-z1)]��[(x0-x2),(y0-y2),(z0-z2)]��˵õ���������ģ��
                    // ���ģ������0.2*V1[0]�͵�[x0,y0,z0]���ɵ�ƽ���ı��ε����
                    // ��Ϊ[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // l12��ʾ����0.2*(||V1[0]||)
                    // Ҳ����ƽ���ı���һ���׵ĳ���
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // ���˽��[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // LLL=[la,lb,lc]��0.2*V1[0]�������ϵĵ�λ��������||LLL||=1��
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                    
                    // �����pointSel��ֱ�ߵľ���
                    // ������Ҫ�ر�˵������ld2������ǵ�pointSel������[cx,cy,cz]�ķ�������ֱ�ߵľ���
                    float ld2 = a012 / l12;

                    // ������������״̬�Ļ���ld2Ӧ��Ϊ0����ʾ����ֱ����
                    // ������״̬s=1��
                    float s = 1 - 0.9 * fabs(ld2);
                    
                    // coeff����ϵ������˼
                    // coff���ڱ������ķ�������
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;

                    // intensity�����Ϲ�����һ���˺�����ld2Խ�ӽ���1������Խ��
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;
                    
                    // ���Ծ�Ӧ����Ϊ������Ǳ�Ե��
                    // s>0.1 Ҳ����Ҫ��㵽ֱ�ߵľ���ld2ҪС��1m
                    // sԽ��˵��ld2ԽС(���Ե��Խ��)��������˵����pointOri��ֱ����
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

	//����ĵ�������û����!!!!!!!!!!!!!
	//����㵽ƽ��ľ����ƽ�淨����
    void surfOptimization(int iterCount)
	{
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++)//�����������������forѭ��
		{
            pointOri = laserCloudSurfTotalLastDS->points[i];//��ǰ֡��(�����ƽ��㽵����+�����outlier������)�ٽ���һ�ν����������������Ϊ��ͼƥ���ƽ���
            pointAssociateToMap(&pointOri, &pointSel); 
			//�����ͼƥ��ĵ�������5�����뵱ǰ֡�ĵ����
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0)//��Զ�ĵ�ҪС��1��
			{
                for (int j = 0; j < 5; j++) 
				{
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // matB0��һ��5x1�ľ���
                // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                // matX0��3x1�ľ���
                // ��ⷽ��matA0*matX0=matB0
                // ��ʽ��ʵ��������matA0�еĵ㹹�ɵ�ƽ��ķ�����matX0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                // [pa,pb,pc,pd]=[matX0,pd]
                // ��������£�������planeValid�ж���������Ӧ����
                // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                // ����pd����Ϊ1
                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                // ��[pa,pb,pc,pd]���е�λ��
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // �����ٴμ��ƽ���Ƿ�����Чƽ��
                bool planeValid = true;
                for (int j = 0; j < 5; j++) 
				{
					//����ƽ��ķ���Ax+By+Cz+D = 0
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
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;//�㵽ƽ��ľ���

                    // ���沿����������[pa,pb,pc,pd]��pointSel�ļн�����ֵ(����sqrt����ʵ����������ֵ)
                    // ����н�����ֵԽСԽ�ã�ԽС֤�������[pa,pb,pc,pd]��ƽ��Խ��ֱ
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // �ж��Ƿ��Ǻϸ�ƽ�棬�Ǿͼ���laserCloudOri
                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // �ⲿ�ֵĴ����ǻ��ڸ�˹ţ�ٷ����Ż�������zhang ji�������ᵽ�Ļ���L-M���Ż�����
    // �ⲿ�ֵĴ���ʹ����ת�����ŷ�����󵼣��Ż�ŷ���ǣ�����zhang ji�������ᵽ��ʹ��angle-axis���Ż�
    bool LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);//��ǰ֡�ľ���������λ�ˣ����transformAssociateToMap����
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();//ƥ���ϵĵ������
        // laser cloud original ����̫�٣����������ѭ��
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

        for (int i = 0; i < laserCloudSelNum; i++)//����ƥ���ϵĵ�
		{
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            // ���ſ˱Ⱦ����е�Ԫ�أ�����d��roll�Ƕȵ�ƫ������d(d)/d(roll)
            // ����ϸ����ѧ�Ƶ��ο�wykxwyc.github.io
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            // ͬ�ϣ������Ƕ�pitch��ƫ����
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;


            /*
            ����㵽ֱ�ߵľ���ʱ��coeff��ʾ������������
            [la,lb,lc]��ʾ���ǵ㵽ֱ�ߵĴ�ֱ���߷���s�ǳ���
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            ����㵽ƽ��ľ���ʱ��coeff��ʾ����
            [pa,pb,pc]��ʾ������ƽ��ķ�������s���ߵĳ���
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */
            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;

            // �ⲿ�����ſ˱Ⱦ����о����ƽ�Ƶ�ƫ��
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;

            // �в���
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // ��������matAת������matAt
        // �Ƚ��м��㣬�Ա��ں�ߵ��� cv::solve���
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // ���ø�˹ţ�ٷ�������⣬
        // ��˹ţ�ٷ���ԭ����J^(T)*J * delta(x) = -J*f(x)
        // J���ſ˱Ⱦ���������A��f(x)���Ż�Ŀ�꣬������-B(�����ڸ�B��ֵʱ��ͷŽ�ȥ��)
        // ͨ��QR�ֽ�ķ�ʽ�����matAtA*matX=matAtB���õ���matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 ˵���ǵ�һ�ε�������Ҫ��ʼ��
        if (iterCount == 0) 
		{
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // �Խ��Ƶ�Hessian����������ֵ������������
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

        // ��ת����ƽ�����㹻С��ֹͣ��ε�������
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization()
	{
        //���ͼƥ��Ľǵ㽵����֮������Ҫ����10�����ͼƥ���ƽ��㽵����֮������Ҫ����100
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) 
		{

            
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);//����ؼ�֡��Χ��ͼ�Ľǵ�
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);//����ؼ�֡����Χ��ƽ����outlier

            for (int iterCount = 0; iterCount < 10; iterCount++) 
			{
                // ��forѭ�����Ƶ���������������10��
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);//����ǰ֡�ĵ����ͼƥ��ĵ���ƥ�䣬Ȼ��������ʹ��ߵķ���.ע������ĵ�������û����!!!!!!!!!!!!!!
                surfOptimization(iterCount);//����㵽ƽ��ľ����ƽ�淨����

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // ��������������ص�ת�ƾ���
            transformUpdate();//ʹ��transformSum����transformBefMapped��ʹ��transformTobeMapped����transformAftMapped
        }
    }

    //�жϵ�ǰ֡�ǲ��ǹؼ�֡�������ƶ��ľ�������ж�
    //����ǹؼ�֡����һ���ؼ�֡��λ�˺͵�ǰ�ؼ�֡��λ�˼��뵽pgo�У�������һ���Ż���
    //���Ż���Ĺؼ�֡��λ�˸���transformAftMapped
    void saveKeyFramesAndFactor(){

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3)//�������ƶ���0.3�ײŻ�����µĹؼ�֡
        {
            saveThisKeyFrame = false;
        }

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;

        if (cloudKeyPoses3D->points.empty())//����ǵ�һ���ؼ�֡
		{
            // NonlinearFactorGraph����һ��PriorFactor����
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            // initialEstimate������������Values,��ʵ����һ��map��������0��Ӧ��ֵ���汣����һ��Pose3
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transformLast[i] = transformTobeMapped[i];
        }
        else
		{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));//Ӧ������һ���ؼ�֡��λ��
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));//��ǰ֡�Ĺؼ�֡
			
            // ���캯��ԭ��:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
																		 Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));//����ǰ֡�ĳ�ֵ����gtsam״̬��
        }

        // initialEstimate�Ǽӵ�ϵͳ�е��±����ĳ�ʼ��
        isam->update(gtSAMgraph, initialEstimate);
        // update ����Ϊʲô��Ҫ�������Σ�
        isam->update();

		// ɾ������?
        gtSAMgraph.resize(0);
		initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);//�õ�����һ֡��λ��

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
        cloudKeyPoses6D->push_back(thisPose6D);//���¹ؼ�֡��λ��

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

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);//��ÿ���ؼ�֡�ļ���㱣��
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    }

	//��������˻ػ�����������еĹؼ�֡������pgo�Ż��Ľ���������еĹؼ�֡
    void correctPoses()
    {
	    if (aLoopIsClosed == true)//�����˻ػ�
		{
	            recentCornerCloudKeyFrames. clear();
	            recentSurfCloudKeyFrames.   clear();
	            recentOutlierCloudKeyFrames.clear();

	            int numPoses = isamCurrentEstimate.size();
				for (int i = 0; i < numPoses; ++i)//�������еĹؼ�֡������pgo�Ż��Ľ���������еĹؼ�֡
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

            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval)//Ĭ����0.3
	    	{

                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();//��ǰ֡�������λ��=T_��һ���ؼ�֡��ͼ�Ż�֮���λ��*(T_��ǰ֡��̼Ƶ�λ��.invserse()*T_��һ���ؼ�֡��̼Ƶ�λ��)

                extractSurroundingKeyFrames();//�õ����ͼƥ��ļ���ĵ��ƽ�ĵ�:laserCloudCornerFromMapDS��laserCloudSurfFromMapDS

                downsampleCurrentScan();//�������ƽ��㣬�ǵ��outlier���н�����

                // ��ǰɨ����б�Ե�Ż���ͼ�Ż��Լ�����LM�Ż��Ĺ���
                //������Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                scan2MapOptimization();
				
                //�жϵ�ǰ֡�ǲ��ǹؼ�֡�������ƶ��ľ�������ж�
    			//����ǹؼ�֡����һ���ؼ�֡��λ�˺͵�ǰ�ؼ�֡��λ�˼��뵽pgo�У�������һ���Ż���
    			//���Ż���Ĺؼ�֡��λ�˸���transformAftMapped
                saveKeyFramesAndFactor();

                correctPoses();//��������˻ػ�����������еĹؼ�֡������pgo�Ż��Ľ���������еĹؼ�֡

                publishTF();//���Ż��ĵ�ǰλ��ͨ��tpopic ������ȥ��tf��������ȥ

                publishKeyPosesAndFrames();//�������ʲô��û��

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

    // ���бջ������ջ��Ĺ���
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
	
    // ���߳��н��еĹ�����publishGlobalMap(),�����ݷ�����ros�У����ӻ�
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();//������Ҫ�ĺ���!!!!!!!!!!!!!!!!

        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}



