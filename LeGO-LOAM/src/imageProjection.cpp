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

#include "utility.h"


class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;//���յ��ĵ�������

    pcl::PointCloud<PointType>::Ptr fullCloud;//PointType = PointXYZI�������ߵ�˳��Ե������ݽ��д洢
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;//��Ӧtopic segmented_cloud����������
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;//��Ӧtopic outlier_cloud����������

    PointType nanPoint;

    cv::Mat rangeMat;//�е����к��Ǹ����ǵ���ţ��е����к��ǵ����ǵ���ţ������ǵ㵽����ľ��룬�����16�߼����Ǿ�һ��ֻ��16��
    cv::Mat labelMat;.//��ʼ����Ϊ0
    cv::Mat groundMat;
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg;//��Ӧtopic name = segmented_cloud_info�ķ�������
    std_msgs::Header cloudHeader;//���յ��ĵ�������

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

    uint16_t *allPushedIndX;
    uint16_t *allPushedIndY;

    uint16_t *queueIndX;
    uint16_t *queueIndY;

public:
    ImageProjection():
        nh("~"){
        // ��������velodyne�״�������topic ("/velodyne_points")
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &ImageProjection::cloudHandler, this);//�ǳ���Ҫ�ĺ���!!!!!!!!!
        
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

	// ��ʼ����������Լ������ڴ�
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

		// labelComponents�������õ����������
		// �þ���������ĳ�������������4���ڽӵ�
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

	// ��ʼ��/���ø����������
    void resetParameters()
    {
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));//Horizon_SCAN=1800
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}
	
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        // ��ROS�е�sensor_msgs::PointCloud2ConstPtr����ת����pcl���ƿ�ָ��
        cloudHeader = laserCloudMsg->header;
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        copyPointCloud(laserCloudMsg);
        findStartEndAngle();//����˳ʱ���˳�������Ƶ���ʼ�Ƕȣ������ǶȺ���ת���ĽǶȡ�
        groundRemoval();//������㣬����ĵ㲻�������
        projectPointCloud();
        cloudSegmentation();//�Էǵ������о��࣬Ȼ�󽫾����ĵ��1/5�ĵ���㷢����ȥ
        publishCloud();
        resetParameters();
    }

	//����˳ʱ���˳�������Ƶ���ʼ�Ƕȣ������ǶȺ���ת���ĽǶȡ�
    void findStartEndAngle(){
        // �״�����ϵ����->X,ǰ->Y,��->Z
        // �״��ڲ���תɨ�跽��Z�ḩ��������˳ʱ�뷽��Z�����ֶ�����

        //ע����segMsg��Ӧsegmented_cloud_info topic����������
        // ��Ϊ�ڲ��״���ת����ԭ������atan2(..)����ǰ����Ҫ��һ������
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        // ������仰�������߿���д���ˣ�laserCloudIn->points.size() - 2Ӧ����laserCloudIn->points.size() - 1
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                         laserCloudIn->points[laserCloudIn->points.size() - 2].x) + 2 * M_PI;
		// ��ʼ�ͽ����ĽǶȲ�һ���Ƕ��٣�
		// һ��velodyne �״����ݰ�ת���ĽǶȶ��
        // �״�һ���������һȦ�����ݣ����ԽǶȲ�һ����2*PI��һ�����ݰ�ת��360��
		// segMsg.endOrientation - segMsg.startOrientation��ΧΪ(0,4PI)
        // ����ǶȲ����3Pi��С��Pi��˵���ǶȲ������⣬���е�����
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
		// segMsg.orientationDiff�ķ�ΧΪ(PI,3PI),һȦ��СΪ2PI��Ӧ����2PI����
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

	//���ݵ�����������rangMat(�����ͼ)�Ͱ����ߵ�˳��洢�����ݽṹfullInfoCloud,
	//�м��мǣ�����Խ���Ӧ���Ǹߴ������ɨ�赽�ĵ㣬fullInfoCloudҲ�Ǵ���������Ǹ������߿�ʼ���д洢��
    void projectPointCloud()
    {
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i)
		{

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // ������ֱ�����ϵĽǶȣ��״�ĵڼ��ߣ�
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
			
            // rowIdn������õ㼤���״�����ֱ�����ϵڼ��ߵ�
			// �������ϼ�����-15�ȼ�Ϊ��ʼ�ߣ���0�ߣ�һ��16��(N_SCAN=16)
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;//ang_bottom=16.ang_res_y=2
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // atan2(y,x)�����ķ���ֵ��Χ(-PI,PI],��ʾ�븴��x+yi�ķ���
            // �·��Ƕ�atan2(..)������x��y��λ�ã����������y��������ļнǴ�С(����y=x���ԳƱ任)
            // ���������״�����ϵ������������ǰ���ļнǴ�С
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

			// round����������������ȡ��
			// ���ȷ�����Ǽ�ȥ180��???  ����
			// �״�ˮƽ������ĳ���ǶȺ�ˮƽ�ڼ��ߵĹ�����ϵ???��ϵ���£�
			// horizonAngle:(-PI,PI],columnIdn:[H/4,5H/4]-->[0,H] (H:Horizon_SCAN)
			// �����ǰ�����ϵ��z����ת,��columnIdn�������Ա任
			// x+==>Horizon_SCAN/2,x-==>Horizon_SCAN
			// y+==>Horizon_SCAN*3/4,y-==>Horizon_SCAN*5/4,Horizon_SCAN/4
            //
            //          3/4*H
            //          | y+
            //          |
            // (x-)H---------->H/2 (x+)
            //          |
            //          | y-
            //    5/4*H   H/4
            //
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;//ang_res_x=0.2,Horizon_SCAN=1800
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            // ��������columnIdn -= Horizon_SCAN�ı任���columnIdn�ֲ���
            //          3/4*H
            //          | y+
            //     H    |
            // (x-)---------->H/2 (x+)
            //     0    |
            //          | y-
            //         H/4
            //
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            rangeMat.at<float>(rowIdn, columnIdn) = range;

			// columnIdn:[0,H] (H:Horizon_SCAN)==>[0,1800]
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;

            fullInfoCloud->points[index].intensity = range;
        }
    }

    //����groudMat��labelMat��groundCloud
    //groudMat��С16*1800,���ݵ���1��ʾΪ�����,����-1��ʾ�����ľ������ֵ��Ч����ǰ��һ����Ĳ���ֵ��Ч
    //labelMat��С16*1800, ���ݵ���-1��ʾΪ��������û�в⵽����
    //groundCloudΪ����㣬pcl���ݽṹ
    void groundRemoval()
    {
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        //�����н������ݵı���
        for (size_t j = 0; j < Horizon_SCAN; ++j)
		{
           
            for (size_t i = 0; i < groundScanInd; ++i)//�����������7���ߵ����ݣ� groundScanInd=7
			{

                lowerInd = j + ( i )*Horizon_SCAN;//j���кã�i���к�
                upperInd = j + (i+1)*Horizon_SCAN;

                // ��ʼ����ʱ����nanPoint.intensity = -1 ���
                // ����-1 ֤���ǿյ�nanPoint
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1)
                {
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }

				// ����������֮����XYZλ�õõ�����֮��ĸ�����
				// �����������10�����ڣ����ж�(i,j)Ϊ�����,groundMat[i][j]=1
				// �������ǵ���㣬���к�������
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10)//��ʾС��10�ȣ�sensorMountAngle = 0
				{
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

		// �ҵ����е��еĵ������߾���ΪFLT_MAX(rangeMat�ĳ�ʼֵ)�ĵ㣬�������Ǳ��Ϊ-1
        for (size_t i = 0; i < N_SCAN; ++i)//N_SCAN = 16
		{
            for (size_t j = 0; j < Horizon_SCAN; ++j)//Horizon_SCAN =1800
			{
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX)//����ǵ���ĵ���������Ĳ���ֵ��Ч�����ǲ����������о���
				{
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

		// ����нڵ㶩��groundCloud����ô����Ҫ�ѵ���㷢������
		// ����ʵ�ֹ��̣��ѵ�ŵ�groundCloud������ȥ
        if (pubGroundCloud.getNumSubscribers() != 0)
		{
            for (size_t i = 0; i <= groundScanInd; ++i)
			{
                for (size_t j = 0; j < Horizon_SCAN; ++j)
				{
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);//��Ӧ��topic name = ground_cloud
                }
            }
        }
    }

	//
    void cloudSegmentation(){
        //�����н��б���
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)//���Ե���ĵ�Ͳ���ֵ��Ч�ĵ����
                    labelComponents(i, j);//������Ҫ�ĺ���

        int sizeOfSegCloud = 0;
		//�����е�˳Ѷ���б���
        for (size_t i = 0; i < N_SCAN; ++i) 
		{
			
			// segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// ��ʾ��i�ߵĵ�����ʼ���к���ֹ����
			// �Կ�ʼ�ߺ�ĵ�6��Ϊ��ʼ���Խ�����ǰ�ĵ�6��Ϊ����
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;//segMsg��Ӧsegmented_cloud_info����������

            for (size_t j = 0; j < Horizon_SCAN; ++j) 
			{
				//�������ǶԴ�������˸�д�������Ķ�
                
					// labelMat��ֵΪ999999��ʾ���������Ϊ������������30���������ĵ�
					//�������Ч��4/5�ĵ���㣬�������������
                    if ((labelMat.at<int>(i,j) == 999999)&&(i > groundScanInd && j % 5 == 0))
					{
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);//��Ӧoutlier_cloud topic��������Ϣ��ֻ����1/5��outlier
                            continue;
                    }
					else if(((labelMat.at<int>(i,j) == 999999)&&(i > groundScanInd && j % 5 != 0)))||((groundMat.at<int8_t>(i,j) == 1)&&(j%5!=0 && j>5 && j<Horizon_SCAN-5))
					{
						    continue;
					}
					//��������������������ʾ�� �Ѿ�����ĵ�����ǵ���㡣ע��!����ĵ��в����������
					if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1)
			     	{
						segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
	                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
	                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
	                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);//��Ӧsegmented_cloud topic��������Ϣ
	                    ++sizeOfSegCloud;
                   }
            }

            // �Խ�����ǰ�ĵ�5��Ϊ����
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }

		
		// ��ô�ѵ������ݱ��浽segmentedCloudPure��ȥ
        if (pubSegmentedCloudPure.getNumSubscribers() != 0)
		{
            for (size_t i = 0; i < N_SCAN; ++i)
			{
                for (size_t j = 0; j < Horizon_SCAN; ++j)
				{
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999)// ��Ҫѡ���ǵ����(labelMat[i][j]!=-1)��û�������ĵ�
					{
					    //segmentedCloudPure����������pcl���ݽṹ
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

	//��Ҫ��Ϊ�˸���labelMat�������ݺͱ�ǩ�ĸ���labelCount
	/*labelMat1�� labelMat.at(i,j) = 0����ʼֵ
	2�� labelMat.at(i,j) = -1����Ч�㣻
	3��labelMat.at(thisIndX, thisIndY) = labelCount������ĵ㣻
	4��labelMat.at(allPushedIndX[i], allPushedIndY[i]) = 999999����Ҫ�����ĵ㣬��������30��
	*/
    void labelComponents(int row, int col){
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        // ��׼�Ŀ�������㷨
        // BFS����������(row��col)Ϊ������������ɢ��
        // �ж�(row,col)�Ƿ������ƽ����һ��
        while(queueSize > 0)
		{
            fromIndX = queueIndX[queueStartInd];//Ҫ��ע�������ͼ���е����
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
			// labelCount�ĳ�ʼֵΪ1����������
			//labelCount��һ��ȫ�ֵı���
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;

			// neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
			// ������[fromIndX,fromIndY]���ϵ��ĸ��ڵ�
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
			{

                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;

                // �Ǹ���״��ͼƬ��������ͨ
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

				// �����[thisIndX,thisIndY]�Ѿ���ǹ�
				// labelMat�У�-1������Ч�㣬0����δ���б�ǹ�������Ϊ�����ı��
				// ���labelMat�Ѿ����Ϊ�����������Ѿ�������ɣ�����Ҫ�ٴζԸõ����
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)//�������������Ҫ��������bfs�������Ĺ����в����ظ�����
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

				// alpha����Ƕȷֱ��ʣ�
				// X�����ϽǶȷֱ�����segmentAlphaX(rad)
				// Y�����ϽǶȷֱ�����segmentAlphaY(rad)
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;//ˮƽ����Ƕȷֱ��ʣ���λ��rad
                else
                    alpha = segmentAlphaY;//��ֱ����Ƕȷֱ��ʣ���λ��rad

				// ͨ������Ĺ�ʽ����������֮���Ƿ���ƽ������
				// atan2(y,x)��ֵԽ��d1��d2֮��Ĳ��ԽС,Խƽ̹
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta)// segmentTheta=1.0472rad=60��
				{
					
					// �������Ƕȴ���60�ȣ���������Ǹ�ƽ��
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;//�����ֱ�����ϵĵ���

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;//��������++
                }
            }//forѭ������
        }//whileѭ������


        bool feasibleSegment = false;

		// ������೬��30���㣬ֱ�ӱ��Ϊһ�����þ��࣬labelCount��Ҫ����
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum)//segmentValidPointNum = 5
		{
			// ����������С��30���ڵ���5��ͳ����ֱ�����ϵľ������
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;

			// ��ֱ�����ϳ���3��Ҳ�������Ϊ��Ч����
            if (lineCount >= segmentValidLineNum)//segmentValidLineNum=3
                feasibleSegment = true;            
        }

        if (feasibleSegment == true)//��ʾ�������������Ч�����
		{
            ++labelCount;
        }else{
            for (size_t i = 0; i < allPushedIndSize; ++i)
			{
				// ���Ϊ999999������Ҫ�����ľ���ĵ㣬��Ϊ���ǵ�����С��30��
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    // ���������������
    void publishCloud(){
    	// ����cloud_msgs::cloud_info��Ϣ
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);//topic name ="segmented_cloud_info"

        sensor_msgs::PointCloud2 laserCloudTemp;

		// pubOutlierCloud�����������
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);//topic name = outlier_cloud

		// pubSegmentedCloud�����ֿ����
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);//topic name  = segmented_cloud

        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);//topic name = full_cloud_projected
        }

        if (pubGroundCloud.getNumSubscribers() != 0)
		{
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);//topic name = ground_cloud
        }

        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);//topic name = segmented_cloud_pure
        }

        if (pubFullInfoCloud.getNumSubscribers() != 0)
		{
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);//topic name  =full_cloud_info
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
