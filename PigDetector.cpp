#include "PigDetector.h"
#include "DirectionJudge.hpp"
#include <cmath>
#include <io.h>       //判定文件夹是否存在
#include <direct.h>   //创建文件夹
#include <algorithm>
#include <fstream>
#include <stdio.h>  //删除文件函数remove 需要此文件头
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace caffe::Frcnn;


//****************静态变量***********
//不同类别对应的框的颜色, 仔猪为类别0，红色； 大猪为类别1，黄色
std::map<unsigned, cv::Scalar> PigDetector::label_color = { { 1, Scalar(0, 0, 255) }, { 2, Scalar(0, 255, 255) } };
//目标检测模型
FRCNN_API::Detector* PigDetector::obj_detect = new FRCNN_API::Detector
(PIG_DETECT_MODEL_FOLD + "\\pigDetect.prototxt", PIG_DETECT_MODEL_FOLD + "\\pigDetect.caffemodel", PIG_DETECT_MODEL_FOLD + "\\pigDetect.json", true, true);


//************** 非静态成员函数部分 **********
PigDetector::PigDetector(const std::string& Writer_NO, unsigned camera_channel, const std::string& ip,
	const std::string& saved_fold)
	:CONTINUE_SECOND(10), K(5 * CONTINUE_SECOND / CAMERA_NUM), RATE(0.8), pigCounts(0), first_process(true),
	realFrames(0), hasBorn(false), CDWriter_NO(Writer_NO), CHANNEL_NO(camera_channel), IP(ip), SAVED_FOLD(saved_fold)
{
	static bool tag = true;
	if (tag)
	{
		tag = false;
		cout << endl;
		cout << "CONTINUE_SECOND: " << CONTINUE_SECOND << " s" << endl;
		cout << "CAMERA_NUM: " << CAMERA_NUM << " cameras" << endl;
		cout << "K: " << K << " frames" << endl;
		cout << endl;
	}
	//创建保存图片文件夹
	if (_access(SAVED_FOLD.c_str(), 0) == -1)   //文件夹不存在，则创建文件夹
		_mkdir(SAVED_FOLD.c_str());
	std::cout << "正在启动硬盘刻录机, 编号" << CDWriter_NO << ", 摄像头通道号" << CHANNEL_NO << ", 图片保存于: " << SAVED_FOLD << endl;
}

vector<BBox<float> > PigDetector::frameProcess(const cv::Mat& _frame)
{
	Mat frame = _frame.clone();
	//第一次处理需要检测出大猪的躺向
	if (first_process)
	{
		DirectionJudge judger(PIG_DIRECTION_MODEL_FOLD + "\\pigDirection_deploy.prototxt",
			PIG_DIRECTION_MODEL_FOLD + "\\pigDirection.caffemodel",
			PIG_DIRECTION_MODEL_FOLD + "\\pigDirection_mean.binaryproto");
		//出生区域方向，共6种， 0-1-2-3-4-5，分别代表左上，右上，左下，右下，正左，正右
		this->bornAreaDirection = judger.judge(frame);
		map<unsigned, string> m{ { 0, "左上" }, { 1, "右上" }, { 2, "左下" }, { 3, "右下" }, { 4, "正左" }, { 5, "正右" } };
		cout << "摄像头通道号" << CHANNEL_NO << ", 出生区域在" << m[bornAreaDirection] << "方" << endl;
		first_process = false;
	}
	vector<BBox<float> > wholeResults = PigDetector::obj_detect->predict(frame); //先检测所有类别
	//将仔猪框放在前面,大猪框放在后面, 返回第一个指向大猪框的迭代器
	auto iter = std::partition(wholeResults.begin(), wholeResults.end(), [](const BBox<float>& box){return box.id == 1; }); //划分算法，Lamdba表达式
	vector<BBox<float> > big_pig_boxes(iter, wholeResults.end());  //取出全部类别的框中属于母猪的框
	if (big_pig_boxes.size() > 0)    //因为可能没有检测到大猪，所以需要加个判断
	{
		this->big_pig_rect = bbox2Rect(big_pig_boxes[0]);
		this->setBornArea(frame.cols, frame.rows, big_pig_boxes[0]);
	}

	this->preResults = this->curResults;         //保存当前仔猪结果到上帧结果中
	this->curResults = { wholeResults.begin(), iter };    //取出全部类别的框中属于仔猪的框
	this->pairBBox(preResults, curResults);      //将当前帧与上一帧的结果进行匹配
	int detectNum = curResults.size();           // 当前帧检测出的框的个数

	// !!!!!!!!!!!!!!!!!!!!!!!
	pig_info.push_back(Pig_Info(frame, curResults, time(NULL))); //保存最新的 帧、检测结果、时间
	M[detectNum]++;    //将最新的目标框数目+1
	if (++realFrames > K)  //检测帧数+1, 如果已经储存了连续的K帧
	{
		realFrames = K;    //特殊处理，防止程序长时间运行导致的整数溢出
		//弹出最旧的帧，更新pig_info,M
		M[pig_info.front().boxes.size()]--;  //将最旧的目标框数目-1
		pig_info.pop_front();

		//*****************开始计算是否达到更新条件**********************
		unsigned tmp_pigCounts1 = computePigCountsByWay1();
		unsigned tmp_pigCounts2 = computePigCountsByWay2();
		unsigned tmp_max = std::max(tmp_pigCounts1, tmp_pigCounts2);  //二者较大者
		unsigned tmp_min = std::min(tmp_pigCounts1, tmp_pigCounts2);  //二者较小者
		assert((tmp_min >= pigCounts) || (tmp_max <= pigCounts));

		bool hasChanged = false;   //仔猪是否发生变化
		unsigned oldPigCounts;
		//判断仔猪是增加了，减少了，还是不变
		if (tmp_max > pigCounts || tmp_min < pigCounts)   //仔猪有变化
		{
			hasChanged = true;
			oldPigCounts = pigCounts;     //保存之前的仔猪数

			if (tmp_max > pigCounts)     //仔猪增加
			{
				pigCounts = tmp_max;   //更新pigCounts
				// 当仔猪数增加时，需要分清楚是第一只仔猪出生a，还是非第一只仔猪出生b，或者是召回c
				bool inBornArea = false;   //是否有仔猪在出生区域
				for (int ii = 0; ii < detectNum; ++ii)
				{
					if (isInBornArea(curResults[ii])){
						inBornArea = true;
						break;
					}
				}
				if (inBornArea)   //有猪出现在出生区域, 还要进一步判断是第一只仔猪出生a还是非第一只仔猪出生b，	
				{
					if (!hasBorn&&pigCounts == 1)  //只有当刚开始出生，且检测到仔猪只有一只，才判定为新出生的第一只
						this->status = 'a';
					else                       //否则都判定为第二只新出生
						this->status = 'b';
				}
				else  //不是新出生的仔猪，而是可能之前被遮挡，现在又出现的仔猪,即召回c
					this->status = 'c';
				this->hasBorn = true;   //已经开始分娩了
			}
			else        //仔猪减少
			{
				pigCounts = tmp_min;   //更新pigCounts
				this->status = 'd';     //消失
			}
		}
		if (hasChanged)
		{
			//根据状态，输出信息并保存图片(第一只出生a、非第一只出生b、召回c、消失d)
			cout << "硬盘刻录机编号" << CDWriter_NO << ", 摄像头通道号" << CHANNEL_NO << ": 仔猪数变化：" << oldPigCounts << " -> " << pigCounts;
			if (this->status == 'a')
				cout << ", 新出生(第一只)\n";
			else if (this->status == 'b')
				cout << ", 新出生(非第一只)\n";
			else if (this->status == 'c')
				cout << ", 召回\n";
			else
				cout << ", 消失\n";

			Pig_Info need_to_save;      //需要保存的那一帧
			BBox<float> bornArea_box;   //如果是有新出生的仔猪，则保存那个框在这里
			//找到目标框数目为pigCount且最旧的一帧
			for (auto iter = pig_info.begin(); iter != pig_info.end(); ++iter)
			{
				if (iter->boxes.size() == pigCounts)
				{
					if ((this->status == 'a' || this->status == 'b'))   //有小猪出生，还需要定位到在出生区域的帧
					{
						if (atleast_one_in_born_area(iter->boxes, bornArea_box))
						{
							need_to_save = *iter;
							break;
						}
					}
					else
					{
						need_to_save = *iter;
						break;
					}
				}
			}//end for	
			string name = CDWriter_NO + num2str(this->CHANNEL_NO) + getPigCounts2string() + this->status + getTime(need_to_save.t) + ".jpg";   //设备号+当前仔猪数+仔猪状态+当前时间+后缀名
			//保存那一帧，不包括目标框信息		
			//imwrite("C:\\Users\\Lab1060-i7\\Desktop\\母猪分娩智能监控\\PigImages\\imgs\\" + name, iter->img, { 1, 100 });
			//保存那一帧，且画边框信息
			drawRects(need_to_save.img, need_to_save.boxes, true);
			if (this->status == 'a' || this->status == 'b')  //如果是新出生的仔猪，需要画绿色框
			{
				rectangle(need_to_save.img, bbox2Rect(bornArea_box), Scalar(0, 255, 0), 2);
				putText(need_to_save.img, num2str(bornArea_box.order + 1), cv::Point(bornArea_box[0], bornArea_box[1]), 1, 1.0, Scalar(0, 255, 0), 2);
			}
			imwrite(SAVED_FOLD + "\\" + name, need_to_save.img, { 1, 100 });   //保存
		}//end if 
	}
	//!!!!!!!!!!!!!!!!
	return curResults;
}

//方案1求得的当前仔猪数量：如果连续K帧中有rate比例以上的目标框数为n，则判定当前仔猪数为n，否则不更新
unsigned PigDetector::computePigCountsByWay1()
{
	int count;   // 出现帧数
	//pig_info 出现最多的仔猪数量，以及对应出现的帧数
	int value = getMaxOfMap(M, &count);  //仔猪数
	return (count >= K*RATE) ? value : pigCounts;
}

//方案2求得的当前仔猪数量：讲连续的K帧检测出的目标框数的均值，作为当前仔猪数量
unsigned PigDetector::computePigCountsByWay2()
{
	float sum = 0.0;
	for (auto it = pig_info.begin(); it != pig_info.end(); ++it)
		sum += (it->boxes.size())*(it->weight);
	float mean = sum / pig_info.size();    //均值
	if ((mean <= pigCounts - 1) || (mean >= pigCounts + 1))
		return nearbyint(mean);
	return pigCounts;
}

bool PigDetector::isInBornArea(const BBox<float>& box){
	Rect rect(Point(box[0], box[1]), Point(Point(box[2], box[3])));
	return (rect&bornArea) == rect;
}
//是否至少有一只仔猪在出生区域，如果是，返回在出生区域的框
bool PigDetector::atleast_one_in_born_area(const std::vector<BBox<float> >& boxes, BBox<float>& box){
	vector<BBox<float> > in_born_area_boxes;
	for (int i = 0; i < boxes.size(); i++)
	{
		if (isInBornArea(boxes[i]))
			in_born_area_boxes.push_back(boxes[i]);
	}
	if (in_born_area_boxes.size() != 0)  //选择跟踪序号最大的返回
	{
		int max_order = -1;
		for (int i = 0; i<in_born_area_boxes.size(); ++i)
		{
			if (in_born_area_boxes[i].order>max_order)
			{
				max_order = in_born_area_boxes[i].order;
				box = in_born_area_boxes[i];
			}
		}
		return true;
	}
	return false;
}
//根据大猪位置设置出生区域
void PigDetector::setBornArea(const int WIDTH, const int HEIGHT, const BBox<float>& big_pig_box){
	int width = (big_pig_box[2] - big_pig_box[0]) / 8;  //以大猪框宽度的八分之一为小块
	int height = (big_pig_box[3] - big_pig_box[1]) / 4;  //以大猪框的高度四分之一为小块
	switch (bornAreaDirection)   //根据大猪屁股朝向进行分娩区域的精准划定
	{
	case 0:  //左上
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = std::min<int>(0, big_pig_box[1] - height);
		this->bornArea.width = 3 * width;
		this->bornArea.height = 3 * height;
		break;
	case 2:   //左下
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = (big_pig_box[1] + big_pig_box[3]) / 2;
		this->bornArea.width = 3 * width;
		this->bornArea.height = std::min<int>(HEIGHT - 1 - this->bornArea.y, 3 * height);
		break;
	case 1:   //右上
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = std::min<int>(0, big_pig_box[1] - height);
		this->bornArea.width = std::min<int>(WIDTH - 1 - this->bornArea.x, 3 * width);
		this->bornArea.height = 3 * height;
		break;
	case 3:   //右下
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = (big_pig_box[1] + big_pig_box[3]) / 2;
		this->bornArea.width = std::min<int>(WIDTH - 1 - this->bornArea.x, 3 * width);
		this->bornArea.height = std::min<int>(HEIGHT - 1 - this->bornArea.y, 3 * height);
		break;
	case 4:   //正左
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = big_pig_box[1];
		this->bornArea.width = 3 * width;
		this->bornArea.height = big_pig_box[3] - big_pig_box[1];
		break;
	case 5:   //正右
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = big_pig_box[1];
		this->bornArea.width = std::min<int>(3 * width, WIDTH - 1 - bornArea.x);
		this->bornArea.height = big_pig_box[3] - big_pig_box[1];
		break;
	default:
		break;
	}
}

//************** 静态成员函数部分 **********
string PigDetector::getTime(const time_t& t)
{
	tm* T = localtime(&t);
	string year = num2str(T->tm_year + 1900);   //年
	string mon = num2str(T->tm_mon + 1);   //月
	string day = num2str(T->tm_mday);     //日
	string hour = num2str(T->tm_hour);   //时
	string minute = num2str(T->tm_min);  //分
	string sec = num2str(T->tm_sec);    //秒

	if (mon.size() == 1)
		mon = "0" + mon;
	if (day.size() == 1)
		day = "0" + day;
	if (hour.size() == 1)
		hour = "0" + hour;
	if (minute.size() == 1)
		minute = "0" + minute;
	if (sec.size() == 1)
		sec = "0" + sec;

	string result = year + mon + day + hour + minute + sec;
	return result;
}

int PigDetector::getMaxOfMap(const map<unsigned, unsigned>& lhs, int* count){  //求出一个map里出现次数最多的数，以及其出现的次数
	int times = 0;
	int value;
	for (map<unsigned, unsigned>::const_iterator it = lhs.begin(); it != lhs.end(); it++){
		if (it->second > times){
			times = it->second;
			value = it->first;
		}
	}
	*count = times;
	return value;
}

void PigDetector::pairBBox(const vector<BBox<float> >& lastBBox, vector<BBox<float> >& currentBBox){//使当前帧的检测结果与上一帧的检测结果一一对应
	for (int i = 0; i < currentBBox.size(); i++)
		currentBBox[i].order = -1;

	const int lastNum = lastBBox.size();  //上一帧的目标数
	const int curNum = currentBBox.size(); //当前帧的目标数

	const int MAXCOUNT = 99;   //最多纪录20只仔猪,最开始20个编号都可用状态
	vector<bool>  availableIndex(MAXCOUNT);
	for (int i = 0; i < MAXCOUNT; ++i)
		availableIndex[i] = true;

	if (curNum <= lastNum){
		//对当前帧的结果的每一个目标框，依次寻找上一帧最近的目标框，并赋予序号
		for (int i = 0; i < curNum; ++i){
			int minDis = 1000000000;
			int minIndex;
			for (int j = 0; j < lastNum; ++j){
				if (availableIndex[lastBBox[j].order] == true){  //如果第j个猪的id可用
					int dis = getDis(lastBBox[j], currentBBox[i]);
					if (dis < minDis){
						minDis = dis;
						minIndex = j;
					}
				}
			}
			currentBBox[i].order = lastBBox[minIndex].order;
			availableIndex[lastBBox[minIndex].order] = false;  //不可用
		}
	}
	else{  // curNum>lastNum
		for (int i = 0; i < lastNum; ++i){
			int minDis = 1000000000;
			int minIndex;
			for (int j = 0; j < curNum; ++j){
				if (currentBBox[j].order == -1){   //如果第j个猪的id可用
					int dis = getDis(lastBBox[i], currentBBox[j]);
					if (dis < minDis){
						minDis = dis;
						minIndex = j;
					}
				}
			}
			currentBBox[minIndex].order = lastBBox[i].order;
			availableIndex[currentBBox[minIndex].order] = false;  //不可用
		}
		//给多出的框编号
		for (int i = 0; i < curNum; i++){
			if (currentBBox[i].order == -1){  //该box还没有分配编号
				for (int index = 0; index < MAXCOUNT; index++){
					if (availableIndex[index] == true){
						availableIndex[index] = false;
						currentBBox[i].order = index;
						break;
					}
				}
			}
		}
	}
}
int PigDetector::getDis(const BBox<float>& lhs, BBox<float>& rhs){  //求两个框的距离
	int x1 = (lhs[0] + lhs[2]) / 2;
	int y1 = (lhs[1] + lhs[3]) / 2;
	int x2 = (rhs[0] + rhs[2]) / 2;
	int y2 = (rhs[1] + rhs[3]) / 2;
	return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
}

string PigDetector::num2str(int i){
	std::stringstream ss;
	ss << i;
	return ss.str();
}
void PigDetector::drawRect(cv::Mat& frame, const caffe::Frcnn::BBox<float>& box, bool drawOrder){ //画框
	Scalar color = PigDetector::label_color[box.id];  //根据类别，画不同颜色的框
	cv::rectangle(frame, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), color);
	if (drawOrder)  //如果需要画框的跟踪序号
		putText(frame, num2str(box.order + 1), cv::Point(box[0], box[1]), 1, 1.0, color);
}
void PigDetector::drawRects(cv::Mat& frame, const std::vector<caffe::Frcnn::BBox<float> >& boxes, bool drawOrder)
{
	for (int i = 0; i < boxes.size(); ++i)
		drawRect(frame, boxes[i], drawOrder);
}

