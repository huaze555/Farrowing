#include "PigDetector.h"
#include "DirectionJudge.hpp"
#include <cmath>
#include <io.h>       //�ж��ļ����Ƿ����
#include <direct.h>   //�����ļ���
#include <algorithm>
#include <fstream>
#include <stdio.h>  //ɾ���ļ�����remove ��Ҫ���ļ�ͷ
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace caffe::Frcnn;


//****************��̬����***********
//��ͬ����Ӧ�Ŀ����ɫ, ����Ϊ���0����ɫ�� ����Ϊ���1����ɫ
std::map<unsigned, cv::Scalar> PigDetector::label_color = { { 1, Scalar(0, 0, 255) }, { 2, Scalar(0, 255, 255) } };
//Ŀ����ģ��
FRCNN_API::Detector* PigDetector::obj_detect = new FRCNN_API::Detector
(PIG_DETECT_MODEL_FOLD + "\\pigDetect.prototxt", PIG_DETECT_MODEL_FOLD + "\\pigDetect.caffemodel", PIG_DETECT_MODEL_FOLD + "\\pigDetect.json", true, true);


//************** �Ǿ�̬��Ա�������� **********
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
	//��������ͼƬ�ļ���
	if (_access(SAVED_FOLD.c_str(), 0) == -1)   //�ļ��в����ڣ��򴴽��ļ���
		_mkdir(SAVED_FOLD.c_str());
	std::cout << "��������Ӳ�̿�¼��, ���" << CDWriter_NO << ", ����ͷͨ����" << CHANNEL_NO << ", ͼƬ������: " << SAVED_FOLD << endl;
}

vector<BBox<float> > PigDetector::frameProcess(const cv::Mat& _frame)
{
	Mat frame = _frame.clone();
	//��һ�δ�����Ҫ�������������
	if (first_process)
	{
		DirectionJudge judger(PIG_DIRECTION_MODEL_FOLD + "\\pigDirection_deploy.prototxt",
			PIG_DIRECTION_MODEL_FOLD + "\\pigDirection.caffemodel",
			PIG_DIRECTION_MODEL_FOLD + "\\pigDirection_mean.binaryproto");
		//���������򣬹�6�֣� 0-1-2-3-4-5���ֱ�������ϣ����ϣ����£����£���������
		this->bornAreaDirection = judger.judge(frame);
		map<unsigned, string> m{ { 0, "����" }, { 1, "����" }, { 2, "����" }, { 3, "����" }, { 4, "����" }, { 5, "����" } };
		cout << "����ͷͨ����" << CHANNEL_NO << ", ����������" << m[bornAreaDirection] << "��" << endl;
		first_process = false;
	}
	vector<BBox<float> > wholeResults = PigDetector::obj_detect->predict(frame); //�ȼ���������
	//����������ǰ��,�������ں���, ���ص�һ��ָ������ĵ�����
	auto iter = std::partition(wholeResults.begin(), wholeResults.end(), [](const BBox<float>& box){return box.id == 1; }); //�����㷨��Lamdba���ʽ
	vector<BBox<float> > big_pig_boxes(iter, wholeResults.end());  //ȡ��ȫ�����Ŀ�������ĸ��Ŀ�
	if (big_pig_boxes.size() > 0)    //��Ϊ����û�м�⵽����������Ҫ�Ӹ��ж�
	{
		this->big_pig_rect = bbox2Rect(big_pig_boxes[0]);
		this->setBornArea(frame.cols, frame.rows, big_pig_boxes[0]);
	}

	this->preResults = this->curResults;         //���浱ǰ����������֡�����
	this->curResults = { wholeResults.begin(), iter };    //ȡ��ȫ�����Ŀ�����������Ŀ�
	this->pairBBox(preResults, curResults);      //����ǰ֡����һ֡�Ľ������ƥ��
	int detectNum = curResults.size();           // ��ǰ֡�����Ŀ�ĸ���

	// !!!!!!!!!!!!!!!!!!!!!!!
	pig_info.push_back(Pig_Info(frame, curResults, time(NULL))); //�������µ� ֡���������ʱ��
	M[detectNum]++;    //�����µ�Ŀ�����Ŀ+1
	if (++realFrames > K)  //���֡��+1, ����Ѿ�������������K֡
	{
		realFrames = K;    //���⴦����ֹ����ʱ�����е��µ��������
		//������ɵ�֡������pig_info,M
		M[pig_info.front().boxes.size()]--;  //����ɵ�Ŀ�����Ŀ-1
		pig_info.pop_front();

		//*****************��ʼ�����Ƿ�ﵽ��������**********************
		unsigned tmp_pigCounts1 = computePigCountsByWay1();
		unsigned tmp_pigCounts2 = computePigCountsByWay2();
		unsigned tmp_max = std::max(tmp_pigCounts1, tmp_pigCounts2);  //���߽ϴ���
		unsigned tmp_min = std::min(tmp_pigCounts1, tmp_pigCounts2);  //���߽�С��
		assert((tmp_min >= pigCounts) || (tmp_max <= pigCounts));

		bool hasChanged = false;   //�����Ƿ����仯
		unsigned oldPigCounts;
		//�ж������������ˣ������ˣ����ǲ���
		if (tmp_max > pigCounts || tmp_min < pigCounts)   //�����б仯
		{
			hasChanged = true;
			oldPigCounts = pigCounts;     //����֮ǰ��������

			if (tmp_max > pigCounts)     //��������
			{
				pigCounts = tmp_max;   //����pigCounts
				// ������������ʱ����Ҫ������ǵ�һֻ�������a�����Ƿǵ�һֻ�������b���������ٻ�c
				bool inBornArea = false;   //�Ƿ��������ڳ�������
				for (int ii = 0; ii < detectNum; ++ii)
				{
					if (isInBornArea(curResults[ii])){
						inBornArea = true;
						break;
					}
				}
				if (inBornArea)   //��������ڳ�������, ��Ҫ��һ���ж��ǵ�һֻ�������a���Ƿǵ�һֻ�������b��	
				{
					if (!hasBorn&&pigCounts == 1)  //ֻ�е��տ�ʼ�������Ҽ�⵽����ֻ��һֻ�����ж�Ϊ�³����ĵ�һֻ
						this->status = 'a';
					else                       //�����ж�Ϊ�ڶ�ֻ�³���
						this->status = 'b';
				}
				else  //�����³������������ǿ���֮ǰ���ڵ��������ֳ��ֵ�����,���ٻ�c
					this->status = 'c';
				this->hasBorn = true;   //�Ѿ���ʼ������
			}
			else        //�������
			{
				pigCounts = tmp_min;   //����pigCounts
				this->status = 'd';     //��ʧ
			}
		}
		if (hasChanged)
		{
			//����״̬�������Ϣ������ͼƬ(��һֻ����a���ǵ�һֻ����b���ٻ�c����ʧd)
			cout << "Ӳ�̿�¼�����" << CDWriter_NO << ", ����ͷͨ����" << CHANNEL_NO << ": �������仯��" << oldPigCounts << " -> " << pigCounts;
			if (this->status == 'a')
				cout << ", �³���(��һֻ)\n";
			else if (this->status == 'b')
				cout << ", �³���(�ǵ�һֻ)\n";
			else if (this->status == 'c')
				cout << ", �ٻ�\n";
			else
				cout << ", ��ʧ\n";

			Pig_Info need_to_save;      //��Ҫ�������һ֡
			BBox<float> bornArea_box;   //��������³����������򱣴��Ǹ���������
			//�ҵ�Ŀ�����ĿΪpigCount����ɵ�һ֡
			for (auto iter = pig_info.begin(); iter != pig_info.end(); ++iter)
			{
				if (iter->boxes.size() == pigCounts)
				{
					if ((this->status == 'a' || this->status == 'b'))   //��С�����������Ҫ��λ���ڳ��������֡
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
			string name = CDWriter_NO + num2str(this->CHANNEL_NO) + getPigCounts2string() + this->status + getTime(need_to_save.t) + ".jpg";   //�豸��+��ǰ������+����״̬+��ǰʱ��+��׺��
			//������һ֡��������Ŀ�����Ϣ		
			//imwrite("C:\\Users\\Lab1060-i7\\Desktop\\ĸ��������ܼ��\\PigImages\\imgs\\" + name, iter->img, { 1, 100 });
			//������һ֡���һ��߿���Ϣ
			drawRects(need_to_save.img, need_to_save.boxes, true);
			if (this->status == 'a' || this->status == 'b')  //������³�����������Ҫ����ɫ��
			{
				rectangle(need_to_save.img, bbox2Rect(bornArea_box), Scalar(0, 255, 0), 2);
				putText(need_to_save.img, num2str(bornArea_box.order + 1), cv::Point(bornArea_box[0], bornArea_box[1]), 1, 1.0, Scalar(0, 255, 0), 2);
			}
			imwrite(SAVED_FOLD + "\\" + name, need_to_save.img, { 1, 100 });   //����
		}//end if 
	}
	//!!!!!!!!!!!!!!!!
	return curResults;
}

//����1��õĵ�ǰ�����������������K֡����rate�������ϵ�Ŀ�����Ϊn�����ж���ǰ������Ϊn�����򲻸���
unsigned PigDetector::computePigCountsByWay1()
{
	int count;   // ����֡��
	//pig_info �������������������Լ���Ӧ���ֵ�֡��
	int value = getMaxOfMap(M, &count);  //������
	return (count >= K*RATE) ? value : pigCounts;
}

//����2��õĵ�ǰ������������������K֡������Ŀ������ľ�ֵ����Ϊ��ǰ��������
unsigned PigDetector::computePigCountsByWay2()
{
	float sum = 0.0;
	for (auto it = pig_info.begin(); it != pig_info.end(); ++it)
		sum += (it->boxes.size())*(it->weight);
	float mean = sum / pig_info.size();    //��ֵ
	if ((mean <= pigCounts - 1) || (mean >= pigCounts + 1))
		return nearbyint(mean);
	return pigCounts;
}

bool PigDetector::isInBornArea(const BBox<float>& box){
	Rect rect(Point(box[0], box[1]), Point(Point(box[2], box[3])));
	return (rect&bornArea) == rect;
}
//�Ƿ�������һֻ�����ڳ�����������ǣ������ڳ�������Ŀ�
bool PigDetector::atleast_one_in_born_area(const std::vector<BBox<float> >& boxes, BBox<float>& box){
	vector<BBox<float> > in_born_area_boxes;
	for (int i = 0; i < boxes.size(); i++)
	{
		if (isInBornArea(boxes[i]))
			in_born_area_boxes.push_back(boxes[i]);
	}
	if (in_born_area_boxes.size() != 0)  //ѡ�����������ķ���
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
//���ݴ���λ�����ó�������
void PigDetector::setBornArea(const int WIDTH, const int HEIGHT, const BBox<float>& big_pig_box){
	int width = (big_pig_box[2] - big_pig_box[0]) / 8;  //�Դ�����ȵİ˷�֮һΪС��
	int height = (big_pig_box[3] - big_pig_box[1]) / 4;  //�Դ����ĸ߶��ķ�֮һΪС��
	switch (bornAreaDirection)   //���ݴ���ƨ�ɳ�����з�������ľ�׼����
	{
	case 0:  //����
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = std::min<int>(0, big_pig_box[1] - height);
		this->bornArea.width = 3 * width;
		this->bornArea.height = 3 * height;
		break;
	case 2:   //����
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = (big_pig_box[1] + big_pig_box[3]) / 2;
		this->bornArea.width = 3 * width;
		this->bornArea.height = std::min<int>(HEIGHT - 1 - this->bornArea.y, 3 * height);
		break;
	case 1:   //����
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = std::min<int>(0, big_pig_box[1] - height);
		this->bornArea.width = std::min<int>(WIDTH - 1 - this->bornArea.x, 3 * width);
		this->bornArea.height = 3 * height;
		break;
	case 3:   //����
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = (big_pig_box[1] + big_pig_box[3]) / 2;
		this->bornArea.width = std::min<int>(WIDTH - 1 - this->bornArea.x, 3 * width);
		this->bornArea.height = std::min<int>(HEIGHT - 1 - this->bornArea.y, 3 * height);
		break;
	case 4:   //����
		this->bornArea.x = std::max<int>(0, big_pig_box[0] - 2 * width);
		this->bornArea.y = big_pig_box[1];
		this->bornArea.width = 3 * width;
		this->bornArea.height = big_pig_box[3] - big_pig_box[1];
		break;
	case 5:   //����
		this->bornArea.x = big_pig_box[2] - width;
		this->bornArea.y = big_pig_box[1];
		this->bornArea.width = std::min<int>(3 * width, WIDTH - 1 - bornArea.x);
		this->bornArea.height = big_pig_box[3] - big_pig_box[1];
		break;
	default:
		break;
	}
}

//************** ��̬��Ա�������� **********
string PigDetector::getTime(const time_t& t)
{
	tm* T = localtime(&t);
	string year = num2str(T->tm_year + 1900);   //��
	string mon = num2str(T->tm_mon + 1);   //��
	string day = num2str(T->tm_mday);     //��
	string hour = num2str(T->tm_hour);   //ʱ
	string minute = num2str(T->tm_min);  //��
	string sec = num2str(T->tm_sec);    //��

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

int PigDetector::getMaxOfMap(const map<unsigned, unsigned>& lhs, int* count){  //���һ��map����ִ������������Լ�����ֵĴ���
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

void PigDetector::pairBBox(const vector<BBox<float> >& lastBBox, vector<BBox<float> >& currentBBox){//ʹ��ǰ֡�ļ��������һ֡�ļ����һһ��Ӧ
	for (int i = 0; i < currentBBox.size(); i++)
		currentBBox[i].order = -1;

	const int lastNum = lastBBox.size();  //��һ֡��Ŀ����
	const int curNum = currentBBox.size(); //��ǰ֡��Ŀ����

	const int MAXCOUNT = 99;   //����¼20ֻ����,�ʼ20����Ŷ�����״̬
	vector<bool>  availableIndex(MAXCOUNT);
	for (int i = 0; i < MAXCOUNT; ++i)
		availableIndex[i] = true;

	if (curNum <= lastNum){
		//�Ե�ǰ֡�Ľ����ÿһ��Ŀ�������Ѱ����һ֡�����Ŀ��򣬲��������
		for (int i = 0; i < curNum; ++i){
			int minDis = 1000000000;
			int minIndex;
			for (int j = 0; j < lastNum; ++j){
				if (availableIndex[lastBBox[j].order] == true){  //�����j�����id����
					int dis = getDis(lastBBox[j], currentBBox[i]);
					if (dis < minDis){
						minDis = dis;
						minIndex = j;
					}
				}
			}
			currentBBox[i].order = lastBBox[minIndex].order;
			availableIndex[lastBBox[minIndex].order] = false;  //������
		}
	}
	else{  // curNum>lastNum
		for (int i = 0; i < lastNum; ++i){
			int minDis = 1000000000;
			int minIndex;
			for (int j = 0; j < curNum; ++j){
				if (currentBBox[j].order == -1){   //�����j�����id����
					int dis = getDis(lastBBox[i], currentBBox[j]);
					if (dis < minDis){
						minDis = dis;
						minIndex = j;
					}
				}
			}
			currentBBox[minIndex].order = lastBBox[i].order;
			availableIndex[currentBBox[minIndex].order] = false;  //������
		}
		//������Ŀ���
		for (int i = 0; i < curNum; i++){
			if (currentBBox[i].order == -1){  //��box��û�з�����
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
int PigDetector::getDis(const BBox<float>& lhs, BBox<float>& rhs){  //��������ľ���
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
void PigDetector::drawRect(cv::Mat& frame, const caffe::Frcnn::BBox<float>& box, bool drawOrder){ //����
	Scalar color = PigDetector::label_color[box.id];  //������𣬻���ͬ��ɫ�Ŀ�
	cv::rectangle(frame, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), color);
	if (drawOrder)  //�����Ҫ����ĸ������
		putText(frame, num2str(box.order + 1), cv::Point(box[0], box[1]), 1, 1.0, color);
}
void PigDetector::drawRects(cv::Mat& frame, const std::vector<caffe::Frcnn::BBox<float> >& boxes, bool drawOrder)
{
	for (int i = 0; i < boxes.size(); ++i)
		drawRect(frame, boxes[i], drawOrder);
}

