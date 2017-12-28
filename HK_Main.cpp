#include <fstream>
#include <io.h>       //�ж��ļ����Ƿ����
#include <thread>     //�����߳�
#include <mutex>
#include "PigDetector.h"    // PigDetector.h������util.h֮ǰ��
#include "Camera.h"

using namespace std;
using namespace cv;
using caffe::Frcnn::BBox;

mutex locker;

const string CONFIG_FILE = "config_file.yaml";   //��������ͷIP�����ļ���Ϣ
const string SAVED_FOLD = "saved_images\\";   //����ͼƬ���ļ���
//const string SAVED_FOLD = "C:\\Users\\Lab1060-i7\\Desktop\\ĸ��������ܼ��\\PigImages\\source_files\\";   //����ͼƬ���ļ���
const string PIG_DETECT_MODEL_FOLD = "models\\pigDetect";   //����������ģ�͵��ļ���
const string PIG_DIRECTION_MODEL_FOLD = "models\\pigDirection";   //����ĸ������ģ�͵��ļ���
const int MAX_NUM = 8;  //���֧��8·

vector<Mat> g_frames;   //8·��ÿһ·�������µ�֡
vector<int> camera_channels;    //ͨ��
string CDWriter_NO, IP, UserName, PassWord;   // Ӳ�̿�¼����š�IP���˺š�����
int CAMERA_NUM;        //ʵ�ʲ��Ե�����ͷ���������ļ����ȡ

void readMsgFromFile(const string& config_file);  //��ȡӲ��¼���������Ϣ


//��������ͷ�߳�
void beginCamera(unsigned channel_num){
	Camera camera(IP, UserName, PassWord, channel_num);
	try
	{
		camera.start();
	}
	catch (string& msg)   //�������ʧ�ܣ����������Ϣ�������߳�
	{
		locker.lock();
		cout << "�����쳣���쳣��ϢΪ: \"" << msg << "\"" << endl;
		locker.unlock();
		return;
	}
}


//���߳�
int main(int argc, char** argv)
{
	try
	{
		readMsgFromFile(CONFIG_FILE);   //��ȡ�����ļ��������������ͷ��IP���˺š�����
		
		vector<thread> threads(CAMERA_NUM); //�����̺߳�ͼ��֡ȫ�ֱ���
		for (int i = 0; i < CAMERA_NUM; ++i)
		{
			g_frames.push_back(Mat());
			threads[i] = std::thread(beginCamera, camera_channels[i]);
			threads[i].detach();
		}

		vector<PigDetector> detectors;    //���8��������
		for (int i = 0; i < CAMERA_NUM; ++i)
			detectors.push_back(PigDetector(CDWriter_NO, camera_channels[i], IP, SAVED_FOLD));

		while (true)
		{
			for (int i = 0; i < CAMERA_NUM; ++i)
			{
				if (g_frames[i].data)
				{
					Mat tmp = g_frames[i].clone();         //!!!!!�˴�����Ϊ����	
	
					clock_t t = clock();
					vector<BBox<float> > boxes = detectors[i].frameProcess(tmp);   //���м��
					//cout << clock() - t << endl;
					detectors[i].drawRects(tmp, boxes, true); //������Ŀ�,�Լ��������
					rectangle(tmp, detectors[i].big_pig_rect, Scalar(0, 255, 255),2);  //����������
					rectangle(tmp, detectors[i].bornArea, Scalar(240, 32, 160),2);  //����������
					imshow(num2str(detectors[i].getChannelNo()), tmp);
				}
			}
			if (waitKey(1) == 'q')
				break;
		}
	}
	catch (string& msg)
	{
		cout << "�����쳣���쳣��ϢΪ: \" " << msg << " \"" << endl;
	}
	return 0;
}


void readMsgFromFile(const string& config_file)  // �Ƿ�����Ч�ĵ�¼
{
	if (_access(config_file.c_str(), 0) == -1)  //�ļ�������
		throw string("�����ļ�: " + CONFIG_FILE + " is not exist!");

	FileStorage fs(config_file, FileStorage::READ);   //��ȡyaml�ļ�
	if (!fs.isOpened())
		throw string("�����ļ�: " + CONFIG_FILE + " open failed!");

	fs["VCR_NO"] >> CDWriter_NO;        // ��ȡӲ��¼������
	fs["IP"] >> IP;                     // IP
	fs["UserName"] >> UserName;         // �û���
	fs["PassWord"] >> PassWord;         // ����
	fs["channel_num"] >> camera_channels;   //��������ͷͨ��
	
	CAMERA_NUM = camera_channels.size();   //����ͷ·��
	if (CAMERA_NUM > MAX_NUM)
	{
		stringstream ss;
		ss << MAX_NUM;
		throw string("���֧�� " + ss.str() + " ·!");
	}
	fs.release();
}