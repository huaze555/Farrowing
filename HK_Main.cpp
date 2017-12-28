#include <fstream>
#include <io.h>       //判定文件夹是否存在
#include <thread>     //创建线程
#include <mutex>
#include "PigDetector.h"    // PigDetector.h必须在util.h之前！
#include "Camera.h"

using namespace std;
using namespace cv;
using caffe::Frcnn::BBox;

mutex locker;

const string CONFIG_FILE = "config_file.yaml";   //保存摄像头IP配置文件信息
const string SAVED_FOLD = "saved_images\\";   //保存图片的文件夹
//const string SAVED_FOLD = "C:\\Users\\Lab1060-i7\\Desktop\\母猪分娩智能监控\\PigImages\\source_files\\";   //保存图片的文件夹
const string PIG_DETECT_MODEL_FOLD = "models\\pigDetect";   //保存仔猪检测模型的文件夹
const string PIG_DIRECTION_MODEL_FOLD = "models\\pigDirection";   //保存母猪躺向模型的文件夹
const int MAX_NUM = 8;  //最多支持8路

vector<Mat> g_frames;   //8路，每一路保存最新的帧
vector<int> camera_channels;    //通道
string CDWriter_NO, IP, UserName, PassWord;   // 硬盘刻录机编号、IP、账号、密码
int CAMERA_NUM;        //实际测试的摄像头数量，从文件里读取

void readMsgFromFile(const string& config_file);  //读取硬盘录像机配置信息


//开启摄像头线程
void beginCamera(unsigned channel_num){
	Camera camera(IP, UserName, PassWord, channel_num);
	try
	{
		camera.start();
	}
	catch (string& msg)   //如果启动失败，输出错误信息并结束线程
	{
		locker.lock();
		cout << "发生异常，异常信息为: \"" << msg << "\"" << endl;
		locker.unlock();
		return;
	}
}


//主线程
int main(int argc, char** argv)
{
	try
	{
		readMsgFromFile(CONFIG_FILE);   //读取配置文件，保存各个摄像头的IP、账号、密码
		
		vector<thread> threads(CAMERA_NUM); //创建线程和图像帧全局变量
		for (int i = 0; i < CAMERA_NUM; ++i)
		{
			g_frames.push_back(Mat());
			threads[i] = std::thread(beginCamera, camera_channels[i]);
			threads[i].detach();
		}

		vector<PigDetector> detectors;    //最多8个跟踪器
		for (int i = 0; i < CAMERA_NUM; ++i)
			detectors.push_back(PigDetector(CDWriter_NO, camera_channels[i], IP, SAVED_FOLD));

		while (true)
		{
			for (int i = 0; i < CAMERA_NUM; ++i)
			{
				if (g_frames[i].data)
				{
					Mat tmp = g_frames[i].clone();         //!!!!!此处必须为拷贝	
	
					clock_t t = clock();
					vector<BBox<float> > boxes = detectors[i].frameProcess(tmp);   //进行检测
					//cout << clock() - t << endl;
					detectors[i].drawRects(tmp, boxes, true); //画仔猪的框,以及跟踪序号
					rectangle(tmp, detectors[i].big_pig_rect, Scalar(0, 255, 255),2);  //画大猪区域
					rectangle(tmp, detectors[i].bornArea, Scalar(240, 32, 160),2);  //画出生区域
					imshow(num2str(detectors[i].getChannelNo()), tmp);
				}
			}
			if (waitKey(1) == 'q')
				break;
		}
	}
	catch (string& msg)
	{
		cout << "发生异常，异常信息为: \" " << msg << " \"" << endl;
	}
	return 0;
}


void readMsgFromFile(const string& config_file)  // 是否是有效的登录
{
	if (_access(config_file.c_str(), 0) == -1)  //文件不存在
		throw string("配置文件: " + CONFIG_FILE + " is not exist!");

	FileStorage fs(config_file, FileStorage::READ);   //读取yaml文件
	if (!fs.isOpened())
		throw string("配置文件: " + CONFIG_FILE + " open failed!");

	fs["VCR_NO"] >> CDWriter_NO;        // 读取硬盘录像机编号
	fs["IP"] >> IP;                     // IP
	fs["UserName"] >> UserName;         // 用户名
	fs["PassWord"] >> PassWord;         // 密码
	fs["channel_num"] >> camera_channels;   //各个摄像头通道
	
	CAMERA_NUM = camera_channels.size();   //摄像头路数
	if (CAMERA_NUM > MAX_NUM)
	{
		stringstream ss;
		ss << MAX_NUM;
		throw string("最多支持 " + ss.str() + " 路!");
	}
	fs.release();
}