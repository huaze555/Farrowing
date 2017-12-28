#ifndef CAMERA_H_
#define CAMERA_H_

#include <ctime>
#include <string>
#include <vector>
#include <Windows.h>
#include <plaympeg4.h>
#include <HCNetSDK.h>
#include <opencv2\opencv.hpp>

using cv::Mat;
using cv::Size;
using std::vector;
using std::map;
using std::string;

extern vector<Mat> g_frames;    //全局变量
extern vector<int> camera_channels;    //各个通道
class Camera{
public:
	//构造函数：摄像头的IP、用户名、密码、通道号
	Camera(const string& ip, const string& username, const string& password, unsigned channel_num);
	~Camera();
	//启动摄像头
	void start();
	static const int Camera::FRAME_WIDTH;  //将宽度缩放到固定长度

private:
	LONG lUserID;  //取得的ID
	//摄像头的IP、用户名、密码、通道号
	const string IP, USERNAME, PASSWORD;
	const unsigned CHANNEL_NUM; 
	
};

inline string num2str(float i){
	std::stringstream ss;
	ss << i;
	string result = ss.str();
	return result;
}
#endif