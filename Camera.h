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

extern vector<Mat> g_frames;    //ȫ�ֱ���
extern vector<int> camera_channels;    //����ͨ��
class Camera{
public:
	//���캯��������ͷ��IP���û��������롢ͨ����
	Camera(const string& ip, const string& username, const string& password, unsigned channel_num);
	~Camera();
	//��������ͷ
	void start();
	static const int Camera::FRAME_WIDTH;  //��������ŵ��̶�����

private:
	LONG lUserID;  //ȡ�õ�ID
	//����ͷ��IP���û��������롢ͨ����
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