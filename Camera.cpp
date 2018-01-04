#include "Camera.h"
#include<thread>
#include<algorithm>

//海康回调函数的一些参数
LONG nPort1 = -1;
LONG nPort2 = -1;
LONG nPort3 = -1;
LONG nPort4 = -1;
LONG nPort5 = -1;
LONG nPort6 = -1;
LONG nPort7 = -1;
LONG nPort8 = -1;
HWND hWnd = NULL;

void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep)
{
	int col, row;
	unsigned int Y, U, V;
	int tmp;
	int idx;

	for (row = 0; row<height; row++)
	{
		idx = row * widthStep;
		int rowptr = row*width;

		for (col = 0; col<width; col++)
		{

			tmp = (row / 2)*(width / 2) + (col / 2);
			Y = (unsigned int)inYv12[row*width + col];
			U = (unsigned int)inYv12[width*height + width*height / 4 + tmp];
			V = (unsigned int)inYv12[width*height + tmp];
			outYuv[idx + col * 3] = Y;
			outYuv[idx + col * 3 + 1] = U;
			outYuv[idx + col * 3 + 2] = V;
		}
	}
}
void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
{
	char tempbuf[256] = { 0 };
	switch (dwType)
	{
	case EXCEPTION_RECONNECT:    //预览时重连
		printf("----------reconnect--------%d\n", time(NULL));
		break;
	default:
		break;
	}
}
//实际调用的函数
//解码回调 视频为YUV数据(YV12)，音频为PCM数据
void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2, const int channel_num){  //flag表示第几路摄像头
	long lFrameType = pFrameInfo->nType;
	//处理视频帧的过程
	if (lFrameType == T_YV12){
		IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//得到图像的Y分量  
		yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//得到全部RGB图像
		cvCvtColor(pImgYCrCb, pImgYCrCb, CV_YCrCb2RGB);
		//获得视频帧
		Mat tmp(pImgYCrCb);
		double rate = Camera::FRAME_WIDTH * 1.0 / tmp.cols;
		resize(tmp, tmp, Size(), rate, rate);
		//首先找到通道号channel_num摄像头对应的全局变量下标，再拷贝到该全局变量中
		tmp.copyTo(g_frames[find(camera_channels.begin(), camera_channels.end(), channel_num) - camera_channels.begin()]); 
		//析构
		cvReleaseImage(&pImgYCrCb);
	}
}

//下面八个都是调用的 DecCBFun 函数 
void CALLBACK DecCBFun1(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 1);
}
void CALLBACK DecCBFun2(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 2);
}
void CALLBACK DecCBFun3(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 3);
}
void CALLBACK DecCBFun4(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 4);
}
void CALLBACK DecCBFun5(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 5);
}
void CALLBACK DecCBFun6(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 6);
}
void CALLBACK DecCBFun7(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 7);
}
void CALLBACK DecCBFun8(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	DecCBFun(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2, 8);
}
///实时流回调
void CALLBACK fRealDataCallBack1(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	//最开始，是case一次NET_DVR_SYSHEAD,然后一直是case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort1)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort1, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort1, DecCBFun1))   //核心函数DecCBFun0
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort1, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort1))
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort1 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort1, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort1, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack2(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort2)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort2, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort2, DecCBFun2))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort2, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort2))
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort2 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort2, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort2, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack3(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	//最开始，是case一次NET_DVR_SYSHEAD,然后一直是case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort3)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort3, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort3, DecCBFun3))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort3, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort3))
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort3 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort3, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort3, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack4(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort4)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort4, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort4, DecCBFun4))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort4, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort4))
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort4 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort4, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort4, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack5(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	//最开始，是case一次NET_DVR_SYSHEAD,然后一直是case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort5)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort5, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort5, DecCBFun5))   //核心函数DecCBFun0
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort5, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort5))
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort5 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort5, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort5, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack6(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort6)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort6, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort6, DecCBFun6))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort6, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort6))
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort6 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort6, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort6, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack7(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	//最开始，是case一次NET_DVR_SYSHEAD,然后一直是case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort7)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort7, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort7, DecCBFun7))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort7, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort7))
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort7 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort7, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort7, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}
void CALLBACK fRealDataCallBack8(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort8)) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort8, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort8, DecCBFun8))   //核心函数DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}

			//打开视频解码
			if (!PlayM4_Play(nPort8, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort8))
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort8 != -1)
		{
			BOOL inData = PlayM4_InputData(nPort8, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort8, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}



const int Camera::FRAME_WIDTH = 800;
Camera::Camera(const string& ip, const string& username, const string& password, unsigned channel_num)
:IP(ip), USERNAME(username), PASSWORD(password), CHANNEL_NUM(channel_num){}

Camera::~Camera()
{
	//注销用户
	NET_DVR_Logout(lUserID);
	NET_DVR_Cleanup();
}
void Camera::start()
{
	if (CHANNEL_NUM == 3)
	{
		using namespace cv;
		VideoCapture capture("4.mp4");
		Mat frame;
		while (capture.read(frame))
		{
			double rate = Camera::FRAME_WIDTH * 1.0 / frame.cols;
			resize(frame, frame, Size(), rate, rate);
			//首先找到通道号channel_num摄像头对应的全局变量下标，再拷贝到该全局变量中
			frame.copyTo(g_frames[find(camera_channels.begin(), camera_channels.end(), 3) - camera_channels.begin()]);
			std::this_thread::sleep_for(std::chrono::milliseconds(35));
		}
	}
	else
	{

		// ①初始化,必须的
		//NET_DVR_Init();
		//②设置连接时间与重连时间，可选
		/*NET_DVR_SetConnectTime(5000, 1);
		NET_DVR_SetReconnect(10000, true);*/

		//③ 注册设备，必选
		NET_DVR_DEVICEINFO_V30 struDeviceInfo;
		this->lUserID = NET_DVR_Login_V30(const_cast<char*>(IP.c_str()), 8000, const_cast<char*>(USERNAME.c_str()), const_cast<char*>(PASSWORD.c_str()), &struDeviceInfo);
		if (lUserID < 0)
			throw string("通道号" + num2str(CHANNEL_NUM) + ": " + "Login error, " + num2str(NET_DVR_GetLastError()));

		//④设置异常回调函数
		NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);

		//⑤启动预览并设置摄像头回调数据流   ,主实时回调函数
		NET_DVR_CLIENTINFO ClientInfo;
		ClientInfo.lChannel = MAX_ANALOG_CHANNUM + this->CHANNEL_NUM;      //Channel number 设备通道号, MAX_ANALOG_CHANNUM=32，为最大通道数
		ClientInfo.hPlayWnd = NULL;     //窗口为空，设备SDK不解码只取流
		ClientInfo.lLinkMode = 0;       //Main Stream
		ClientInfo.sMultiCastIP = NULL;


		void (CALLBACK *fRealDataCallBack)(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser) = fRealDataCallBack1;  //函数指针
		switch (CHANNEL_NUM)
		{
		case 1:
			fRealDataCallBack = fRealDataCallBack1;
			break;
		case 2:
			fRealDataCallBack = fRealDataCallBack2;
			break;
		case 3:
			fRealDataCallBack = fRealDataCallBack3;
			break;
		case 4:
			fRealDataCallBack = fRealDataCallBack4;
			break;
		case 5:
			fRealDataCallBack = fRealDataCallBack5;
			break;
		case 6:
			fRealDataCallBack = fRealDataCallBack6;
			break;
		case 7:
			fRealDataCallBack = fRealDataCallBack7;
			break;
		case 8:
			fRealDataCallBack = fRealDataCallBack8;
			break;
		default:
			break;
		}
		LONG lRealPlayHandle = NET_DVR_RealPlay_V30(lUserID, &ClientInfo, fRealDataCallBack, NULL, TRUE);

		if (lRealPlayHandle < 0)
			throw string("通道号" + num2str(CHANNEL_NUM) + ": " + " NET_DVR_RealPlay_V30 failed! Error number: " + num2str(NET_DVR_GetLastError()));
		Sleep(-1);
	}
}