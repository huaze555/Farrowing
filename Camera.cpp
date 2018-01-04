#include "Camera.h"
#include<thread>
#include<algorithm>

//�����ص�������һЩ����
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
	case EXCEPTION_RECONNECT:    //Ԥ��ʱ����
		printf("----------reconnect--------%d\n", time(NULL));
		break;
	default:
		break;
	}
}
//ʵ�ʵ��õĺ���
//����ص� ��ƵΪYUV����(YV12)����ƵΪPCM����
void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2, const int channel_num){  //flag��ʾ�ڼ�·����ͷ
	long lFrameType = pFrameInfo->nType;
	//������Ƶ֡�Ĺ���
	if (lFrameType == T_YV12){
		IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//�õ�ͼ���Y����  
		yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//�õ�ȫ��RGBͼ��
		cvCvtColor(pImgYCrCb, pImgYCrCb, CV_YCrCb2RGB);
		//�����Ƶ֡
		Mat tmp(pImgYCrCb);
		double rate = Camera::FRAME_WIDTH * 1.0 / tmp.cols;
		resize(tmp, tmp, Size(), rate, rate);
		//�����ҵ�ͨ����channel_num����ͷ��Ӧ��ȫ�ֱ����±꣬�ٿ�������ȫ�ֱ�����
		tmp.copyTo(g_frames[find(camera_channels.begin(), camera_channels.end(), channel_num) - camera_channels.begin()]); 
		//����
		cvReleaseImage(&pImgYCrCb);
	}
}

//����˸����ǵ��õ� DecCBFun ���� 
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
///ʵʱ���ص�
void CALLBACK fRealDataCallBack1(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	//�ʼ����caseһ��NET_DVR_SYSHEAD,Ȼ��һֱ��case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort1)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort1, DecCBFun1))   //���ĺ���DecCBFun0
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort1, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort1))
			{
				dRet = PlayM4_GetLastError(nPort1);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort2)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort2, DecCBFun2))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort2, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort2))
			{
				dRet = PlayM4_GetLastError(nPort2);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	//�ʼ����caseһ��NET_DVR_SYSHEAD,Ȼ��һֱ��case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort3)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort3, DecCBFun3))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort3, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort3))
			{
				dRet = PlayM4_GetLastError(nPort3);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort4)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort4, DecCBFun4))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort4, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort4))
			{
				dRet = PlayM4_GetLastError(nPort4);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	//�ʼ����caseһ��NET_DVR_SYSHEAD,Ȼ��һֱ��case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort5)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort5, DecCBFun5))   //���ĺ���DecCBFun0
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort5, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort5))
			{
				dRet = PlayM4_GetLastError(nPort5);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort6)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort6, DecCBFun6))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort6, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort6))
			{
				dRet = PlayM4_GetLastError(nPort6);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	//�ʼ����caseһ��NET_DVR_SYSHEAD,Ȼ��һֱ��case NET_DVR_STREAMDATA
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort7)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort7, DecCBFun7))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort7, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort7))
			{
				dRet = PlayM4_GetLastError(nPort7);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	case NET_DVR_SYSHEAD:    //ϵͳͷ
		if (!PlayM4_GetPort(&nPort8)) //��ȡ���ſ�δʹ�õ�ͨ����
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
			//���ý���ص����� ֻ���벻��ʾ
			if (!PlayM4_SetDecCallBack(nPort8, DecCBFun8))   //���ĺ���DecCBFun1
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}

			//����Ƶ����
			if (!PlayM4_Play(nPort8, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}

			//����Ƶ����, ��Ҫ�����Ǹ�����
			if (!PlayM4_PlaySound(nPort8))
			{
				dRet = PlayM4_GetLastError(nPort8);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //��������
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
	//ע���û�
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
			//�����ҵ�ͨ����channel_num����ͷ��Ӧ��ȫ�ֱ����±꣬�ٿ�������ȫ�ֱ�����
			frame.copyTo(g_frames[find(camera_channels.begin(), camera_channels.end(), 3) - camera_channels.begin()]);
			std::this_thread::sleep_for(std::chrono::milliseconds(35));
		}
	}
	else
	{

		// �ٳ�ʼ��,�����
		//NET_DVR_Init();
		//����������ʱ��������ʱ�䣬��ѡ
		/*NET_DVR_SetConnectTime(5000, 1);
		NET_DVR_SetReconnect(10000, true);*/

		//�� ע���豸����ѡ
		NET_DVR_DEVICEINFO_V30 struDeviceInfo;
		this->lUserID = NET_DVR_Login_V30(const_cast<char*>(IP.c_str()), 8000, const_cast<char*>(USERNAME.c_str()), const_cast<char*>(PASSWORD.c_str()), &struDeviceInfo);
		if (lUserID < 0)
			throw string("ͨ����" + num2str(CHANNEL_NUM) + ": " + "Login error, " + num2str(NET_DVR_GetLastError()));

		//�������쳣�ص�����
		NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);

		//������Ԥ������������ͷ�ص�������   ,��ʵʱ�ص�����
		NET_DVR_CLIENTINFO ClientInfo;
		ClientInfo.lChannel = MAX_ANALOG_CHANNUM + this->CHANNEL_NUM;      //Channel number �豸ͨ����, MAX_ANALOG_CHANNUM=32��Ϊ���ͨ����
		ClientInfo.hPlayWnd = NULL;     //����Ϊ�գ��豸SDK������ֻȡ��
		ClientInfo.lLinkMode = 0;       //Main Stream
		ClientInfo.sMultiCastIP = NULL;


		void (CALLBACK *fRealDataCallBack)(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser) = fRealDataCallBack1;  //����ָ��
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
			throw string("ͨ����" + num2str(CHANNEL_NUM) + ": " + " NET_DVR_RealPlay_V30 failed! Error number: " + num2str(NET_DVR_GetLastError()));
		Sleep(-1);
	}
}