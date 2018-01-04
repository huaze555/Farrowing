#ifndef PIGDETECTOR_H_
#define PIGDETECTOR_H_
#include <string>
#include <sstream>
#include <ctime>
#include <map>
#include <list>
#include <opencv2\opencv.hpp>

#include <caffe/common.hpp>
#include "caffe/layers/input_layer.hpp"  
#include "caffe/layers/inner_product_layer.hpp"  
#include "caffe/layers/dropout_layer.hpp"  
#include "caffe/layers/conv_layer.hpp"  
#include "caffe/layers/relu_layer.hpp"  
#include "caffe/layers/prelu_layer.hpp"  
#include "caffe/layers/pooling_layer.hpp"  
#include "caffe/layers/lrn_layer.hpp"  
#include "caffe/layers/softmax_layer.hpp"  
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/roi_pooling_layer.hpp"

//faster-rcnn
#include "caffe/FRCNN/frcnn_proposal_layer.hpp""
#include <caffe/api/FRCNN/frcnn_api.hpp>  

extern std::string CDWriter_NO;   //Ӳ�̿�¼��������ͨ��������ͷ������ͬһ����¼�����
extern const std::string PIG_DETECT_MODEL_FOLD;   //����������ģ�͵��ļ���
extern int CAMERA_NUM;        //ʵ�ʲ��Ե�����ͷ���������ļ����ȡ

//��Ŀ�����Ŀ��ͼƬ��ʱ���
struct Pig_Info
{
	cv::Mat img;          // ͼƬ
	std::vector<caffe::Frcnn::BBox<float> > boxes;   // Ŀ���
	time_t t;             // ʱ��
	float weight;         //Ȩ��
	Pig_Info(const cv::Mat& _img, const std::vector<caffe::Frcnn::BBox<float> >& _boxes,
		time_t _t) :boxes(_boxes), img(_img), t(_t), weight(0.0)
	{
		for (int i = 0; i < boxes.size(); ++i)
			weight += boxes[i].confidence;
		weight /= boxes.size();
	}
	Pig_Info(){}
};

class PigDetector
{
public:
	//***************** ���б������� **************
	
	static std::map<unsigned, cv::Scalar> label_color; //��ͬ����Ӧ�Ŀ����ɫ
	cv::Rect big_pig_rect;              //����Ŀ�

	//***************** ���к����ӿ� **************
	//���캯����Ӳ�̿�¼����š�����ͷͨ���š�����IP������ͼƬ���ļ���
	PigDetector(const std::string& Writer_NO, unsigned camera_channel, const std::string& ip,
		const std::string& saved_fold = "saved_images");

	std::vector<caffe::Frcnn::BBox<float> > frameProcess(const cv::Mat& frame);           //��ͼƬ֡���м��
	unsigned getPigCounts()const{ return pigCounts; }
	unsigned getChannelNo()const{ return CHANNEL_NO; }
	std::string getIP()const{ return IP; }
	cv::Rect getBornArea()const{ return bornArea; }
	
	//***************** ���о�̬���� **************
	static std::string getTime(const time_t& t);  //��ȡʱ��
	static std::string num2str(int i);
	static void drawRect(cv::Mat& frame, const caffe::Frcnn::BBox<float>& box, bool drawOrder = true); //����
	static void drawRects(cv::Mat& frame, const std::vector<caffe::Frcnn::BBox<float> >& boxes, bool drawOrder = true);

private:
	//*****************˽�б�������**************
	const int CONTINUE_SECOND;   //��������CONTINUE_SECOND �����ͳ�ƣ�Ȼ���ж�
	const int K;   //����k֡,���ռ�ȼ�⵽��Ŀ�����ȵ�ǰȷ�����������࣬���������һֻ�����ڳ�����������Ϊ������һֻ��������Ϊ�ٻ�
	const float RATE;   //��������֡����ֵİٷֱ�

	std::list<Pig_Info> pig_info;   //����������K1֡ͼƬ��Ŀ�����Ŀ��ͼƬ��ʱ��
	std::map<unsigned, unsigned> M;    //ͳ�ƶ���Q1������������֡��������M1[4]=20 ��ʾ��20֡�����Ϊ4ֻ����

	unsigned long long realFrames;      //����ͳ�ƽ���Ŀ�����֡��
	
	unsigned int pigCounts;        //��ǰȷ���Ѿ���������ĸ�������ʼ��Ϊ0
	bool hasBorn;           // �Ƿ�ʼ���䣬����ȷ���Ƿ�Ϊ��һֻ����
	char status;                // a: ��һֻ����  b:�ǵ�һֻ����   c:�ٻ�   d����ʧ    e������   f:����
	cv::Rect bornArea;         //��������

	std::vector<caffe::Frcnn::BBox<float> > curResults, preResults;   //�ֱ𱣴���һ֡�͵�ǰ֡�ļ����

	static std::shared_ptr<FRCNN_API::Detector> obj_detect;  //Ŀ������

	const std::string CDWriter_NO;    //Ӳ�̿�¼�����
	const std::string IP;          //����ͷ��IP
	const unsigned CHANNEL_NO;                        //����ͷͨ����
	const std::string SAVED_FOLD;                      //����ͼƬ�ļ�������


	//**************  ˽����ͨ����  ********************
	std::string getPigCounts2string()const{    //����ǰ������תΪstring���ͣ�����2->"02",  11->"11"
		return (pigCounts < 10) ? ("0" + num2str(pigCounts)) : num2str(pigCounts);
	}
	void setBornArea(const int WIDTH, const int HEIGHT, const caffe::Frcnn::BBox<float>& big_pig_box);  //���ݴ���λ�����������
	bool isInBornArea(const caffe::Frcnn::BBox<float>& box);  //�Ƿ��ڳ�������
	bool atleast_one_in_born_area(const std::vector<caffe::Frcnn::BBox<float> >& boxes, caffe::Frcnn::BBox<float>& box);  //�Ƿ�������һֻ�����ڳ�����������ǣ������ڳ�������Ŀ�

	//*********************pigCounts���²���*************
	//����1��õĵ�ǰ�����������������K֡����rate�������ϵ�Ŀ�����Ϊn�����ж���ǰ������Ϊn
	unsigned computePigCountsByWay1();
	//����2��õĵ�ǰ������������������K֡������Ŀ������ľ�ֵ����Ϊ��ǰ��������
	unsigned computePigCountsByWay2();

	//************  ˽�о�̬����  **********
	static int getDis(const caffe::Frcnn::BBox<float>& lhs, caffe::Frcnn::BBox<float>& rhs); //��������ľ���
	//ʹ��ǰ֡�ļ��������һ֡�ļ����һһ��Ӧ
	static void pairBBox(const std::vector<caffe::Frcnn::BBox<float> >& lastBBox, std::vector<caffe::Frcnn::BBox<float> >& currentBBox);
	static int getMaxOfMap(const std::map<unsigned, unsigned>& lhs, int* count);  //���һ��map����ִ������������Լ�����ֵĴ���
	cv::Rect bbox2Rect(const caffe::Frcnn::BBox<float>& box){
		return cv::Rect(cv::Point(box[0], box[1]), cv::Point(box[2], box[3]));
	}
};

//ע���
namespace caffe
{
	namespace Frcnn{
		extern INSTANTIATE_CLASS(FrcnnProposalLayer);
		REGISTER_LAYER_CLASS(FrcnnProposal);
	}
	extern INSTANTIATE_CLASS(InputLayer);
	REGISTER_LAYER_CLASS(Input);

	extern INSTANTIATE_CLASS(SplitLayer);
	REGISTER_LAYER_CLASS(Split);

	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);

	extern INSTANTIATE_CLASS(InnerProductLayer);
	REGISTER_LAYER_CLASS(InnerProduct);

	extern INSTANTIATE_CLASS(DropoutLayer);
	REGISTER_LAYER_CLASS(Dropout);

	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);

	extern INSTANTIATE_CLASS(PReLULayer);
	REGISTER_LAYER_CLASS(PReLU);

	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);

	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);

	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);

	extern INSTANTIATE_CLASS(ROIPoolingLayer);
	REGISTER_LAYER_CLASS(ROIPooling);

	extern INSTANTIATE_CLASS(FlattenLayer);
	REGISTER_LAYER_CLASS(Flatten);

	extern INSTANTIATE_CLASS(ConcatLayer);
	REGISTER_LAYER_CLASS(Concat);

	extern INSTANTIATE_CLASS(ReshapeLayer);
	REGISTER_LAYER_CLASS(Reshape);

}

#endif