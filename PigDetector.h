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

extern std::string CDWriter_NO;   //硬盘刻录机的所有通道的摄像头，公用同一个刻录机编号
extern const std::string PIG_DETECT_MODEL_FOLD;   //保存仔猪检测模型的文件夹
extern int CAMERA_NUM;        //实际测试的摄像头数量，从文件里读取

//将目标框数目、图片和时间绑定
struct Pig_Info
{
	cv::Mat img;          // 图片
	std::vector<caffe::Frcnn::BBox<float> > boxes;   // 目标框
	time_t t;             // 时间
	float weight;         //权重
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
	//***************** 公有变量部分 **************
	
	static std::map<unsigned, cv::Scalar> label_color; //不同类别对应的框的颜色
	cv::Rect big_pig_rect;              //大猪的框

	//***************** 公有函数接口 **************
	//构造函数，硬盘刻录机编号、摄像头通道号、输入IP、保存图片的文件夹
	PigDetector(const std::string& Writer_NO, unsigned camera_channel, const std::string& ip,
		const std::string& saved_fold = "saved_images");

	std::vector<caffe::Frcnn::BBox<float> > frameProcess(const cv::Mat& frame);           //对图片帧进行检测
	unsigned getPigCounts()const{ return pigCounts; }
	unsigned getChannelNo()const{ return CHANNEL_NO; }
	std::string getIP()const{ return IP; }
	cv::Rect getBornArea()const{ return bornArea; }
	
	//***************** 公有静态函数 **************
	static std::string getTime(const time_t& t);  //获取时间
	static std::string num2str(int i);
	static void drawRect(cv::Mat& frame, const caffe::Frcnn::BBox<float>& box, bool drawOrder = true); //画框
	static void drawRects(cv::Mat& frame, const std::vector<caffe::Frcnn::BBox<float> >& boxes, bool drawOrder = true);

private:
	//*****************私有变量部分**************
	const int CONTINUE_SECOND;   //对连续的CONTINUE_SECOND 秒进行统计，然后判断
	const int K;   //连续k帧,最大占比检测到的目标数比当前确定的仔猪数多，如果至少有一只仔猪在出生区域，则认为出生了一只仔猪，否则为召回
	const float RATE;   //在连续的帧里，出现的百分比

	std::list<Pig_Info> pig_info;   //保存连续的K1帧图片，目标框数目、图片和时间
	std::map<unsigned, unsigned> M;    //统计队列Q1各仔猪数量的帧数，比如M1[4]=20 表示有20帧检测结果为4只仔猪

	unsigned long long realFrames;      //用于统计进行目标检测的帧数
	
	unsigned int pigCounts;        //当前确定已经出生仔猪的个数，初始化为0
	bool hasBorn;           // 是否开始分娩，用于确定是否为第一只仔猪
	char status;                // a: 第一只出生  b:非第一只出生   c:召回   d：消失    e：死猪   f:包衣
	cv::Rect bornArea;         //出生区域

	std::vector<caffe::Frcnn::BBox<float> > curResults, preResults;   //分别保存上一帧和当前帧的检测结果

	static std::shared_ptr<FRCNN_API::Detector> obj_detect;  //目标检测器

	const std::string CDWriter_NO;    //硬盘刻录机编号
	const std::string IP;          //摄像头的IP
	const unsigned CHANNEL_NO;                        //摄像头通道号
	const std::string SAVED_FOLD;                      //保存图片文件夹名称


	//**************  私有普通函数  ********************
	std::string getPigCounts2string()const{    //将当前仔猪数转为string类型，比如2->"02",  11->"11"
		return (pigCounts < 10) ? ("0" + num2str(pigCounts)) : num2str(pigCounts);
	}
	void setBornArea(const int WIDTH, const int HEIGHT, const caffe::Frcnn::BBox<float>& big_pig_box);  //根据大猪位置求出生区域
	bool isInBornArea(const caffe::Frcnn::BBox<float>& box);  //是否在出生区域
	bool atleast_one_in_born_area(const std::vector<caffe::Frcnn::BBox<float> >& boxes, caffe::Frcnn::BBox<float>& box);  //是否至少有一只仔猪在出生区域，如果是，返回在出生区域的框

	//*********************pigCounts更新策略*************
	//方案1求得的当前仔猪数量：如果连续K帧中有rate比例以上的目标框数为n，则判定当前仔猪数为n
	unsigned computePigCountsByWay1();
	//方案2求得的当前仔猪数量：讲连续的K帧检测出的目标框数的均值，作为当前仔猪数量
	unsigned computePigCountsByWay2();

	//************  私有静态函数  **********
	static int getDis(const caffe::Frcnn::BBox<float>& lhs, caffe::Frcnn::BBox<float>& rhs); //求两个框的距离
	//使当前帧的检测结果与上一帧的检测结果一一对应
	static void pairBBox(const std::vector<caffe::Frcnn::BBox<float> >& lastBBox, std::vector<caffe::Frcnn::BBox<float> >& currentBBox);
	static int getMaxOfMap(const std::map<unsigned, unsigned>& lhs, int* count);  //求出一个map里出现次数最多的数，以及其出现的次数
	cv::Rect bbox2Rect(const caffe::Frcnn::BBox<float>& box){
		return cv::Rect(cv::Point(box[0], box[1]), cv::Point(box[2], box[3]));
	}
};

//注册层
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