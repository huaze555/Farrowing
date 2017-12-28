#ifndef DIRECTION_JUDGE_H_
#define DIRECTION_JUDGE_H_
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdio.h>  //删除文件函数remove 需要此文件头
#include <cstdlib>
#include <io.h>       //判定文件夹是否存在
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;

namespace zxj{
	/* Pair (label, confidence) representing a prediction. */
	typedef std::pair<string, float> Prediction;

	class Classifier {
	public:
		Classifier(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file, bool useGPU = true){
			if (useGPU)
				Caffe::set_mode(Caffe::GPU);
			else
				Caffe::set_mode(Caffe::CPU);


			/* Load the network. */
			net_.reset(new Net<float>(model_file, TEST));
			net_->CopyTrainedLayersFrom(trained_file);

			CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
			CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

			Blob<float>* input_layer = net_->input_blobs()[0];
			num_channels_ = input_layer->channels();
			CHECK(num_channels_ == 3 || num_channels_ == 1)
				<< "Input layer should have 1 or 3 channels.";
			input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

			/* Load the binaryproto mean file. */
			SetMean(mean_file);

			/* Load labels. */
			std::ifstream labels(label_file.c_str());
			CHECK(labels) << "Unable to open labels file " << label_file;
			string line;
			while (std::getline(labels, line))
				labels_.push_back(string(line));

			Blob<float>* output_layer = net_->output_blobs()[0];
			CHECK_EQ(labels_.size(), output_layer->channels())
				<< "Number of labels is different from the output layer dimension.";
		}

		std::vector<Prediction> Classify(const cv::Mat& img, int N = 3){
			std::vector<float> output = Predict(img);
			N = std::min<int>(labels_.size(), N);
			std::vector<int> maxN = Argmax(output, N);
			std::vector<Prediction> predictions;
			for (int i = 0; i < N; ++i) {
				int idx = maxN[i];
				predictions.push_back(std::make_pair(labels_[idx], output[idx]));
			}
			return predictions;
		}

	private:
		void SetMean(const string& mean_file){
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

			/* Convert from BlobProto to Blob<float> */
			Blob<float> mean_blob;
			mean_blob.FromProto(blob_proto);
			CHECK_EQ(mean_blob.channels(), num_channels_)
				<< "Number of channels of mean file doesn't match input layer.";

			/* The format of the mean file is planar 32-bit float BGR or grayscale. */
			std::vector<cv::Mat> channels;
			float* data = mean_blob.mutable_cpu_data();
			for (int i = 0; i < num_channels_; ++i) {
				/* Extract an individual channel. */
				cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
				channels.push_back(channel);
				data += mean_blob.height() * mean_blob.width();
			}

			/* Merge the separate channels into a single image. */
			cv::Mat mean;
			cv::merge(channels, mean);

			/* Compute the global mean pixel value and create a mean image
			* filled with this value. */
			cv::Scalar channel_mean = cv::mean(mean);
			mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
		}
		std::vector<float> Predict(const cv::Mat& img){
			Blob<float>* input_layer = net_->input_blobs()[0];
			input_layer->Reshape(1, num_channels_,
				input_geometry_.height, input_geometry_.width);
			/* Forward dimension change to all layers. */
			net_->Reshape();

			std::vector<cv::Mat> input_channels;
			WrapInputLayer(&input_channels);

			Preprocess(img, &input_channels);

			net_->Forward();

			/* Copy the output layer to a std::vector */
			Blob<float>* output_layer = net_->output_blobs()[0];
			const float* begin = output_layer->cpu_data();
			const float* end = begin + output_layer->channels();
			return std::vector<float>(begin, end);
		}
		void WrapInputLayer(std::vector<cv::Mat>* input_channels){
			Blob<float>* input_layer = net_->input_blobs()[0];

			int width = input_layer->width();
			int height = input_layer->height();
			float* input_data = input_layer->mutable_cpu_data();
			for (int i = 0; i < input_layer->channels(); ++i) {
				cv::Mat channel(height, width, CV_32FC1, input_data);
				input_channels->push_back(channel);
				input_data += width * height;
			}
		}
		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels){
			/* Convert the input image to the input image format of the network. */
			cv::Mat sample;
			if (img.channels() == 3 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
			else if (img.channels() == 4 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
			else if (img.channels() == 4 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
			else if (img.channels() == 1 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
			else
				sample = img;

			cv::Mat sample_resized;
			if (sample.size() != input_geometry_)
				cv::resize(sample, sample_resized, input_geometry_);
			else
				sample_resized = sample;

			cv::Mat sample_float;
			if (num_channels_ == 3)
				sample_resized.convertTo(sample_float, CV_32FC3);
			else
				sample_resized.convertTo(sample_float, CV_32FC1);

			cv::Mat sample_normalized;
			cv::subtract(sample_float, mean_, sample_normalized);

			/* This operation will write the separate BGR planes directly to the
			* input layer of the network because it is wrapped by the cv::Mat
			* objects in input_channels. */
			cv::split(sample_normalized, *input_channels);

			CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
				== net_->input_blobs()[0]->cpu_data())
				<< "Input channels are not wrapping the input layer of the network.";
		}
		/* Return the indices of the top N values of vector v. */
		static std::vector<int> Argmax(const std::vector<float>& v, int N) {
			std::vector<std::pair<float, int> > pairs;
			for (size_t i = 0; i < v.size(); ++i)
				pairs.push_back(std::make_pair(v[i], i));
			std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

			std::vector<int> result;
			for (int i = 0; i < N; ++i)
				result.push_back(pairs[i].second);
			return result;
		}
		static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
			return lhs.first > rhs.first;
		}
	private:
		std::shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;
		std::vector<string> labels_;

	};
}
//*********************************分娩区域朝向分类器**************************
class DirectionJudge{
public:
	DirectionJudge(const string& model_file, const string& trained_file, const string& mean_file, bool useGPU = true){
		//确保三个配置文件都存在
		if (_access(model_file.c_str(), 0) == -1)
			throw string(model_file + " isn't exist!");

		if (_access(trained_file.c_str(), 0) == -1)
			throw string(trained_file + " isn't exist!");

		if (_access(mean_file.c_str(), 0) == -1)
			throw string(mean_file + " isn't exist!");

		const std::string LABEL_NAME = "label-name.txt";
		std::ofstream fout(LABEL_NAME);   //生成一个标签、名字对应文件，在程序结束后删除
		if (fout.is_open()){
			for (int i = 0; i < 6; ++i)
				fout << i << std::endl;
		}
		fout.close();
		this->cnn = std::make_shared<zxj::Classifier>(model_file, trained_file, mean_file, LABEL_NAME, useGPU); //CNN类 初始化
		remove(LABEL_NAME.c_str());
	}
	int judge(const cv::Mat& img){   //分娩区域在左边，返回0，分娩区域在右，返回1
		zxj::Prediction p = this->cnn->Classify(img)[0];
		string result = p.first;  //返回0或者1
		return atoi(result.c_str());
	}
	string num2str(float i){
		stringstream ss;
		ss << i;
		return ss.str();
	}
private:
	std::shared_ptr<zxj::Classifier> cnn;    //识别类
};

#endif
