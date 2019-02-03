/**
* CS4185 Multimedia Technologies and Applications
* Course Assignment
* Image Retrieval Project
*/

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <math.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector>
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <fstream>

using namespace std;
using namespace cv;

#define IMAGE_LIST_FILE "inputimage.txt"  // NOTE: this is relative to current file
#define WINDOW_NAME "CVUI Test"

// human feature detector

CascadeClassifier face_cascade = CascadeClassifier("haarcascade_frontalface_alt2.xml");
CascadeClassifier nose_cascade = CascadeClassifier("haarcascade_mcs_nose.xml");
CascadeClassifier eye0_cascade = CascadeClassifier("haarcascade_lefteye_2splits.xml");
CascadeClassifier eye1_cascade = CascadeClassifier("haarcascade_eye.xml");

double humanFeatureDetector(Mat &img1) {
	vector<Rect> rect0, rect1, rect2, rect3;
	face_cascade.detectMultiScale(img1, rect0, 1.5, 4, 0);
	nose_cascade.detectMultiScale(img1, rect1, 1.5, 4, 0);
	eye0_cascade.detectMultiScale(img1, rect2, 1.5, 4, 0);

	double result = 0;
	if (rect0.size() > 0)result += 2;
	if (rect1.size() > 0)result += 1;
	if (rect2.size() > 0)result += 2;

	return result;
}

//string replace
void string_replace(std::string &strBig, const std::string &strsrc, const std::string &strdst)
{
	std::string::size_type pos = 0;
	std::string::size_type srclen = strsrc.size();
	std::string::size_type dstlen = strdst.size();
	while ((pos = strBig.find(strsrc, pos)) != std::string::npos) {
		strBig.replace(pos, srclen, strdst);
		pos += dstlen;
	}
}

//split string
void split_str_to_double(vector<double>& result, string str, vector<char> delimiters){

	result.clear();
	auto start = 0;
	while (start < str.size())
	{
		//根据多个分割符分割
		auto itRes = str.find(delimiters[0], start);
		for (int i = 1; i < delimiters.size(); ++i)
		{
			auto it = str.find(delimiters[i], start);
			if (it < itRes)
				itRes = it;
		}
		if (itRes == string::npos)
		{
			result.push_back(atof(str.substr(start, str.size() - start).c_str()));
			break;
		}
		result.push_back(atof(str.substr(start, itRes - start).c_str()));
		start = itRes;
		++start;
	}
}

class MyImg{

private:
	//cosine similarity
	double cosine_similarity(vector<double> &A,vector<double> &B, unsigned int size){

		double mul = 0.0, d_a = 0.0, d_b = 0.0;
		for (unsigned int i = 0; i < size; ++i)
		{
			mul += A[i] * B[i];
			d_a += A[i] * A[i];
			d_b += B[i] * B[i];
		}
		return mul / (sqrt(d_a) * sqrt(d_b));
	}
	//euclid distance
	double euclidDist(vector<double> &A, vector<double> &B, unsigned int size) {
		double sum = 0;
		for (unsigned int i = 0; i < size; ++i)
		{
			double tmp = pow(A[i] - B[i], 2);
			sum += tmp;
		}
		return sqrt(sum);
	}

	//avaerage hash
	string getHashValue(Mat &src){
		string rst(64, '\0');
		Mat img;
		if (src.channels() == 3)
			cvtColor(src, img, CV_BGR2GRAY);
		else
			img = src.clone();
		resize(img, img, Size(8, 8));
		uchar *pData;
		for (int i = 0; i<img.rows; i++)
		{
			pData = img.ptr<uchar>(i);
			for (int j = 0; j<img.cols; j++)
			{
				pData[j] = pData[j] / 4;
			}
		}
		int average = mean(img).val[0];
		Mat mask = (img >= (uchar)average);
		int index = 0;
		for (int i = 0; i<mask.rows; i++)
		{
			pData = mask.ptr<uchar>(i);
			for (int j = 0; j<mask.cols; j++)
			{
				if (pData[j] == 0)
					rst[index++] = '0';
				else
					rst[index++] = '1';
			}
		}
		return rst;
	}

	int HanmingDistance(string &str1, string &str2){
		if ((str1.size() != 64) || (str2.size() != 64))
			return -1;
		int difference = 0;
		for (int i = 0; i<64; i++){
			if (str1[i] != str2[i])
				difference++;
		}
		return difference;
	}

public:
		Mat img;
		string name;
		double score;
		int label;
		double vice_score; //备用分数 
		MyImg() {
			img = Mat();
			string c = "";
			name = c;
			score = 0;
			label = -1;
			vice_score = 0;
		}
		MyImg(Mat input,string filename) {
			img = input;
			name = filename;
			score = 0;
			vice_score = 0;
		}
		MyImg copy() {
			MyImg tmp = MyImg();
			tmp.name = name;
			tmp.score = score;
			tmp.img = img.clone();
			tmp.label = label;
			tmp.vice_score = vice_score;
			return tmp;
		}
		//去除前后缀 获得数字
		void modifyName() {
			string_replace(name, ".jpg", "");
			string_replace(name, "../image.orig\\", "");
		}

		//获得标签 get label
		void getMyLabel() {
			//man 0-99 beach 100-199 building 200-299 bus 300-399 
			//dinosaur 400-499 elephant 500-599 flower 600-699
			//horse 700-799 mountain 800-899 food 900-999
			int num = atoi(name.c_str());
			if (num >= 0 && num <= 99)label = 6;
			else if (num >= 100 && num <= 199)label = 0;
			else if (num >= 200 && num <= 299)label = 1;
			else if (num >= 300 && num <= 399)label = 2;
			else if (num >= 400 && num <= 499)label = 3;
			else if (num >= 500 && num <= 599)label = 7;
			else if (num >= 600 && num <= 699)label = 4;
			else if (num >= 700 && num <= 799)label = 5;
			else if (num >= 800 && num <= 899)label = 8;
			else if (num >= 900 && num <= 999)label = 9;
		}

		double compareHistValue2(MyImg &input) {
			Mat img2 = input.img;
			MatND hist1;
			MatND hist2;
			Mat new_img2 = img2.clone();
			resize(new_img2, img2,img.size());
			Mat hsv_img1, hsv_img2;
			cvtColor(img, hsv_img1, CV_BGR2HSV);
			cvtColor(img2, hsv_img2, CV_BGR2HSV);
			int h_bins = 50; int s_bins = 32;
			int histSize[] = { h_bins, s_bins };
			float h_ranges[] = { 0, 256 };
			float s_ranges[] = { 0, 180 };
			const float* ranges[] = { h_ranges, s_ranges };
			int channels[] = { 0, 1 };
			calcHist(&img, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
			normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
			calcHist(&img2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
			normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
			return compareHist(hist1, hist2, 3);
		}

		double avgHash(MyImg &input) {
			string hv1 = getHashValue(img);
			string hv2 = getHashValue(input.img);
			return HanmingDistance(hv1, hv2);
		}

		double perHash(MyImg &input) {
			cv::Mat matSrc1, matSrc2;
			matSrc1 = img;
			matSrc2 = input.img;
			cv::Mat matDst1, matDst2;
			cv::resize(matSrc1, matDst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
			cv::resize(matSrc2, matDst2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
			//2.简化色彩
			//将缩小后的图片，转为64级灰度。也就是说，所有像素点总共只有64种颜色。
			cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
			cv::cvtColor(matDst2, matDst2, CV_BGR2GRAY);
			//3.计算平均值
			//计算所有64个像素的灰度平均值。
			int iAvg1 = 0, iAvg2 = 0;
			int arr1[64], arr2[64];
			for (int i = 0; i < 8; i++) {
				uchar* data1 = matDst1.ptr<uchar>(i);
				uchar* data2 = matDst2.ptr<uchar>(i);

				int tmp = i * 8;

				for (int j = 0; j < 8; j++) {
					int tmp1 = tmp + j;

					arr1[tmp1] = data1[j] / 4 * 4;
					arr2[tmp1] = data2[j] / 4 * 4;

					iAvg1 += arr1[tmp1];
					iAvg2 += arr2[tmp1];
				}
			}
			iAvg1 /= 64;
			iAvg2 /= 64;
			//4.比较像素的灰度
			//将每个像素的灰度，与平均值进行比较。大于或等于平均值，记为1；小于平均值，记为0。
			for (int i = 0; i < 64; i++) {
				arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
				arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
			}
			//5.计算哈希值
			//将上一步的比较结果，组合在一起，就构成了一个64位的整数，这就是这张图片的指纹。
			//组合的次序并不重要，只要保证所有图片都采用同样次序就行了。
			int iDiffNum = 0;

			for (int i = 0; i < 64; i++)
				if (arr1[i] != arr2[i])
					++iDiffNum;

			return iDiffNum;
		}

		double compareByHistogram(MyImg &input){

			Mat hsv_base = img.clone();
			Mat hsv_test;
			Mat src_input2 = input.img.clone();
			cvtColor(src_input2, hsv_test, COLOR_BGR2HSV);
			cvtColor(hsv_base, hsv_base, COLOR_BGR2HSV);
			int h_bins = 50; int s_bins = 60;
			int histSize[] = { h_bins, s_bins };
			// hue varies from 0 to 179, saturation from 0 to 255
			float h_ranges[] = { 0, 180 };
			float s_ranges[] = { 0, 256 };
			const float* ranges[] = { h_ranges, s_ranges };
			// Use the o-th and 1-st channels
			int channels[] = { 0, 1 };

			MatND hist_base;
			MatND hist_test;

			calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
			normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

			calcHist(&hsv_test, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false);
			normalize(hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat());
			return compareHist(hist_base, hist_test, CV_COMP_BHATTACHARYYA);
		}

		double getSimilarity(MyImg &input) {
			/*return avgHash(input)/64;*/
			double val1 = compareByHistogram(input)*30;
			double val2 = perHash(input);
			return val1+val2;
		}

		double getSimiByPhash(MyImg &input) {
			return perHash(input)/64;
		}

		double getSimiByHist(MyImg &input) {
			return compareByHistogram(input);
		}

		double getSimiByHumanFeature(MyImg &input) {
			double v1 = humanFeatureDetector(input.img);
			double v2 = humanFeatureDetector(img);

			return abs(v1 - v2);
		}

		double getSimilarityByFeatureVec(vector<double> &mine,vector<double> &compareTarget) {
			double result = euclidDist(mine, compareTarget,mine.size());
			//double result = cosine_similarity(mine, compareTarget, mine.size());
			//result = -((result + 1) / 2) + 1; //normalize
			return result;
		}

		double compareImgs_SIFT(Mat img_2)
		{
			//-- Step 1: Detect the keypoints using SURF Detector
			int minHessian = 400;

			SiftFeatureDetector detector;

			std::vector<KeyPoint> keypoints_1, keypoints_2;

			detector.detect(img, keypoints_1);
			detector.detect(img_2, keypoints_2);

			//-- Step 2: Calculate descriptors (feature vectors)
			SurfDescriptorExtractor extractor;

			Mat descriptors_1, descriptors_2;

			extractor.compute(img, keypoints_1, descriptors_1);
			extractor.compute(img_2, keypoints_2, descriptors_2);

			//-- Step 3: Matching descriptor vectors with a brute force matcher
			BFMatcher matcher(NORM_L2);
			std::vector< DMatch > matches;
			matcher.match(descriptors_1, descriptors_2, matches);

			double distance_sum = 0;
			//for (int i = 0; i < matches.size(); i++)
			//{
			//	distance_sum += matches[i].distance;
			//}

			//distance_sum = 0;
			for (int i = 0; i < matches.size(); i++)
			{
				if (matches[i].distance < 0.2)
				{
					//L1 norm
					distance_sum += (1 - matches[i].distance);

					//l2 norm
					//distance_sum += pow((1 - matches[i].distance), 2);
				}
			}

			//L2

			cout << distance_sum << endl;

			////-- Draw matches
			//Mat img_matches;
			//drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

			////-- Show detected matches
			//cv::imshow("Matches", img_matches);

			//waitKey(0);

			return 1 / distance_sum;
		}

		//double getSimilarity2(MyImg &input, vector<double> &mine, vector<double> &compareTarget) {
		//	double result2 = compareByHistogram(input);
		//	double result = euclidDist(mine, compareTarget, mine.size());
		//	return sqrt(pow(result, 2) + pow(result2, 2));
		//}
		double getSimiBySIFT(MyImg &img) {
			return compareImgs_SIFT(img.img);
		}

		double compareImgs_HOG(Mat img2, CvSize sz)
		{
			Mat new_img1;
			resize(img, new_img1, sz);
			Mat new_img2;
			resize(img2, new_img2, sz);

			//blur(new_img1, new_img1, Size(6, 6));
			//blur(new_img2, new_img2, Size(6, 6));

			//imshow("al1", new_img1);
			//imshow("al2", new_img2);
			//waitKey(0);

			HOGDescriptor *hog = new HOGDescriptor(sz, cvSize(16, 16), cvSize(16, 16), cvSize(8, 8), 16);
			vector<float> feature_v1, feature_v2;

			hog->compute(new_img1, feature_v1, Size(16, 16), Size(0, 0));
			hog->compute(new_img2, feature_v2, Size(16, 16), Size(0, 0));

			normalize(feature_v1, feature_v1, 0, 1, NORM_MINMAX, -1, Mat());
			normalize(feature_v2, feature_v2, 0, 1, NORM_MINMAX, -1, Mat());
			double sum = 0;
			for (int i = 0; i < feature_v1.size(); i++)
			{
				double a = feature_v1[i], b = feature_v2[i];
				double thresholdH = 0;
				if (feature_v1[i] < thresholdH)
				{
					a = 0;
				}
				if (feature_v2[i] < thresholdH)
				{
					b = 0;
				}
				sum += abs(a - b);
			}
			//sum = sqrt(sum);
			cout << sum << endl;
			return sum;
		}

		double getSimiByHOG(MyImg &img) {
			return compareImgs_HOG(img.img, cv::Size(640, 480));
		}
};

//sortfunction 
bool compareImgScore(MyImg &a, MyImg &b) {
	return a.score < b.score;
}

void selectImg(Mat &input,Mat &selectedDisp,string path) {  //设置选中图片效果
	selectedDisp = imread(path);
	input = imread(path);
	resize(selectedDisp, selectedDisp,Size(130, 130));
}

//load Feature Vector
void loadFeatureVector(vector<vector<double>> &input, vector<vector<double>> &db) {

	ifstream infile("7n3.txt");
	string line;
	vector<char> token;
	token.push_back(' ');

	while (getline(infile, line)) {
		vector<double> cnm;
		split_str_to_double(cnm, line, token);
		input.push_back(cnm);
	}
	infile.close();

	ifstream infile2("1000n3.txt");
	while (getline(infile2, line)) {
		vector<double> cnm;
		split_str_to_double(cnm, line, token);
		db.push_back(cnm);
	}
	infile2.close();
}

//global variable
MyImg IMGDB[1000];
bool isloaded = false;

vector<vector<double>> inputVec, dbVec; // feature vector
bool featureIsLoaded = false;

void normalize(vector<double> &a) {  //normalize a vector
	std::vector<double>::iterator biggest1 = max_element(a.begin(), a.end());
	for (int i = 0; i < a.size(); i++) {
		a[i] = a[i] * (1 / (*biggest1));
	}
}

void loadImageDB() {

	FILE* fp;
	char imagepath[200];
	fopen_s(&fp, IMAGE_LIST_FILE, "r");
	printf("Loading images...\n");
	Mat db_img;
	int db_id = 0;

	while (!feof(fp))
	{
		while (fscanf_s(fp, "%s ", imagepath, sizeof(imagepath)) > 0)
		{
			printf("%s\n", imagepath);
			char tempname[200];
			sprintf_s(tempname, 200, "../%s", imagepath);

			db_img = imread(tempname); // read database image
			if (!db_img.data)
			{
				printf("Cannot find the database image number %d!\n", db_id + 1);
				system("pause");
				return;
			}
			string name(tempname);
			string_replace(name, ".jpg", "");
			string_replace(name, "../image.orig\\", "");
			MyImg dbimg(db_img, name);
			IMGDB[db_id] = dbimg;
			db_id++;
		}
	}
	fclose(fp);
}

//Run Retireval Here return myImg Array
void runRetrieval(int inputLabel,Mat &input,Mat &bestMatch,MyImg *set,int type){ //运行提取算法

	//initialize
	bestMatch = Mat();
	Mat src_input = input.clone();
	if (inputLabel == -1 || input.empty())return;
	////vector features of input and data base picture,public variable

	
	
	if (type == 2) { //type2 calculate with featrue vector
		if (!featureIsLoaded) {
			cout << "Loading Vectors.....";
			loadFeatureVector(inputVec, dbVec);
			if (!(inputVec.size() > 0 && dbVec.size() > 0)) {
				cout << "fail to load vector!!\n";
				return;
			}
			featureIsLoaded = true;
		}
	}
	if (!isloaded) {
		loadImageDB();
		isloaded = true;
	}
	for (int i = 0; i < 999; i++) {
		set[i] = IMGDB[i].copy();
	}
	
	//run here
	MyImg src_img(src_input, "input");
	if (type == 1) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type1 " << set[i].name<<"\n";
			set[i].score = src_img.getSimilarity(set[i]);
		}
	}
	/// type1 hist or hash
	else if (type == 2) {
		/*vector<double> vecsimi, histval;
		for (int i = 0; i < 999; i++) {
			cout << "computing: type2 " << set[i].name << "\n";
			vecsimi.push_back(src_img.compareByHistogram(set[i]));
			histval.push_back(src_img.getSimilarityByFeatureVec(inputVec[inputLabel], dbVec[atoi(set[i].name.c_str())]));
		}
		normalize(vecsimi);
		normalize(histval);
		for (int i = 0; i < 999; i++) {
			set[i].score = sqrt(pow(vecsimi[i],2)+pow(histval[i],2));
		}*/
		for (int i = 0; i < 999; i++) {
			cout << "computing: type: " <<type << " " << set[i].name << "\n";
			set[i].score = src_img.getSimilarityByFeatureVec(inputVec[inputLabel], dbVec[atoi(set[i].name.c_str())]);
		}
	}

	else if (type == 3) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type " <<type << " " << set[i].name << "\n";
			set[i].score = src_img.getSimiByHumanFeature(set[i]);
		}
	}

	else if (type == 4) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type " << type << " " << set[i].name << "\n";
			set[i].score = src_img.getSimiByHist(set[i]);
		}
	}

	else if (type == 5) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type " << type<<" "<< set[i].name << "\n";
			set[i].score = src_img.getSimiByPhash(set[i]);
		}
	}

	else if (type == 6) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type " << type << " " << set[i].name << "\n";
			set[i].score = src_img.getSimiBySIFT(set[i]);
		}
	}

	else if (type == 7) {
		for (int i = 0; i < 999; i++) {
			cout << "computing: type " << type << " " << set[i].name << "\n";
			set[i].score = src_img.getSimiByHOG(set[i]);
		}
	}

	sort(set, set + 999, compareImgScore);
	cout << "score:" << set[0].score << "\n" << set[0].name << '\n';
	printf("Done \n");

	// handle skewed problem
	double biggest = set[998].score;
	cout << "biggest score:" << biggest << "\n";
	for (int i = 0; i < 1000; i++) {
		set[i].score = set[i].score*(1 / biggest);
	}

	Mat maximg = set[0].img;
	//for (int i = 0; i < 10; i++) {
	//	imshow(set[i].name, set[i].img);
	//	cout << set[i].score<<"\n";
	//	waitKey(0);
	//}
	bestMatch = maximg.clone();
}

//Filter by a Threshold
int filterByThreshold(MyImg *set,MyImg *result,double threshold) {   //使用一个阈值进行筛选
	int pos = 0;
	for (int i = 0; i < 1000; i++) {
		if (set[i].score > threshold) {
			pos = i;
			break;
		}
	}
	if (pos == 0) pos = 1000;
	for (int i = 0; i < pos; i++) {
		result[i] = set[i].copy();
	}
	return pos;
}

// print several pics
void printMany(Mat &frame,MyImg *set,int num) {

	if (num >= 135)num = 135; //clipped

	int X = 310, Y = 80;
	for (int i = 0; i < num; i++) {
		if (set != NULL) {
			Mat tmp = set[i].img.clone();
			if (!tmp.empty()) {
				resize(tmp, tmp, Size(50, 50));
				cvui::image(frame, X, Y, tmp);
			}
			X += 50;
			if ((i + 1) % 15 == 0) {
				X = 310;
				Y += 50;
			}
		}
	}
}

//get percision and recall
void getPercisionAndRecall(double &percision,double &recall,int targetLabel,MyImg *filtered,int length) {
	int count = 0;
	for (int i = 0; i < length; i++) {
		filtered[i].getMyLabel();
		if (filtered[i].label == targetLabel)count++;
	}
	cout << targetLabel;
	percision = (double)count / (double)length;
	recall = (double)count / (double)100;
}

int main(int argc, char** argv){

	cout << "welcome to our image retrieval system\n";
	system("pause");
	cvui::init(WINDOW_NAME);
	cv::Mat frame = cv::Mat(cv::Size(1088, 612), CV_8UC3);

	int appType = 0; //0 main 1 detail 2 spider

	string searchResultPath = "./rs/";

	//Input for Retrieval
	Mat inputImg = Mat();
	int inputLabel = -1; //IMPORTANT!! IT'S USED FOR RESULT ANALYSIS,NOT FOR CHEATING

	//selectedImg
	Mat selectedImg = Mat();

	//static Variable
	string remind = "Wait For Order";
	int GraphicDispType = 0;
	double chosenThreshold = 0;

	//Running Order 它说明你选择了哪种算法来执行图像提取
	int runningOrder = 0;

	//Search Result
	Mat bestMatch = Mat();
	MyImg set[1000]; // pointer points to an array
	MyImg filtered[1000];
	int filteredLen = 0;
	double percision = 0;
	double recall = 0;

	while (true) {
		
		if (appType == 0) {

			frame = cv::Scalar(49, 52, 49);
			cvui::text(frame, 300, 10, "Image Retrieval System", 1.0);

			int bottomBias = 70;

			//window
			cvui::window(frame, 300, 60, 768, 500, "Graphic Display");
			cvui::window(frame, 120,135, 150, 165, "You Just Selected:");
			cvui::window(frame, 10, 540, 270, 60, "Remind Msg:");

			//如果已经下达运行命令 在此运行
			if (runningOrder) {
				cout << "running\n";

				//初始化
				bestMatch = Mat();
				filteredLen = 0;
				//use running function here
				if (inputImg.empty()) {
					remind = "Select A Picture First!!!!";
				}
				else {
					runRetrieval(inputLabel,inputImg,bestMatch,set,runningOrder);
					remind = "Retrival Done! This the BestMatch";
					GraphicDispType = 1;
				}
				runningOrder = 0;
				
			}

			//threshold trackbar
			cvui::trackbar(frame, 40, 470, 220, &chosenThreshold, (double)0, (double)1);
			cvui::text(frame, 40, 520, "You select a threshold:"+to_string(chosenThreshold), 0.4);

			//select Pictures
			//man 0-99 beach 100-199 building 200-299 bus 300-399 
			//dinosaur 400-499 elephant 500-599 flower 600-699
			//horse 700-799 mountain 800-899 food 900-999
			if (cvui::button(frame, 20, bottomBias + 30, "  Beach ")) {
				selectImg(inputImg, selectedImg, "beach.jpg");
				inputLabel = 0;
			}
			if (cvui::button(frame, 20, bottomBias + 60, "   Man  ")) {
				selectImg(inputImg, selectedImg, "man.jpg");
				inputLabel = 6;
			}
			if (cvui::button(frame, 20, bottomBias + 90, "  Horse ")) {
				selectImg(inputImg, selectedImg, "horse.jpg");
				inputLabel = 5;
			}
			if (cvui::button(frame, 20, bottomBias + 120, "Building")) {
				selectImg(inputImg, selectedImg, "building.jpg ");
				inputLabel = 1;
			}
			if (cvui::button(frame, 20, bottomBias + 150, "Dinosaur")) {
				selectImg(inputImg, selectedImg, "dinosaur.jpg");
				inputLabel = 3;
			}
			if (cvui::button(frame, 20, bottomBias + 180, " Flower ")) {
				selectImg(inputImg, selectedImg, "flower.jpg");
				inputLabel = 4;
			}
			if (cvui::button(frame, 20, bottomBias + 210, "  BUS   ")) {
				selectImg(inputImg, selectedImg, "bus.jpg");
				inputLabel = 2;
			}

			//select DisplayMode
			if (cvui::button(frame, 300, 570, "     Best Match      ")) {
				GraphicDispType = 1;
				for (int i = 0; i < 1000; i++) {
					cout << set[i].score << " ";
				}
			}
			if (cvui::button(frame, 480, 570, "     Top 100      ")) {
				GraphicDispType = 2;
				getPercisionAndRecall(percision, recall, inputLabel, set, 100);
			}
			if (cvui::button(frame, 640, 570, "Filter By Similarity")) {
				GraphicDispType = 3;
				filteredLen = filterByThreshold(set, filtered, chosenThreshold);
				getPercisionAndRecall(percision,recall,inputLabel,filtered,filteredLen);
			}
			if (cvui::button(frame, 790, 570, "Save Similar Picture")) {
				remind = "Result saved!";
				if (filteredLen != 0) {
					for (int i = 0; i < filteredLen; i++) {
						string fp = searchResultPath + to_string(i) + ".jpg";
						imwrite(fp, filtered[i].img);
					}
					cout << "Result saved!\n";
				}
				else {
					cout << "Result Set is empty!!";
				}
			}

			//select Algorithms and Run!
			if (cvui::button(frame,120, 320, "Color-Historgram")) {
				remind = "I AM Running!";
				runningOrder = 4;
			}
			if (cvui::button(frame, 120, 350, "pHash")) {
				remind = "I AM Running!";
				runningOrder = 5;
			}
			if (cvui::button(frame, 35, 380, "SIFT")) {
				remind = "I AM Running!";
				runningOrder = 6;
			}
			if (cvui::button(frame, 35, 410, "HOG")) {
				remind = "I AM Running!";
				runningOrder = 7;
			}
			if (cvui::button(frame, 120, 380, "ColorHistogram+pHash")) {
				remind = "I AM Running!";
				runningOrder = 1;
			}
			if (cvui::button(frame, 120, 410, "Deep Feature method")) {
				remind = "I AM Running!";
				runningOrder = 2;
			}
			if (cvui::button(frame, 120, 440, "HumanFace Feature")) {
				remind = "I AM Running!";
				runningOrder = 3;
			}

			//setSelectedImg
			if (!selectedImg.empty()) {
				cvui::image(frame, 130,160, selectedImg);
			}

			//DisplayResult 
			if (GraphicDispType) {
				if (GraphicDispType == 1) { //display BestMatch
					if (!bestMatch.empty()) {
						resize(bestMatch, bestMatch, Size(720,480));
						cvui::image(frame, 320, 80, bestMatch);
					}
					remind = "This is the best match";
				}
				else if (GraphicDispType == 2) { // see Top 100 similar
					printMany(frame, set, 100);
					remind = "top100:Recall:" + to_string(recall) +"Percision:" + to_string(percision);
				}
				else if (GraphicDispType == 3) { //filterBy Similarity
					if (filteredLen == 0) {
						//donothing
					}
					else {
						printMany(frame, filtered, filteredLen);
					}
					remind = "Recall:"+to_string(recall)+" "+" Percision:"+to_string(percision);
				}
			}

			//display text on frame
			//title 
			string guide1 = "Please Select a Target For Searching";
			cvui::text(frame, 20, 60, guide1, 0.4);
			cvui::text(frame, 15, 570, remind, 0.4);

			//finally render this app window
			cvui::imshow(WINDOW_NAME, frame);
		}
	
		if (cv::waitKey(20) == 27) {
			break;
		}
	}

	return 0;

	return 0;
}
