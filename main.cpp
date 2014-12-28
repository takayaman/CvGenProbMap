/*=============================================================================
 * Project : CvGenProbMap
 * Code : main.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/25
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Generate probability map for GrabCut
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "em.h"

/*=== Local Define / Local Const ============================================*/

/*=== Local Variable ========================================================*/
typedef cv::Vec<double_t, 5> SampleData;

typedef enum CalcEMMode_TAG {
  MODE_DEFAULT = 0,
  MODE_XY,
  MODE_COLOR,
  MODE_NUM
} CalcEMMode;

/* x, y, b, g, r */
static std::vector<SampleData> fore_data;
static std::vector<SampleData> back_data;

/*=== Local Function Define =================================================*/

bool generateEMModel(const std::vector<SampleData> &data,
                     CalcEMMode mode,
                     const cvgenprobmap_base::Em::Params &emparams,
                     cvgenprobmap_base::Em &emmodel, cv::Mat &loglikelihood,
                     cv::Mat &labels, cv::Mat &probabilities);

bool generateEMModel(const cv::Mat &data,
                     const cvgenprobmap_base::Em::Params &emparams,
                     cvgenprobmap_base::Em &emmodel, cv::Mat &loglikelihood,
                     cv::Mat &labels, cv::Mat &probabilities);

bool generateLikelihoodImage(const cvgenprobmap_base::Em &emmodel,
                             const cv::Mat &reference_image,
                             CalcEMMode mode, cv::Mat &likelihood_image);

bool generateProbabilityImage(const cvgenprobmap_base::Em &emmodel,
                              const cv::Mat &reference_image,
                              CalcEMMode mode, cv::Mat &probability_image);

/*=== Local Function Implementation =========================================*/

bool generateEMModel(const std::vector<SampleData> &data,
                     CalcEMMode mode,
                     const cvgenprobmap_base::Em::Params &emparams,
                     cvgenprobmap_base::Em &emmodel,
                     cv::Mat &loglikelihood,
                     cv::Mat &labels, cv::Mat &probabilities) {
  cv::Mat datamat;
  if(MODE_XY == mode) {
    datamat = cv::Mat(data.size(), 2, CV_64FC1);
    for(int32_t row = 0; row < datamat.rows; row++) {
      for(int32_t col = 0; col < datamat.cols; col++) {
        datamat.at<double_t>(row, col) = data[row].val[col];
      }
    }
  } else if(MODE_COLOR == mode) {
    datamat = cv::Mat(data.size(), 3, CV_64FC1);
    for(int32_t row = 0; row < datamat.rows; row++) {
      for(int32_t col = 0; col < datamat.cols; col++) {
        datamat.at<double_t>(row, col) = data[row].val[col + 2];
      }
    }
  }
  return generateEMModel(datamat, emparams, emmodel, loglikelihood, labels,
                         probabilities);
}

bool generateEMModel(const cv::Mat &data,
                     const cvgenprobmap_base::Em::Params &emparams,
                     cvgenprobmap_base::Em &emmodel,
                     cv::Mat &loglikelihood,
                     cv::Mat &labels, cv::Mat &probabilities) {
  if(data.empty()) {
    LOG(ERROR) << "data is empty!!" << std::endl;
    return false;
  }

  bool result = emmodel.train(data, loglikelihood, labels, probabilities,
                              emparams);
  if(!result) {
    LOG(ERROR) << "Failed to cv::ml::EM::train()!!" << std::endl;
    return false;
  }
  return true;
}

bool generateLikelihoodImage(const cvgenprobmap_base::Em &emmodel,
                             const cv::Mat &reference_image,
                             CalcEMMode mode, cv::Mat &likelihood_image) {
  if(reference_image.empty()) {
    LOG(ERROR) << "reference_image must not be empty!!" << std::endl;
    return false;
  }
  cv::Mat check = emmodel.getMeans();
  if(MODE_XY == mode) {
    if(2 != check.cols) {
      LOG(ERROR) << "p_emmodel is traind by " << check.cols << " dimentions vectors.\n"
                 << "This model do not match to MODE_XY" << std::endl;
      return false;
    }
  } else if(MODE_COLOR == mode) {
    if(3 != check.cols) {
      LOG(ERROR) << "p_emmodel is traind by " << check.cols << " dimentions vectors.\n"
                 << "This model do not match to MODE_COLOR" << std::endl;
      return false;
    }
  }

  int32_t rows = reference_image.rows;
  int32_t cols = reference_image.cols;
  cv::Mat predict_sample;
  cv::Mat predict_result;
  cv::Mat predict_image = cv::Mat::zeros(reference_image.rows,
                                         reference_image.cols, CV_64FC1);
  likelihood_image = cv::Mat::zeros(reference_image.rows,
                                    reference_image.cols, CV_8UC1);
  double_t maxlikelihood = -DBL_MAX;
  double_t minlikelihood = DBL_MAX;
  if(MODE_XY == mode) {
    predict_sample = cv::Mat(1, 2, CV_64FC1);
    for(int32_t y = 0; y < rows; y++) {
      for(int32_t x = 0; x < cols; x++) {
        predict_sample.at<double_t>(0, 0) = x;
        predict_sample.at<double_t>(0, 1) = y;
        cv::Vec2d likelihood = emmodel.predict2(predict_sample, predict_result);
        if(maxlikelihood < likelihood.val[0])
          maxlikelihood = likelihood.val[0];
        if(minlikelihood > likelihood.val[0])
          minlikelihood = likelihood.val[0];
        predict_image.at<double_t>(y, x) = likelihood.val[0];
      }
    }
  } else if(MODE_COLOR == mode) {
    predict_sample = cv::Mat(1, 3, CV_64FC1);
    for(int32_t y = 0; y < rows; y++) {
      for(int32_t x = 0; x < cols; x++) {
        cv::Vec3b color = reference_image.at<cv::Vec3b>(y, x);
        predict_sample.at<double_t>(0, 0) = static_cast<double_t>(color.val[0]);
        predict_sample.at<double_t>(0, 1) = static_cast<double_t>(color.val[1]);
        predict_sample.at<double_t>(0, 2) = static_cast<double_t>(color.val[2]);
        cv::Vec2d likelihood = emmodel.predict2(predict_sample, predict_result);
        if(maxlikelihood < likelihood.val[0])
          maxlikelihood = likelihood.val[0];
        if(minlikelihood > likelihood.val[0])
          minlikelihood = likelihood.val[0];
        predict_image.at<double_t>(y, x) = likelihood.val[0];
      }
    }
  }
  for(int32_t y = 0; y < rows; y++) {
    for(int32_t x = 0; x < cols; x++) {
      double_t value = (predict_image.at<double_t>(y, x) - minlikelihood )
                       / (maxlikelihood - minlikelihood ) * 255;
      if(value < 0)
        value = 0;
      likelihood_image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::floor(value));
    }
  }
  return true;
}

bool generateProbabilityImage(const cvgenprobmap_base::Em &emmodel,
                              const cv::Mat &reference_image,
                              CalcEMMode mode, cv::Mat &probability_image) {
  if(reference_image.empty()) {
    LOG(ERROR) << "reference_image must not be empty!!" << std::endl;
    return false;
  }
  cv::Mat check = emmodel.getMeans();
  if(MODE_XY == mode) {
    if(2 != check.cols) {
      LOG(ERROR) << "p_emmodel is traind by " << check.cols << " dimentions vectors.\n"
                 << "This model do not match to MODE_XY" << std::endl;
      return false;
    }
  } else if(MODE_COLOR == mode) {
    if(3 != check.cols) {
      LOG(ERROR) << "p_emmodel is traind by " << check.cols << " dimentions vectors.\n"
                 << "This model do not match to MODE_COLOR" << std::endl;
      return false;
    }
  }

  int32_t rows = reference_image.rows;
  int32_t cols = reference_image.cols;
  cv::Mat predict_sample;
  cv::Mat predict_result;
  cv::Mat predict_image = cv::Mat::zeros(reference_image.rows,
                                         reference_image.cols, CV_64FC1);
  probability_image = cv::Mat::zeros(reference_image.rows,
                                     reference_image.cols, CV_8UC1);
  double_t maxprobability = -DBL_MAX;
  double_t minprobability = DBL_MAX;
  cv::Mat weights = emmodel.getWeights();
  if(MODE_XY == mode) {
    predict_sample = cv::Mat(1, 2, CV_64FC1);
    for(int32_t y = 0; y < rows; y++) {
      for(int32_t x = 0; x < cols; x++) {
        predict_sample.at<double_t>(0, 0) = x;
        predict_sample.at<double_t>(0, 1) = y;
        cv::Vec2d likelihood = emmodel.predict2(predict_sample, predict_result);
        double_t sumofprobability = 0;
        /*
        for(int32_t index_cluster = 0; index_cluster < weights.cols; index_cluster++) {
          double_t weight = weights.at<double_t>(0, index_cluster);
          if(weight < 0)
            weight = 0;
          double_t prob = predict_result.at<double_t>(0, index_cluster);
          if(prob < 0)
            prob = 0;
          double_t probability = weight * prob;
          sumofprobability += probability;
        }
        */
        double_t weight = weights.at<double_t>(0, likelihood.val[1]);
        double_t prob = predict_result.at<double_t>(0, likelihood.val[1]);
        sumofprobability = weight * prob;
        if(maxprobability < sumofprobability)
          maxprobability = sumofprobability;
        if(minprobability > sumofprobability)
          minprobability = sumofprobability;
        predict_image.at<double_t>(y, x) = sumofprobability;
      }
    }
  } else if(MODE_COLOR == mode) {
    predict_sample = cv::Mat(1, 3, CV_64FC1);
    for(int32_t y = 0; y < rows; y++) {
      for(int32_t x = 0; x < cols; x++) {
        cv::Vec3b color = reference_image.at<cv::Vec3b>(y, x);
        predict_sample.at<double_t>(0, 0) = static_cast<double_t>(color.val[0]);
        predict_sample.at<double_t>(0, 1) = static_cast<double_t>(color.val[1]);
        predict_sample.at<double_t>(0, 2) = static_cast<double_t>(color.val[2]);
        cv::Vec2d likelihood = emmodel.predict2(predict_sample, predict_result);
        double_t sumofprobability = 0;
        /*
        for(int32_t index_cluster = 0; index_cluster < weights.cols; index_cluster++) {
          double_t weight = weights.at<double_t>(0, index_cluster);
          if(weight < 0)
            weight = 0;
          double_t prob = predict_result.at<double_t>(0, index_cluster);
          if(prob < 0)
            prob = 0;
          double_t probability = weight * prob;
          sumofprobability += probability;
        }
        */
        double_t weight = weights.at<double_t>(0, likelihood.val[1]);
        double_t prob = predict_result.at<double_t>(0, likelihood.val[1]);
        sumofprobability = weight * prob;
        if(maxprobability < sumofprobability)
          maxprobability = sumofprobability;
        if(minprobability > sumofprobability)
          minprobability = sumofprobability;
        predict_image.at<double_t>(y, x) = sumofprobability;
      }
    }
  }
  for(int32_t y = 0; y < rows; y++) {
    for(int32_t x = 0; x < cols; x++) {
      double_t value = (predict_image.at<double_t>(y, x) - minprobability )
                       / (maxprobability - minprobability ) * 255;
      if(value < 0)
        value = 0;
      probability_image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::floor(value));
    }
  }
  return true;
}


/*=== Global Function Implementation ========================================*/

int main(int argc, char *argv[]) {
  /* Initialize */
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  char stringbuff[PATH_MAX];
  getcwd(stringbuff, sizeof(stringbuff));
  chdir("./resource");
  getcwd(stringbuff, sizeof(stringbuff));

  if(3 > argc) {
    LOG(ERROR) << "Usage : CvGenProbMap [image] [mode : 0, 1]" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread(argv[1]);
  if(image.empty()) {
    LOG(ERROR) << "Can not open " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  int32_t mode = std::atoi(argv[2]);


  std::ifstream fore_input;
  fore_input.open("foresampling.cvs");
  if(!fore_input.is_open()) {
    LOG(ERROR) << "Can not open foresample.cvs" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream back_input;
  back_input.open("backsampling.cvs");
  if(!back_input.is_open()) {
    LOG(ERROR) << "Can not open backsample.cvs" << std::endl;
    return EXIT_FAILURE;
  }

  /* CVS読み込み */
  std::string line;
  while(std::getline(fore_input, line)) {
    std::string value_stirng;
    std::istringstream linestringstream(line);
    SampleData data;
    for(int32_t i = 0; i < 5; i++) {
      std::getline(linestringstream, value_stirng, ',');
      data.val[i] = std::atof(value_stirng.c_str());
    }
    fore_data.push_back(data);
  }
  while(std::getline(back_input, line)) {
    std::string value_stirng;
    std::istringstream linestringstream(line);
    SampleData data;
    for(int32_t i = 0; i < 5; i++) {
      std::getline(linestringstream, value_stirng, ',');
      data.val[i] = std::atof(value_stirng.c_str());
    }
    back_data.push_back(data);
  }

  chdir("../result");
  getcwd(stringbuff, sizeof(stringbuff));

  FLAGS_logtostderr = false;
  FLAGS_log_dir = stringbuff;

  /* 座標データの混合ガウス分布を算出して尤度マップを生成する */
  cv::Mat fore_xylabels;
  cv::Mat fore_xylog;
  cv::Mat fore_xyprobs;
  cvgenprobmap_base::Em::Params fore_xyemparams = cvgenprobmap_base::Em::Params(5);
  cvgenprobmap_base::Em fore_xyemmodel = cvgenprobmap_base::Em(fore_xyemparams);
  generateEMModel(fore_data, MODE_XY, fore_xyemparams,
                  fore_xyemmodel, fore_xylog, fore_xylabels, fore_xyprobs);

  cv::Mat fore_xy_likelihoodimage;
  generateLikelihoodImage(fore_xyemmodel, image, MODE_XY,
                          fore_xy_likelihoodimage);

  cv::Mat fore_xy_probabilityimage;
  generateProbabilityImage(fore_xyemmodel, image, MODE_XY,
                           fore_xy_probabilityimage);

  cv::Mat back_xylabels;
  cv::Mat back_xylog;
  cv::Mat back_xyprobs;
  cvgenprobmap_base::Em::Params back_xyemparams = cvgenprobmap_base::Em::Params(5);
  cvgenprobmap_base::Em back_xyemmodel = cvgenprobmap_base::Em(back_xyemparams);
  generateEMModel(back_data, MODE_XY, back_xyemparams,
                  back_xyemmodel, back_xylog, back_xylabels, back_xyprobs);

  cv::Mat back_xy_likelihoodimage;
  generateLikelihoodImage(back_xyemmodel, image, MODE_XY,
                          back_xy_likelihoodimage);

  cv::Mat back_xy_probabilityimage;
  generateProbabilityImage(back_xyemmodel, image, MODE_XY,
                           back_xy_probabilityimage);

  /* 色値の混合ガウス分布を算出して尤度マップを生成する */
  cv::Mat fore_colorlabels;
  cv::Mat fore_colorlog;
  cv::Mat fore_colorprobs;
  cvgenprobmap_base::Em::Params fore_coloremparams = cvgenprobmap_base::Em::Params(5);
  cvgenprobmap_base::Em fore_coloremmodel = cvgenprobmap_base::Em(fore_coloremparams);
  generateEMModel(fore_data, MODE_COLOR, fore_coloremparams,
                  fore_coloremmodel, fore_colorlog, fore_colorlabels,
                  fore_colorprobs);

  cv::Mat fore_color_likelihoodimage;
  generateLikelihoodImage(fore_coloremmodel, image, MODE_COLOR,
                          fore_color_likelihoodimage);

  cv::Mat fore_color_probabilityimage;
  generateProbabilityImage(fore_coloremmodel, image, MODE_COLOR,
                           fore_color_probabilityimage);

  cv::Mat back_colorlabels;
  cv::Mat back_colorlog;
  cv::Mat back_colorprobs;
  cvgenprobmap_base::Em::Params back_coloremparams = cvgenprobmap_base::Em::Params(5);
  cvgenprobmap_base::Em back_coloremmodel = cvgenprobmap_base::Em(back_coloremparams);
  generateEMModel(back_data, MODE_COLOR, back_coloremparams,
                  back_coloremmodel, back_colorlog, back_colorlabels,
                  back_colorprobs);

  cv::Mat back_color_likelihoodimage;
  generateLikelihoodImage(back_coloremmodel, image, MODE_COLOR,
                          back_color_likelihoodimage);

  cv::Mat back_color_probabilityimage;
  generateProbabilityImage(back_coloremmodel, image, MODE_COLOR,
                           back_color_probabilityimage);

  cv::imshow("result", fore_xy_likelihoodimage);
  cv::waitKey(0);

  cv::imshow("result", back_xy_likelihoodimage);
  cv::waitKey(0);

  cv::imshow("result", fore_color_likelihoodimage);
  cv::waitKey(0);

  cv::imshow("result", back_color_likelihoodimage);
  cv::waitKey(0);

  cv::imshow("result", fore_xy_probabilityimage);
  cv::waitKey(0);

  cv::imshow("result", back_xy_probabilityimage);
  cv::waitKey(0);

  cv::imshow("result", fore_color_probabilityimage);
  cv::waitKey(0);

  cv::imshow("result", back_color_probabilityimage);
  cv::waitKey(0);

  /* Finalize */
  google::InstallFailureSignalHandler();

  return EXIT_SUCCESS;
}
