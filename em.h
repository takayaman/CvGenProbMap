/*=============================================================================
 * Project : CvGenProbMap
 * Code : em.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/27
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgenprobmap::Em
 * This class is modified version of EM class in opencv3.0a
 *===========================================================================*/

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef CVGENPROBMAP_EM_H
#define CVGENPROBMAP_EM_H

/*=== Include ===============================================================*/

#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgenprobmap_base {

class Em {
 public:
  // Type of covariation matrices
  typedef enum CovMatType_TAG {
    COV_MAT_SPHERICAL=0,
    COV_MAT_DIAGONAL=1,
    COV_MAT_GENERIC=2,
    COV_MAT_DEFAULT=COV_MAT_DIAGONAL
  } CovMatType;

  // Default parameters
  enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

  // The initial step
  typedef enum StartType_TAG {
    START_E_STEP=1,
    START_M_STEP=2,
    START_AUTO_STEP=0
  } StartType;

  class Params {
   public:

    explicit Params(int nclusters=DEFAULT_NCLUSTERS,
                    int covmattype=COV_MAT_DIAGONAL,
                    const cv::TermCriteria& termcrit=cv::TermCriteria(
                          cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                          DEFAULT_MAX_ITERS, 1e-6));
    int m_nclusters;
    int m_covmattype;
    cv::TermCriteria m_termcrit;
  };

  Em(const Params& params);

  ~Em(void);

  bool train( const cv::Ptr<cv::ml::TrainData>& traindata, int flags=0 );
  bool train(const cv::Mat &samples,
             cv::Mat &loglikelihoods,
             cv::Mat &labels,
             cv::Mat &probs,
             const Params& params=Params());
  bool train_startWithE(const cv::Mat &samples, const cv::Mat &means0,
                        const std::vector<cv::Mat> &covs0,
                        const cv::Mat &weights0,
                        cv::Mat &loglikelihoods,
                        cv::Mat &labels,
                        cv::Mat &probs,
                        const Params& params=Params());
  bool train_startWithM(const cv::Mat & samples, const cv::Mat & probs0,
                        cv::Mat & loglikelihoods,
                        cv::Mat & labels,
                        cv::Mat & probs,
                        const Params& params=Params());

  float_t predict(const cv::Mat &inputs, cv::Mat &outputs, int) const;
  cv::Vec2d predict2(const cv::Mat &sample, cv::Mat &probs) const;

  void setParams(const Params& params);
  Params getParams(void) const;
  cv::Mat getWeights(void) const;
  cv::Mat getMeans(void) const;
  void getCovs(std::vector<cv::Mat>& covs) const;

 private:
  void clear(void);
  bool trainE(const cv::Mat &samples, const cv::Mat &means0,
              const std::vector<cv::Mat> &covs0, const cv::Mat &weights0,
              cv::Mat &loglikelihoods, cv::Mat &labels, cv::Mat &probs);
  bool trainM(const cv::Mat &samples, const cv::Mat &probs0,
              cv::Mat &loglikelihoods, cv::Mat &labels, cv::Mat &probs);
  bool doTrain(int32_t startstep, cv::Mat &loglikelihoods, cv::Mat &labels,
               cv::Mat &probs);
  void eStep(void);
  void mStep(void);

  void preprocessSampleData(const cv::Mat &src, cv::Mat&dst, int32_t dsttype,
                            bool is_alwaysclone);
  void preprocessProbability(cv::Mat &probs);

  void decomposeCovs(void);

  void clusterTrainSamples(void);

  void computeLogWeightDivDet(void);

  cv::Vec2d computeProbabilities(const cv::Mat &sample, cv::Mat *probs,
                                 int32_t ptype) const;

  void writeParams(cv::FileStorage &fs) const;
  void write(cv::FileStorage &fs) const;
  void readParams(const cv::FileNode &fn);
  void read(const cv::FileNode &fn);

  bool isTrained(void) const;
  bool isCassifier(void) const;
  int32_t getVerCount(void) const;

  void checkTrainData(int32_t startstep, const cv::Mat &samples,
                      int32_t nclusters, int32_t covmattype,
                      const cv::Mat *probs, const cv::Mat *means,
                      const std::vector<cv::Mat> *covs,
                      const cv::Mat *weights);
  void setTrainData(int32_t startstep, const cv::Mat &samples,
                    const cv::Mat *probs0, const cv::Mat *means0,
                    const std::vector<cv::Mat> *covs0,
                    const cv::Mat *weights0);

 private:
  Params m_params;

  cv::Mat m_trainsamples;
  cv::Mat m_trainprobs;
  cv::Mat m_trainloglikelihoods;
  cv::Mat m_trainlabels;

  cv::Mat m_weights;
  cv::Mat m_means;
  std::vector<cv::Mat> m_covariants;

  std::vector<cv::Mat> m_covs_eigenvalues;
  std::vector<cv::Mat> m_covs_rotatemats;
  std::vector<cv::Mat> m_inverse_covs_eigenvalues;
  cv::Mat m_logweights_divdeterminant;

};

}  // namesapce cvgenprobmap_base

#endif  // CVGENPROBMAP_EM_H
