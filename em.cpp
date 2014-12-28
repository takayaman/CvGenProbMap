/*=============================================================================
 * Project : CvGenProbMap
 * Code : em.cpp
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

/*=== Include ===============================================================*/

#include "em.h"


namespace cvgenprobmap_base {

const double_t kMinEigenValue = DBL_EPSILON;
#define CV_LOG2PI (1.8378770664093454835606594728112)

static inline void writeTermCrit(cv::FileStorage& fs, const cv::TermCriteria& termCrit) {
  if( (termCrit.type & cv::TermCriteria::EPS) != 0 )
    fs << "epsilon" << termCrit.epsilon;
  if( (termCrit.type & cv::TermCriteria::COUNT) != 0 )
    fs << "ifterations" << termCrit.maxCount;
}

static inline cv::TermCriteria readTermCrit(const cv::FileNode& fn) {
  cv::TermCriteria termCrit;
  double epsilon = (double)fn["epsilon"];
  if( epsilon > 0 ) {
    termCrit.type |= cv::TermCriteria::EPS;
    termCrit.epsilon = epsilon;
  }
  int iters = (int)fn["iterations"];
  if( iters > 0 ) {
    termCrit.type |= cv::TermCriteria::COUNT;
    termCrit.maxCount = iters;
  }
  return termCrit;
}

/*=== Class Implementation ==================================================*/

/*--- Constructor / Destructor / Initialize ---------------------------------*/


Em::Params::Params(int nclusters, int covmattype,
                   const cv::TermCriteria &termcrit)
  : m_nclusters(nclusters),
    m_covmattype(covmattype),
    m_termcrit(termcrit) {
}

Em::Em(const Params &params) {
  setParams(params);
}

Em::~Em(void) {

}

/*--- Operation -------------------------------------------------------------*/

/*--- Public ----------------------------------------------------------------*/
bool Em::train(const cv::Ptr<cv::ml::TrainData> &traindata, int flags) {
  cv::Mat samples = traindata->getTrainSamples();
  cv::Mat loglikelihood, labels, probs;
  return train(samples, loglikelihood, labels, probs, m_params);
}

bool Em::train(const cv::Mat &samples, cv::Mat &loglikelihoods,
               cv::Mat &labels, cv::Mat &probs, const Params &params) {
  setParams(params);
  setTrainData(START_AUTO_STEP, samples, NULL, NULL, NULL, NULL);
  return doTrain(START_AUTO_STEP, loglikelihoods, labels, probs);
}

bool Em::train_startWithE(const cv::Mat &samples, const cv::Mat &means0,
                          const std::vector<cv::Mat> &covs0,
                          const cv::Mat &weights0,
                          cv::Mat &loglikelihoods, cv::Mat &labels,
                          cv::Mat &probs, const Params &params) {
  setParams(params);
  return trainE(samples, means0, covs0, weights0, loglikelihoods, labels,
                probs);
}

bool Em::train_startWithM(const cv::Mat &samples, const cv::Mat &probs0,
                          cv::Mat &loglikelihoods, cv::Mat &labels,
                          cv::Mat &probs, const Params &params) {
  setParams(params);
  return trainM(samples, probs0, loglikelihoods, labels, probs);
}

bool Em::doTrain(int32_t startstep, cv::Mat &loglikelihoods,
                 cv::Mat &labels, cv::Mat &probs) {
  int nclusters = m_params.m_nclusters;
  int dim = m_trainsamples.cols;
  // Precompute the empty initial train data in the cases of
  // START_E_STEP and START_AUTO_STEP
  if(startstep != START_M_STEP) {
    if(m_covariants.empty()) {
      CV_Assert(m_weights.empty());
      clusterTrainSamples();
    }
  }

  if(!m_covariants.empty() && m_covs_eigenvalues.empty() ) {
    CV_Assert(m_inverse_covs_eigenvalues.empty());
    decomposeCovs();
  }

  if(startstep == START_M_STEP)
    mStep();

  double trainloglikelihood, prev_trainloglikelihood = 0.;
  int maxiters = (m_params.m_termcrit.type & cv::TermCriteria::MAX_ITER) ?
                 m_params.m_termcrit.maxCount : DEFAULT_MAX_ITERS;
  double epsilon = (m_params.m_termcrit.type & cv::TermCriteria::EPS) ?
                   m_params.m_termcrit.epsilon : 0.;

  for(int iter = 0; ; iter++) {
    eStep();
    trainloglikelihood = sum(m_trainloglikelihoods)[0];

    if(iter >= maxiters - 1)
      break;

    double trainloglikelihooddelta = trainloglikelihood
                                     - prev_trainloglikelihood;
    if( iter != 0 &&
        (trainloglikelihooddelta < -DBL_EPSILON ||
         trainloglikelihooddelta < epsilon * std::fabs(trainloglikelihood)))
      break;

    mStep();

    prev_trainloglikelihood = trainloglikelihood;
  }

  if( trainloglikelihood <= -DBL_MAX/10000. ) {
    clear();
    return false;
  }

  // postprocess covs
  m_covariants.resize(nclusters);
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    if(m_params.m_covmattype == COV_MAT_SPHERICAL) {
      m_covariants[index_cluster].create(dim, dim, CV_64FC1);
      setIdentity(m_covariants[index_cluster],
                  cv::Scalar(m_covs_eigenvalues[index_cluster].at<double>(0)));
    } else if(m_params.m_covmattype == COV_MAT_DIAGONAL) {
      m_covariants[index_cluster] =
        cv::Mat::diag(m_covs_eigenvalues[index_cluster]);
    }
  }

  m_trainlabels.copyTo(labels);
  m_trainprobs.copyTo(probs);
  m_trainloglikelihoods.copyTo(loglikelihoods);

  m_trainsamples.release();
  m_trainprobs.release();
  m_trainlabels.release();
  m_trainloglikelihoods.release();

  return true;
}

float_t Em::predict(const cv::Mat &inputs, cv::Mat &outputs, int) const {

  cv::Mat samples = inputs, probs, probsrow;
  int ptype = CV_32F;
  float firstres = 0.f;
  int i, nsamples = samples.rows;

  outputs = cv::Mat::zeros(samples.rows, m_params.m_nclusters, ptype);

  for( i = 0; i < nsamples; i++ ) {
    probsrow = probs.row(i);
    cv::Vec2d res = computeProbabilities(samples.row(i), &probsrow, ptype);
    if( i == 0 )
      firstres = (float)res[1];
  }
  return firstres;
}

cv::Vec2d Em::predict2(const cv::Mat &sample, cv::Mat &probs) const {
  int ptype = CV_32F;
  CV_Assert(isTrained());
  CV_Assert(!sample.empty());
  cv::Mat temp;
  if(sample.type() != CV_64FC1) {
    sample.convertTo(temp, CV_64FC1);
  } else
    temp = sample;
  temp.reshape(1, 1);

  if(!probs.empty())
    ptype = probs.type();
  probs = cv::Mat::zeros(1, m_params.m_nclusters, ptype);
  return computeProbabilities(sample, &probs, ptype);
}



/*--- Private ---------------------------------------------------------------*/
void Em::clear(void) {
  m_trainsamples.release();
  m_trainprobs.release();
  m_trainloglikelihoods.release();
  m_trainlabels.release();

  m_weights.release();
  m_means.release();
  m_covariants.clear();

  m_covs_eigenvalues.clear();
  m_inverse_covs_eigenvalues.clear();
  m_covs_rotatemats.clear();

  m_logweights_divdeterminant.release();
}

bool Em::trainE(const cv::Mat &samples, const cv::Mat &means0,
                const std::vector<cv::Mat> &covs0, const cv::Mat &weights0,
                cv::Mat &loglikelihoods, cv::Mat &labels, cv::Mat &probs) {

  setTrainData(START_E_STEP, samples, &probs, &means0, &covs0, &weights0);
  return doTrain(START_E_STEP, loglikelihoods, labels, probs);
}

bool Em::trainM(const cv::Mat &samples, const cv::Mat &probs0,
                cv::Mat &loglikelihoods, cv::Mat &labels, cv::Mat &probs) {
  setTrainData(START_M_STEP, samples, &probs0, 0, 0, 0);;
  return doTrain(START_M_STEP, loglikelihoods, labels, probs);
}

void Em::eStep(void) {
  // Compute probs_ik from means_k, covs_k and weights_k.
  m_trainprobs.create(m_trainsamples.rows, m_params.m_nclusters, CV_64FC1);
  m_trainlabels.create(m_trainsamples.rows, 1, CV_32SC1);
  m_trainloglikelihoods.create(m_trainsamples.rows, 1, CV_64FC1);

  computeLogWeightDivDet();

  CV_DbgAssert(m_trainsamples.type() == CV_64FC1);
  CV_DbgAssert(m_means.type() == CV_64FC1);

  for(int sampleindex = 0; sampleindex < m_trainsamples.rows; sampleindex++) {
    cv::Mat sampleprobs = m_trainprobs.row(sampleindex);
    cv::Vec2d res = computeProbabilities(m_trainsamples.row(sampleindex),
                                         &sampleprobs, CV_64F);
    m_trainloglikelihoods.at<double>(sampleindex) = res[0];
    m_trainlabels.at<int>(sampleindex) = static_cast<int>(res[1]);
  }
}

void Em::mStep(void) {
  // Update means_k, covs_k and weights_k from probs_ik
  int nclusters = m_params.m_nclusters;
  int covmattype = m_params.m_covmattype;
  int dim = m_trainsamples.cols;

  // Update weights
  // not normalized first
  reduce(m_trainprobs, m_weights, 0, cv::REDUCE_SUM);

  // Update means
  m_means.create(nclusters, dim, CV_64FC1);
  m_means = cv::Scalar(0);

  const double minposweight = m_trainsamples.rows * DBL_EPSILON;
  double minweight = DBL_MAX;
  int minweightclusterindex = -1;
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    if(m_weights.at<double>(index_cluster) <= minposweight)
      continue;

    if(m_weights.at<double>(index_cluster) < minweight) {
      minweight = m_weights.at<double>(index_cluster);
      minweightclusterindex = index_cluster;
    }

    cv::Mat clustermean = m_means.row(index_cluster);
    for(int sampleindex = 0; sampleindex < m_trainsamples.rows; sampleindex++)
      clustermean += m_trainprobs.at<double>(sampleindex, index_cluster) * m_trainsamples.row(sampleindex);
    clustermean /= m_weights.at<double>(index_cluster);
  }

  // Update covsEigenValues and invCovsEigenValues
  m_covariants.resize(nclusters);
  m_covs_eigenvalues.resize(nclusters);
  if(covmattype == COV_MAT_GENERIC)
    m_covs_rotatemats.resize(nclusters);
  m_inverse_covs_eigenvalues.resize(nclusters);
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    if(m_weights.at<double>(index_cluster) <= minposweight)
      continue;

    if(covmattype != COV_MAT_SPHERICAL)
      m_covs_eigenvalues[index_cluster].create(1, dim, CV_64FC1);
    else
      m_covs_eigenvalues[index_cluster].create(1, 1, CV_64FC1);

    if(covmattype == COV_MAT_GENERIC)
      m_covariants[index_cluster].create(dim, dim, CV_64FC1);

    cv::Mat clustercov = covmattype != COV_MAT_GENERIC ?
                         m_covs_eigenvalues[index_cluster] : m_covariants[index_cluster];

    clustercov = cv::Scalar(0);

    cv::Mat centeredsample;
    for(int index_sample = 0; index_sample < m_trainsamples.rows; index_sample++) {
      centeredsample = m_trainsamples.row(index_sample) -
                       m_means.row(index_cluster);

      if(covmattype == COV_MAT_GENERIC)
        clustercov += m_trainprobs.at<double>(index_sample, index_cluster) *
                      centeredsample.t() * centeredsample;
      else {
        double p = m_trainprobs.at<double>(index_sample, index_cluster);
        for(int di = 0; di < dim; di++ ) {
          double val = centeredsample.at<double>(di);
          clustercov.at<double>(covmattype != COV_MAT_SPHERICAL ? di : 0) +=
            p*val*val;
        }
      }
    }

    if(covmattype == COV_MAT_SPHERICAL)
      clustercov /= dim;

    clustercov /= m_weights.at<double>(index_cluster);

    // Update covsRotateMats for COV_MAT_GENERIC only
    if(covmattype == COV_MAT_GENERIC) {
      cv::SVD svd(m_covariants[index_cluster],
                  cv::SVD::MODIFY_A + cv::SVD::FULL_UV);
      m_covs_eigenvalues[index_cluster] = svd.w;
      m_covs_rotatemats[index_cluster] = svd.u;
    }

    max(m_covs_eigenvalues[index_cluster], kMinEigenValue,
        m_covs_eigenvalues[index_cluster]);

    // update invCovsEigenValues
    m_inverse_covs_eigenvalues[index_cluster] =
      1. / m_covs_eigenvalues[index_cluster];
  }

  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    if(m_weights.at<double>(index_cluster) <= minposweight) {
      cv::Mat clustermean = m_means.row(index_cluster);
      m_means.row(minweightclusterindex).copyTo(clustermean);
      m_covariants[minweightclusterindex].copyTo(m_covariants[index_cluster]);
      m_covs_eigenvalues[minweightclusterindex].copyTo(
        m_covs_eigenvalues[index_cluster]);
      if(covmattype == COV_MAT_GENERIC)
        m_covs_rotatemats[minweightclusterindex].copyTo(
          m_covs_rotatemats[index_cluster]);
      m_inverse_covs_eigenvalues[minweightclusterindex].copyTo(
        m_inverse_covs_eigenvalues[index_cluster]);
    }
  }

  // Normalize weights
  m_weights /= m_trainsamples.rows;
}

void Em::preprocessSampleData(const cv::Mat &src, cv::Mat &dst,
                              int32_t dsttype, bool is_alwaysclone) {
  if(src.type() == dsttype && !is_alwaysclone)
    dst = src;
  else
    src.convertTo(dst, dsttype);
}

void Em::preprocessProbability(cv::Mat &probs) {
  max(probs, 0., probs);

  const double uniform_probability = (double)(1./probs.cols);
  for(int y = 0; y < probs.rows; y++) {
    cv::Mat sampleprobs = probs.row(y);

    double maxvalue = 0;
    cv::minMaxLoc(sampleprobs, 0, &maxvalue);
    if(maxvalue < FLT_EPSILON)
      sampleprobs.setTo(uniform_probability);
    else
      cv::normalize(sampleprobs, sampleprobs, 1, 0, cv::NORM_L1);
  }
}

void Em::decomposeCovs(void) {
  int nclusters = m_params.m_nclusters, covmattype = m_params.m_covmattype;
  CV_Assert(!m_covariants.empty());
  m_covs_eigenvalues.resize(nclusters);
  if(covmattype == COV_MAT_GENERIC)
    m_covs_rotatemats.resize(nclusters);
  m_inverse_covs_eigenvalues.resize(nclusters);
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    CV_Assert(!m_covariants[index_cluster].empty());

    cv::SVD svd(m_covariants[index_cluster],
                cv::SVD::MODIFY_A + cv::SVD::FULL_UV);

    if(covmattype == COV_MAT_SPHERICAL) {
      double maxsingularval = svd.w.at<double>(0);
      m_covs_eigenvalues[index_cluster] = cv::Mat(1, 1, CV_64FC1,
                                          cv::Scalar(maxsingularval));
    } else if(covmattype == COV_MAT_DIAGONAL) {
      m_covs_eigenvalues[index_cluster] = svd.w;
    } else { //COV_MAT_GENERIC
      m_covs_eigenvalues[index_cluster] = svd.w;
      m_covs_rotatemats[index_cluster] = svd.u;
    }
    max(m_covs_eigenvalues[index_cluster], kMinEigenValue,
        m_covs_eigenvalues[index_cluster]);
    m_inverse_covs_eigenvalues[index_cluster] =
      1. / m_covs_eigenvalues[index_cluster];
  }
}

void Em::clusterTrainSamples(void) {
  int nclusters = m_params.m_nclusters;
  int nsamples = m_trainsamples.rows;

  // Cluster samples, compute/update means

  // Convert samples and means to 32F, because kmeans requires this type.
  cv::Mat trainsamples_flt, means_flt;
  if(m_trainsamples.type() != CV_32FC1)
    m_trainsamples.convertTo(trainsamples_flt, CV_32FC1);
  else
    trainsamples_flt = m_trainsamples;
  if(!m_means.empty()) {
    if(m_means.type() != CV_32FC1)
      m_means.convertTo(means_flt, CV_32FC1);
    else
      means_flt = m_means;
  }

  cv::Mat labels;
  cv::kmeans(trainsamples_flt, nclusters, labels,
             cv::TermCriteria(cv::TermCriteria::COUNT, m_means.empty() ? 10 : 1, 0.5),
             10, cv::KMEANS_PP_CENTERS, means_flt);

  // Convert samples and means back to 64F.
  CV_Assert(means_flt.type() == CV_32FC1);
  if(m_trainsamples.type() != CV_64FC1) {
    cv::Mat trainsamples_buffer;
    trainsamples_flt.convertTo(trainsamples_buffer, CV_64FC1);
    m_trainsamples = trainsamples_buffer;
  }
  means_flt.convertTo(m_means, CV_64FC1);

  // Compute weights and covs
  m_weights = cv::Mat(1, nclusters, CV_64FC1, cv::Scalar(0));
  m_covariants.resize(nclusters);
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    cv::Mat clustersamples;
    for(int index_sample = 0; index_sample < nsamples; index_sample++) {
      if(labels.at<int>(index_sample) == index_cluster) {
        const cv::Mat sample = m_trainsamples.row(index_sample);
        clustersamples.push_back(sample);
      }
    }
    CV_Assert(!clustersamples.empty());

    cv::calcCovarMatrix(clustersamples, m_covariants[index_cluster],
                        m_means.row(index_cluster),
                        cv::COVAR_NORMAL + cv::COVAR_ROWS +
                        cv::COVAR_USE_AVG + cv::COVAR_SCALE,
                        CV_64FC1);
    m_weights.at<double>(index_cluster) =
      static_cast<double>(clustersamples.rows) /
      static_cast<double>(nsamples);
  }

  decomposeCovs();
}

void Em::computeLogWeightDivDet(void) {
  int nclusters = m_params.m_nclusters;
  CV_Assert(!m_covs_eigenvalues.empty());

  cv::Mat logweights;
  cv::max(m_weights, DBL_MIN, m_weights);
  log(m_weights, logweights);

  m_logweights_divdeterminant.create(1, nclusters, CV_64FC1);
  // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    double logdetcov = 0.;
    const int evalcount =
      static_cast<int>(m_covs_eigenvalues[index_cluster].total());
    for(int di = 0; di < evalcount; di++)
      logdetcov += std::log(m_covs_eigenvalues[index_cluster].at<double>(m_params.m_covmattype != COV_MAT_SPHERICAL ? di : 0));

    m_logweights_divdeterminant.at<double>(index_cluster) =
      logweights.at<double>(index_cluster) - 0.5 * logdetcov;
  }
}

cv::Vec2d Em::computeProbabilities(const cv::Mat &sample, cv::Mat *probs,
                                   int32_t ptype) const {
  // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
  // q = arg(max_k(L_ik))
  // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
  // see Alex Smola's blog http://blog.smola.org/page/2 for
  // details on the log-sum-exp trick

  int nclusters = m_params.m_nclusters, covmattype = m_params.m_covmattype;
  int stype = sample.type();
  CV_Assert(!m_means.empty());
  CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
  CV_Assert(sample.size() == cv::Size(m_means.cols, 1));

  int dim = sample.cols;

  cv::Mat labelsmat(1, nclusters, CV_64FC1), centeredsample(1, dim, CV_64F);
  int i, label = 0;
  for(int index_cluster = 0; index_cluster < nclusters; index_cluster++) {
    const double* mptr = m_means.ptr<double>(index_cluster);
    double* dptr = centeredsample.ptr<double>();
    if( stype == CV_32F ) {
      const float* sptr = sample.ptr<float>();
      for( i = 0; i < dim; i++ )
        dptr[i] = sptr[i] - mptr[i];
    } else {
      const double* sptr = sample.ptr<double>();
      for( i = 0; i < dim; i++ )
        dptr[i] = sptr[i] - mptr[i];
    }

    cv::Mat rotatedcenteredsample = covmattype != COV_MAT_GENERIC ?
                                    centeredsample : centeredsample * m_covs_rotatemats[index_cluster];

    double labelvalue = 0;
    for(int di = 0; di < dim; di++) {
      double w = m_inverse_covs_eigenvalues[index_cluster].at<double>(covmattype != COV_MAT_SPHERICAL ? di : 0);
      double val = rotatedcenteredsample.at<double>(di);
      labelvalue += w * val * val;
    }
    CV_DbgAssert(!m_logweights_divdeterminant.empty());
    labelsmat.at<double>(index_cluster) = m_logweights_divdeterminant.at<double>(index_cluster) - 0.5 * labelvalue;

    if(labelsmat.at<double>(index_cluster) > labelsmat.at<double>(label))
      label = index_cluster;
  }

  double maxlavelvalue = labelsmat.at<double>(label);
  double exp_diff_sum = 0;
  for( i = 0; i < labelsmat.cols; i++ ) {
    double v = std::exp(labelsmat.at<double>(i) - maxlavelvalue);
    labelsmat.at<double>(i) = v;
    exp_diff_sum += v; // sum_j(exp(L_ij - L_iq))
  }

  if(probs)
    labelsmat.convertTo(*probs, ptype, 1./exp_diff_sum);

  cv::Vec2d res;
  res[0] = std::log(exp_diff_sum)  + maxlavelvalue - 0.5 * dim * CV_LOG2PI;
  res[1] = label;

  return res;
}

void Em::writeParams(cv::FileStorage &fs) const {
  fs << "nclusters" << m_params.m_nclusters;
  fs << "cov_mat_type" << (m_params.m_covmattype == COV_MAT_SPHERICAL ? cv::String("spherical") :
                           m_params.m_covmattype == COV_MAT_DIAGONAL ? cv::String("diagonal") :
                           m_params.m_covmattype == COV_MAT_GENERIC ? cv::String("generic") :
                           cv::format("unknown_%d", m_params.m_covmattype));
  writeTermCrit(fs, m_params.m_termcrit);
}

void Em::write(cv::FileStorage &fs) const {
  fs << "training_params" << "{";
  writeParams(fs);
  fs << "}";
  fs << "weights" << m_weights;
  fs << "means" << m_means;

  size_t i, n = m_covariants.size();

  fs << "covs" << "[";
  for( i = 0; i < n; i++ )
    fs << m_covariants[i];
  fs << "]";
}

void Em::readParams(const cv::FileNode &fn) {
  Params params;
  params.m_nclusters = (int)fn["nclusters"];
  cv::String s = (cv::String)fn["cov_mat_type"];
  params.m_covmattype = s == "spherical" ? COV_MAT_SPHERICAL :
                        s == "diagonal" ? COV_MAT_DIAGONAL :
                        s == "generic" ? COV_MAT_GENERIC : -1;
  CV_Assert(params.m_covmattype >= 0);
  params.m_termcrit = readTermCrit(fn);
  setParams(params);
}

void Em::read(const cv::FileNode &fn) {
  clear();
  readParams(fn["training_params"]);

  fn["weights"] >> m_weights;
  fn["means"] >> m_means;

  cv::FileNode cfn = fn["covs"];
  cv::FileNodeIterator cfn_it = cfn.begin();
  int i, n = (int)cfn.size();
  m_covariants.resize(n);

  for( i = 0; i < n; i++, ++cfn_it )
    (*cfn_it) >> m_covariants[i];

  decomposeCovs();
  computeLogWeightDivDet();
}

/*--- Accessor --------------------------------------------------------------*/
/*--- Public ----------------------------------------------------------------*/

void Em::setParams(const Params &params) {
  m_params = params;
  CV_Assert(m_params.m_nclusters > 1);
  CV_Assert(m_params.m_covmattype == COV_MAT_SPHERICAL ||
            m_params.m_covmattype == COV_MAT_DIAGONAL ||
            m_params.m_covmattype == COV_MAT_GENERIC);
}

Em::Params Em::getParams(void) const {
  return m_params;
}

cv::Mat Em::getWeights(void) const {
  return m_weights;
}

cv::Mat Em::getMeans(void) const {
  return m_means;
}

void Em::getCovs(std::vector<cv::Mat> &covs) const {
  covs.resize(m_covariants.size());
  std::copy(m_covariants.begin(), m_covariants.end(), covs.begin());
}

/*--- Private ---------------------------------------------------------------*/
bool Em::isTrained(void) const {
  return !m_means.empty();
}

bool Em::isCassifier(void) const {
  return true;
}

int32_t Em::getVerCount(void) const {
  return m_means.cols;
}

void Em::checkTrainData(int32_t startstep, const cv::Mat &samples,
                        int32_t nclusters, int32_t covmattype,
                        const cv::Mat *probs, const cv::Mat *means,
                        const std::vector<cv::Mat> *covs,
                        const cv::Mat *weights) {
  // Check samples.
  CV_Assert(!samples.empty());
  CV_Assert(samples.channels() == 1);

  int nsamples = samples.rows;
  int dim = samples.cols;

  // Check training params.
  CV_Assert(nclusters > 0);
  CV_Assert(nclusters <= nsamples);
  CV_Assert(startstep == START_AUTO_STEP ||
            startstep == START_E_STEP ||
            startstep == START_M_STEP);
  CV_Assert(covmattype == COV_MAT_GENERIC ||
            covmattype == COV_MAT_DIAGONAL ||
            covmattype == COV_MAT_SPHERICAL);

  CV_Assert(!probs ||
            (!probs->empty() &&
             probs->rows == nsamples && probs->cols == nclusters &&
             (probs->type() == CV_32FC1 || probs->type() == CV_64FC1)));

  CV_Assert(!weights ||
            (!weights->empty() &&
             (weights->cols == 1 || weights->rows == 1) && static_cast<int>(weights->total()) == nclusters &&
             (weights->type() == CV_32FC1 || weights->type() == CV_64FC1)));

  CV_Assert(!means ||
            (!means->empty() &&
             means->rows == nclusters && means->cols == dim &&
             means->channels() == 1));

  CV_Assert(!covs ||
            (!covs->empty() &&
             static_cast<int>(covs->size()) == nclusters));
  if(covs) {
    const cv::Size covsize(dim, dim);
    for(size_t i = 0; i < covs->size(); i++) {
      const cv::Mat& m = (*covs)[i];
      CV_Assert(!m.empty() && m.size() == covsize && (m.channels() == 1));
    }
  }

  if(startstep == START_E_STEP) {
    CV_Assert(means);
  } else if(startstep == START_M_STEP) {
    CV_Assert(probs);
  }
}

void Em::setTrainData(int32_t startstep, const cv::Mat &samples,
                      const cv::Mat *probs0, const cv::Mat *means0,
                      const std::vector<cv::Mat> *covs0,
                      const cv::Mat *weights0) {
  int nclusters = m_params.m_nclusters, covmattype = m_params.m_covmattype;
  clear();

  checkTrainData(startstep, samples, nclusters, covmattype, probs0, means0,
                 covs0, weights0);

  bool isKMeansInit = (startstep == START_AUTO_STEP)
                      || (startstep == START_E_STEP && (covs0 == 0 || weights0 == 0));
  // Set checked data
  preprocessSampleData(samples, m_trainsamples,
                       isKMeansInit ? CV_32FC1 : CV_64FC1, false);

  // set probs
  if(probs0 && startstep == START_M_STEP) {
    preprocessSampleData(*probs0, m_trainprobs, CV_64FC1, true);
    preprocessProbability(m_trainprobs);
  }

  // set weights
  if(weights0 && (startstep == START_E_STEP && covs0)) {
    weights0->convertTo(m_weights, CV_64FC1);
    m_weights.reshape(1,1);
    preprocessProbability(m_weights);
  }

  // set means
  if(means0 && (startstep == START_E_STEP/* || startStep == START_AUTO_STEP*/))
    means0->convertTo(m_means, isKMeansInit ? CV_32FC1 : CV_64FC1);

  // set covs
  if(covs0 && (startstep == START_E_STEP && weights0)) {
    m_covariants.resize(nclusters);
    for(size_t i = 0; i < covs0->size(); i++)
      (*covs0)[i].convertTo(m_covariants[i], CV_64FC1);
  }
}

/*--- Event -----------------------------------------------------------------*/


} // namespace cvgenprobmap_base
