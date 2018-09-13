#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

//层类 构造函数==
template <typename Dtype>
FcRistrettoLayer<Dtype>::FcRistrettoLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {//继承于BaseRistrettoLayer和InnerProductLayer
  this->precision_ = this->layer_param_.quantization_param().precision();//量化方法
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();//取整策略
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT://动态定点量化
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();//激活输入量化总位宽
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();//激活输出量化总位宽
    this->bw_params_ = this->layer_param_.quantization_param().bw_params();//卷积参数量化总位宽
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();//激活输入小数部分量化总位宽
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();//激活输出小数部分量化位宽
    this->fl_params_ = this->layer_param_.quantization_param().fl_params();//卷积参数小数部分量化总位宽
    break;
  case QuantizationParameter_Precision_MINIFLOAT://迷你小浮点数量化
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS://2乘方量化
    this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
    this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

//层类初始化函数
template <typename Dtype>
void FcRistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//bottom为上层激活输入作为本层输入  输入通道数*输入h*输入w
  //输出个数
  const int num_output = this->layer_param_.inner_product_param().num_output();
  // 偏置相
  this->bias_term_ = this->layer_param_.inner_product_param().bias_term();
  // 转置
  this->transpose_ = this->layer_param_.inner_product_param().transpose();
  this->N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  this->K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights 权值初始化
    vector<int> weight_shape(2);
	// 是否转置
    if (this->transpose_) {
      weight_shape[0] = this->K_;
      weight_shape[1] = this->N_;
    } else {
      weight_shape[0] = this->N_;
      weight_shape[1] = this->K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
	// 填充权值
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
	// 是否初始化以及填充偏置
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // 参数初始化
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Prepare quantized weights
  // 准备量化权值
  this->weights_quantized_.resize(2);
  vector<int> weight_shape(2);
  weight_shape[0] = this->N_;
  weight_shape[1] = this->K_;
  this->weights_quantized_[0].reset(new Blob<Dtype>(weight_shape));
  vector<int> bias_shape(1, this->N_);
  if (this->bias_term_) {
      this->weights_quantized_[1].reset(new Blob<Dtype>(bias_shape));
  }
}

template <typename Dtype>
void FcRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Trim layer input 修剪(量化)输入
  if (this->phase_ == TEST) {
      this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(),
          bottom[0]->count());
  }
  // Trim weights  修剪(量化)权值
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
      this->weights_quantized_[0]->mutable_cpu_data());
  if (this->bias_term_) {
    caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
        this->weights_quantized_[1]->mutable_cpu_data());
  }
  int rounding = this->phase_ == TEST ? this->rounding_ :
      QuantizationParameter_Rounding_STOCHASTIC;
  this->QuantizeWeights_cpu(this->weights_quantized_, rounding,
      this->bias_term_);
  // Do forward propagation 前向传播
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->weights_quantized_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, this->transpose_ ? CblasNoTrans :
      CblasTrans, this->M_, this->N_, this->K_, (Dtype)1., bottom_data, weight,
      (Dtype)0., top_data);
  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1,
        (Dtype)1., this->bias_multiplier_.cpu_data(),
        this->weights_quantized_[1]->cpu_data(), (Dtype)1., top_data);
  }
  // Trim layer output 量化输出
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
  }
}

// 反向传播
template <typename Dtype>
void FcRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight  权值梯度
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->K_, this->N_, this->M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->N_, this->K_, this->M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias  偏置梯度
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype)1., top_diff,
        this->bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data  底层数据梯度
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->weights_quantized_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->weights_quantized_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FcRistrettoLayer);
#endif

INSTANTIATE_CLASS(FcRistrettoLayer);
REGISTER_LAYER_CLASS(FcRistretto);

}  // namespace caffe
