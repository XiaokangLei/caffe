#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <string>
#include <stdio.h>

#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseRistrettoLayer<Dtype>::BaseRistrettoLayer() {
  // Initialize random number generator
  srand(time(NULL));//生成随机种子
}

//量化卷积参数
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_cpu(
      vector<shared_ptr<Blob<Dtype> > > weights_quantized, const int rounding,//量化权值w、取整策略和偏置b
      const bool bias_term) {
  Dtype* weight = weights_quantized[0]->mutable_cpu_data();//权重w
  const int cnt_weight = weights_quantized[0]->count();//数量
  switch (precision_) {
  //迷你浮点数量化===
  case QuantizationParameter_Precision_MINIFLOAT:
    Trim2MiniFloat_cpu(weight, cnt_weight, fp_mant_, fp_exp_, rounding);
    //bias的范围和weight的范围有时候不一致，造成较大的损失，可以考虑不进行量化
	if (bias_term) {
      Trim2MiniFloat_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), fp_mant_, fp_exp_, rounding);
    }
    break;
  //动态定点量化===
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    //Trim2FixedPoint_cpu(weight, cnt_weight, bw_params_, rounding, fl_params_);
    Trim2FixedPoint_cpu_weightshow(weight, cnt_weight, bw_params_, rounding, fl_params_);
    if (bias_term) {
      Trim2FixedPoint_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), bw_params_, rounding, fl_params_);
      //Trim2FixedPoint_cpu_weightshow(weights_quantized[1]->mutable_cpu_data(),
       //   weights_quantized[1]->count(), bw_params_, rounding, fl_params_);
    }
    break;
  //2乘方量化===
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    Trim2IntegerPowerOf2_cpu(weight, cnt_weight, pow_2_min_exp_, pow_2_max_exp_,
        rounding);
    // Don't trim bias
    break;
	// IAO量化
	//  case QuantizationParameter_Precision_INTEGER_ARITHMETRIC_ONLY:
	//    Trim2IAO_cpu(weight, cnt_weight);
	//    // Don't trim bias
	//    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << precision_;
    break;
  }
}

//量化层输入=======
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerInputs_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
	//动态定点量化
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      //Trim2FixedPoint_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      Trim2FixedPoint_cpu_inputshow(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
	//迷你浮点数量化
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
	  // IAO量化
	//    case QuantizationParameter_Precision_INTEGER_ARITHMETRIC_ONLY:
	//      Trim2IAO_cpu(data, count);
	//    // Don't trim bias
	//      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

//量化层输出===
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
	//动态定点量化
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      //Trim2FixedPoint_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      Trim2FixedPoint_cpu_outputshow(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
	//迷你浮点数量化
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
	  // IAO量化===============================
	//    case QuantizationParameter_Precision_INTEGER_ARITHMETRIC_ONLY:
	//      Trim2IAO_cpu(data, count);
	//   // Don't trim bias
	//      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

//定点小数量化方法
// 传参： 起始指针/数量/量化总位宽/取整策略/小数位位宽
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  FILE *fp = NULL;
  fp = fopen("1.txt","w");
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
	// 例如 0 1 1 0 1 1 0 1 , bit_width= 8, fl = 2
// 最高位符号位，所以余下的数为 1 1 0 1 1 0 1 = 109D (假设全部作为整数位)
// 实际小时位有2位，所以小数点需要向前移动 2位
// 所以实际表示的数为 2^0 * 109 * 2(-2) = 27.25
// 所以 总位宽 bit_width 小数位长度 fl
// 表示的最大数为  (2^(bit_width-1) - 1)/(2^fl) = (2^(bit_width - 1) - 1)*(2^(-fl))
// 最小的数为 -1 * (2^(bit_width - 1) - 1)*(2^(-fl))
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    // 首先数据包和处理，比最大值小，比最小值大
	data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);// 按小数位乘方系数 放大或者缩小
    // data[index] *= pow(2, fl);
	fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
	// 放大/缩小后再取整================== 109.005 ----> 109/110
    switch (rounding) {
		// 最近偶数（NEAREST）
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
	  // 随机舍入（STOCHASTIC）
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
	// 取整后再放大/缩小 109/110 ----> 109  / 2^2  =  27.25
    data[index] *= pow(2, -fl);
	//data[index] /= pow(2, fl);
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    fprintf(fp,"\n");
	}
fclose(fp);
}

/* 部分调整 借鉴inq，逐步量化，先从最大的部分开始量化
// float quant_percent = 0.3; // 需要在前面 定义
////////////////////  固定点量化
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu(
Dtype* data, const int cnt,
const int bit_width, 
const int rounding, 
int fl) 

 {
	// 上下限计算
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    
	// 获取 有序 数列
    Dtype* data_copy=(Dtype*) malloc(cnt*sizeof(Dtype));
    caffe_copy(cnt,data,data_copy);
    caffe_abs(cnt,data_copy,data_copy);
    std::sort(data_copy,data_copy+cnt); //data_copy order from small to large
	
   int partition=int(cnt*(1-quant_percent))-1;
 
    for (int index = 0; index < cnt; ++index) 
    {
	   if(std::abs(data[index]) >= data_copy[partition])
	    {
			// Saturate data
			data[index] = std::max(std::min(data[index], max_data), min_data);
			// Round data
			data[index] /= pow(2, -fl);
			switch (rounding) 
			{
			case QuantizationParameter_Rounding_NEAREST:
			  data[index] = round(data[index]);
			  break;
			case QuantizationParameter_Rounding_STOCHASTIC:
			  data[index] = floor(data[index] + RandUniform_cpu());
			  break;
			default:
			  break;
			}
			data[index] *= pow(2, -fl);
		// mask_vec[i]=0;  // 标记量化标志
	    }
    }
   free(data_copy);// 释放空间
}
*/

// IAO整数 uint8 量化===========================================
//template <typename Dtype>
//void BaseRistrettoLayer<Dtype>::Trim2IAO_cpu(Dtype* data, const int cnt){
// 
//}

///////////////////////////////////////////
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu_weightshow(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  FILE *fp = NULL;
  fp = fopen("fc_weight.txt","w");
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");

    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    fprintf(fp,"%d",bit_width);
    fprintf(fp,"\t");
    fprintf(fp,"%d",fl);
    fprintf(fp,"\t");
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    data[index] *= pow(2, -fl);
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    fprintf(fp,"\n");
	}
fclose(fp);
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu_inputshow(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  FILE *fp = NULL;
  fp = fopen("fc_input.txt","w");
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    fprintf(fp,"%d",bit_width);
    fprintf(fp,"\t");
    fprintf(fp,"%d",fl);
    fprintf(fp,"\t");
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    data[index] *= pow(2, -fl);
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    fprintf(fp,"\n");
	}
fclose(fp);
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu_outputshow(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  FILE *fp = NULL;
  fp = fopen("fc_output.txt","w");
  fstream file("fc_output.txt",ios::out);
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    fprintf(fp,"%d",bit_width);
    fprintf(fp,"\t");
    fprintf(fp,"%d",fl);
    fprintf(fp,"\t");
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    data[index] *= pow(2, -fl);
    fprintf(fp,"%f",data[index]);
    fprintf(fp,"\t");
    fprintf(fp,"\n");
	}
fclose(fp);
}

////////////////////////////////////////

//迷你浮点数量化方法====
typedef union {
  float d;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

// 迷你浮点数量化  传参：尾数位、指数位、符号位
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2MiniFloat_cpu(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding) {
  for (int index = 0; index < cnt; ++index) {
    int bias_out = pow(2, bw_exp - 1) - 1;
    float_cast d2;
    // 将输入转换为单精度
    d2.d = (float)data[index];
    int exponent=d2.parts.exponent - 127 + bias_out;
    double mantisa = d2.parts.mantisa;
    // Special case: input is zero or denormalized number
	//特殊情况：输入为0或者非规范数字
    if (d2.parts.exponent == 0) {
      data[index] = 0;
      return;
    }
    // Special case: denormalized number as output
    if (exponent < 0) {
      data[index] = 0;
      return;
    }
    // Saturation: input float is larger than maximum output float
    int max_exp = pow(2, bw_exp) - 1;
    int max_mant = pow(2, bw_mant) - 1;
    if (exponent > max_exp) {
      exponent = max_exp;
      mantisa = max_mant;
    } else {
      // Convert mantissa from long format to short one. Cut off LSBs.
      double tmp = mantisa / pow(2, 23 - bw_mant);
      switch (rounding) {
      case QuantizationParameter_Rounding_NEAREST:
        mantisa = round(tmp);
        break;
      case QuantizationParameter_Rounding_STOCHASTIC:
        mantisa = floor(tmp + RandUniform_cpu());
        break;
      default:
        break;
      }
    }
    // Assemble result
    data[index] = pow(-1, d2.parts.sign) * ((mantisa + pow(2, bw_mant)) /
        pow(2, bw_mant)) * pow(2, exponent - bias_out);
	}
}

//2乘方量化方法
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2IntegerPowerOf2_cpu(Dtype* data,
      const int cnt, const int min_exp, const int max_exp, const int rounding) {
	for (int index = 0; index < cnt; ++index) {
    float exponent = log2f((float)fabs(data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = round(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      exponent = floorf(exponent + RandUniform_cpu());
      break;
    default:
      break;
    }
    exponent = std::max(std::min(exponent, (float)max_exp), (float)min_exp);
    data[index] = sign * pow(2, exponent);
	}
}

template <typename Dtype>
double BaseRistrettoLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template BaseRistrettoLayer<double>::BaseRistrettoLayer();
template BaseRistrettoLayer<float>::BaseRistrettoLayer();
template void BaseRistrettoLayer<double>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<double> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<float>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<float> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<double>::QuantizeLayerInputs_cpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerInputs_cpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_cpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_cpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseRistrettoLayer<double>::Trim2MiniFloat_cpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2MiniFloat_cpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseRistrettoLayer<double>::Trim2IntegerPowerOf2_cpu(double* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template void BaseRistrettoLayer<float>::Trim2IntegerPowerOf2_cpu(float* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template double BaseRistrettoLayer<double>::RandUniform_cpu();
template double BaseRistrettoLayer<float>::RandUniform_cpu();

}  // namespace caffe
