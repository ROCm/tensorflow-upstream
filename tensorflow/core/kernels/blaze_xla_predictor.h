#ifndef TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_
#define TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_

#include "tensorflow/core/kernels/blaze_predictor.h"

namespace tensorflow {
typedef std::map<std::string, std::vector<NodeDef>> InputNodeMap;
typedef std::map<std::string, NodeDef> NodeMap;

class BlazeXlaPredictor : public BlazePredictor {
 public:
  BlazeXlaPredictor(OpKernelConstruction* ctx) : BlazePredictor(ctx) {
    warmuped_ = false;
    warmuping_ = false;
  }

  BlazeXlaPredictor(const std::vector<std::string>& input_names,
                          const std::vector<std::string>& output_names,
                          const GraphDef& graph_def, const std::string& device,
                          const BlazeKernelOptions& options, const string& device_string,
                          const std::vector<DataType>& input_types,
                          OpKernelConstruction* ctx = nullptr)
      : BlazePredictor(input_names, output_names, graph_def,
                       device, options, device_string, input_types, ctx) {
        warmuped_ = false;
        warmuping_ = false;
      }

  ~BlazeXlaPredictor() override {}

  Status Compute(OpKernelContext* ctx) override;
//  void ComputeNull(OpKernelContext* ctx) override;
 private:
  Status FindBlackPaddingInputs();
  InputNodeMap ToInputNodeMap();

  Status PrepareData() override;

  Status PadToStatic(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>* padded_inputs,
                     int batchsize, int pad_to_batchsize,
                     OpKernelContext* ctx);

  Status PadToStaticCPUToGPU(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>* padded_inputs,
                     int batchsize, int pad_to_batchsize,
                     OpKernelContext* ctx);

  Status SliceToDynamic(const std::vector<Tensor>& padded_outputs,
                        int batchsize, int pad_to_batchsize,
                        std::vector<Tensor>& outputs, OpKernelContext* ctx);
  Status SliceToDynamicCPU(const std::vector<Tensor>& padded_outputs,
                        int batchsize, int pad_to_batchsize,
                        std::vector<Tensor>& outputs, OpKernelContext* ctx);
  int InferBatchSize(const std::vector<Tensor>& tensors);

  int AddNewBatchSize(int padded_size);
  
  Status InitXlaWarmup();
  Status Warmup() override;
  Status Warmup(OpKernelContext* ctx);
  Status CheckShape(const TensorShapeProto& shape);

  std::vector<int32> batch_sizes_;
  std::vector<bool> skip_padding_;
  NodeMap node_map_;
  
  bool warmuped_;
  bool warmuping_;
  mutex warmup_mu_;
  mutex batch_size_mu_;
};
}
#endif
