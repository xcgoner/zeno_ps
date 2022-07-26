// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "ops.h"
#include <atomic>
#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace byteps {
namespace mxnet {

namespace {

std::atomic_int op_count;
const auto MX_EXEC_CTX = Context();
const auto MX_FUNC_PROP = FnProperty::kCPUPrioritized;

// struct to hold parameters for pushpull with MXNet Engine
struct PushPullParam {
  BPSContext* context;
  std::shared_ptr<NDArray> input;
  int version;
  int priority;

  PushPullParam(BPSContext* context, std::shared_ptr<NDArray> input, int version, int priority)
      : context(context), input(input), version(version), priority(priority) {}
};

// callback function to release parameters used for pushpull with MXNet Engine
void DeletePushPullParam(void* param) {
  auto push_pull_param = static_cast<PushPullParam*>(param);
  delete push_pull_param;
}

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return prefix + ".noname." + std::to_string(op_count);
}
}  // namespace

inline void InvokeCompleteCallback(Callback on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

void DoPushPull(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());
  auto on_complete = *static_cast<Callback*>(on_complete_ptr);
  auto push_pull_param = static_cast<PushPullParam*>(param);
  int priority = push_pull_param->priority;
  int version = push_pull_param->version;
  NDArray* input = push_pull_param->input.get();
  BPSContext& context = *push_pull_param->context;

  auto device = TensorUtil::GetDevice(input);
  auto byteps_input = std::make_shared<MXTensor<NDArray>>(input);
  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_input, byteps_input, nullptr, device, priority, version,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      },
      queue_list);
  ThrowIfError(enqueue_result);
}

void DoPush(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());
  auto on_complete = *static_cast<Callback*>(on_complete_ptr);
  auto push_pull_param = static_cast<PushPullParam*>(param);
  int priority = push_pull_param->priority;
  int version = push_pull_param->version;
  NDArray* input = push_pull_param->input.get();
  BPSContext& context = *push_pull_param->context;

  auto device = TensorUtil::GetDevice(input);
  auto byteps_input = std::make_shared<MXTensor<NDArray>>(input);
  auto queue_list = common::GetPushQueueList(device);

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_input, nullptr, nullptr, device, priority, version,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      },
      queue_list);
  ThrowIfError(enqueue_result);
}

void DoPull(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());
  auto on_complete = *static_cast<Callback*>(on_complete_ptr);
  auto push_pull_param = static_cast<PushPullParam*>(param);
  int priority = push_pull_param->priority;
  int version = push_pull_param->version;
  NDArray* output = push_pull_param->input.get();
  BPSContext& context = *push_pull_param->context;

  auto device = TensorUtil::GetDevice(output);
  auto byteps_output = std::make_shared<MXTensor<NDArray>>(output);
  auto queue_list = common::GetPullQueueList(device);

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_output, byteps_output, nullptr, device, priority, version,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      },
      queue_list);
  ThrowIfError(enqueue_result);
}

extern "C" int byteps_mxnet_push_pull_async(NDArray* tensor, char* name,
                                            int version, int priority,
                                            bool is_average) {
  MX_API_BEGIN();

  std::string tensor_name = GetOpName("byteps", name);

  // We need to create a shared_ptr to NDArray object with
  // shallow copy to prevent from NDArray object being freed
  // before MXNet engine process it
  auto tensor_copy = std::make_shared<NDArray>(*tensor);
  auto& context = common::GetContextFromName(tensor_name);
  auto dtype = TensorUtil::GetDType(tensor);
  auto size = TensorUtil::GetSize(tensor);
  auto device = TensorUtil::GetDevice(tensor);
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(
                            std::make_shared<MXTensor<NDArray>>(tensor_copy.get())->data())
                      : nullptr;
  common::InitTensor(context, size, dtype, cpubuff);

  auto push_pull_param = new PushPullParam(&context, tensor_copy, version, priority);
  auto var = tensor->var();
  // Use MXEnginePushAsync instead of Engine::Get()->PushAsync to avoid ABI
  // compatibility issues
  MXEnginePushAsync(DoPushPull, push_pull_param, DeletePushPullParam,
                    &MX_EXEC_CTX, nullptr, 0, &var, 1,
                    &MX_FUNC_PROP, 0, "BytePSPushPull");

  if (is_average) {
    // average the aggregated gradient
    auto num_worker = byteps_size();
    *tensor /= num_worker;
  }

  MX_API_END();
}

extern "C" int byteps_mxnet_push_async(NDArray* tensor, char* name,
                                            int version, int priority) {
  MX_API_BEGIN();

  std::string tensor_name = GetOpName("byteps", name);

  // We need to create a shared_ptr to NDArray object with
  // shallow copy to prevent from NDArray object being freed
  // before MXNet engine process it
  auto tensor_copy = std::make_shared<NDArray>(*tensor);
  auto& context = common::GetContextFromName(tensor_name);
  auto dtype = TensorUtil::GetDType(tensor);
  auto size = TensorUtil::GetSize(tensor);
  auto device = TensorUtil::GetDevice(tensor);
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(
                            std::make_shared<MXTensor<NDArray>>(tensor_copy.get())->data())
                      : nullptr;
  common::InitTensor(context, size, dtype, cpubuff);

  auto push_pull_param = new PushPullParam(&context, tensor_copy, version, priority);
  auto var = tensor->var();
  // Use MXEnginePushAsync instead of Engine::Get()->PushAsync to avoid ABI
  // compatibility issues
  MXEnginePushAsync(DoPush, push_pull_param, DeletePushPullParam,
                    &MX_EXEC_CTX, &var, 1, nullptr, 0,
                    &MX_FUNC_PROP, 0, "BytePSPush");

  MX_API_END();
}

extern "C" int byteps_mxnet_pull_async(NDArray* tensor, char* name,
                                            int version, int priority) {
  MX_API_BEGIN();

  std::string tensor_name = GetOpName("byteps", name);

  // We need to create a shared_ptr to NDArray object with
  // shallow copy to prevent from NDArray object being freed
  // before MXNet engine process it
  auto tensor_copy = std::make_shared<NDArray>(*tensor);
  auto& context = common::GetContextFromName(tensor_name);
  auto dtype = TensorUtil::GetDType(tensor);
  auto size = TensorUtil::GetSize(tensor);
  auto device = TensorUtil::GetDevice(tensor);
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(
                            std::make_shared<MXTensor<NDArray>>(tensor_copy.get())->data())
                      : nullptr;
  common::InitTensor(context, size, dtype, cpubuff);

  auto push_pull_param = new PushPullParam(&context, tensor_copy, version, priority);
  auto var = tensor->var();
  // Use MXEnginePushAsync instead of Engine::Get()->PushAsync to avoid ABI
  // compatibility issues
  MXEnginePushAsync(DoPull, push_pull_param, DeletePushPullParam,
                    &MX_EXEC_CTX, nullptr, 0, &var, 1,
                    &MX_FUNC_PROP, 0, "BytePSPull");

  MX_API_END();
}

extern "C" void byteps_mxnet_declare_tensor(char* name) {
  std::string tensor_name = GetOpName("byteps", name);
  common::IsTensorDeclared(tensor_name);
  // // debug
  // auto& context = common::GetContextFromName(tensor_name);
  // std::cout << "key of " << tensor_name << " is " << context.declared_key << std::endl;
  return;
}

extern "C" void byteps_mxnet_declare_and_init_tensor(char* name, NDArray* tensor) {
  std::string tensor_name = GetOpName("byteps", name);

  common::IsTensorDeclared(tensor_name);
  
  // initialize tensor on server
  auto tensor_copy = std::make_shared<NDArray>(*tensor);
  auto& context = common::GetContextFromName(tensor_name);
  auto dtype = TensorUtil::GetDType(tensor);
  auto size = TensorUtil::GetSize(tensor);
  auto device = TensorUtil::GetDevice(tensor);
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(
                            std::make_shared<MXTensor<NDArray>>(tensor_copy.get())->data())
                      : nullptr;
  common::InitTensor(context, size, dtype, cpubuff);

  return;
}

}  // namespace mxnet
}  // namespace byteps
