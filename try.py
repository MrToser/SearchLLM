import os
import sys
import time
import torch
import subprocess
import ray
from typing import Dict, Any

def diagnose_gpu_environment():
    """诊断GPU环境的实用函数"""
    print("=" * 50)
    print("GPU环境诊断")
    print("=" * 50)
    
    # 系统信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA可用性（即使使用AMD GPU也检查，因为某些库可能依赖此标志）
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"可见GPU数量: {device_count}")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查ROCm/HIP环境
    print("\nROCm/HIP环境检查:")
    try:
        # 尝试检测HIP是否启用
        is_hip_enabled = hasattr(torch, 'hip') or '+rocm' in torch.__version__
        print(f"PyTorch HIP支持: {is_hip_enabled}")
        
        # 尝试运行rocm-smi
        try:
            rocm_smi_output = subprocess.check_output(
                ["rocm-smi"], universal_newlines=True, stderr=subprocess.STDOUT
            )
            print("ROCm-SMI输出:")
            print("-" * 40)
            print(rocm_smi_output)
            print("-" * 40)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"无法运行rocm-smi: {e}")
            
        # 尝试直接使用GPU
        print("\n尝试在PyTorch中使用GPU:")
        try:
            # 尝试创建一个张量并移动到GPU
            test_tensor = torch.ones(1)
            
            # 首先尝试常规的.cuda()方法
            try:
                gpu_tensor = test_tensor.cuda()
                print("成功使用.cuda()方法访问GPU")
                del gpu_tensor
            except Exception as e:
                print(f".cuda()方法失败: {e}")
            
            # 然后尝试使用device参数
            try:
                # 对AMD GPU，尝试使用hip设备
                gpu_tensor = torch.ones(1, device="cuda:0")
                print("成功使用device='hip:0'访问GPU")
                del gpu_tensor
            except Exception as e:
                print(f"device='hip:0'失败: {e}")
                
            # 最后尝试通用方法
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                gpu_tensor = test_tensor.to(device)
                print(f"成功使用to({device})访问设备")
                del gpu_tensor
            except Exception as e:
                print(f"to(device)失败: {e}")
                
        except Exception as e:
            print(f"测试GPU访问时发生错误: {e}")
    
    except Exception as e:
        print(f"检查HIP/ROCm环境时出错: {e}")
    
    # 环境变量
    print("\n与GPU相关的环境变量:")
    relevant_vars = [
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "HSA_ENABLE_SDMA", "GPU_MAX_HEAP_SIZE", "GPU_SINGLE_ALLOC_PERCENT"
    ]
    for var in relevant_vars:
        print(f"  {var}: {os.environ.get(var, '未设置')}")
    
    print("=" * 50)

@ray.remote(num_gpus=1)
class GPUWorker:
    """测试在Ray中使用GPU的worker类"""
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_type = "GPU" if self.device.type == "cuda" else "CPU"
        self.device_info = (
            torch.cuda.get_device_name(0) if self.device.type == "cuda" else "N/A"
        )

    def test_gpu(self):
        """测试GPU是否可以使用"""
        try:
            # 创建一个简单的张量并移动到GPU
            x = torch.ones(10, 10)
            x = x.to(self.device)
            
            # 执行一个简单的操作
            y = x + x
            
            # 获取一些基本信息
            result = {
                "device_type": self.device_type,
                "device_info": self.device_info,
                "tensor_device": str(y.device),
                "operation_successful": True
            }
            
            # 如果是GPU，获取更多信息
            if self.device.type == "cuda":
                result.update({
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved(),
                    "max_memory_allocated": torch.cuda.max_memory_allocated()
                })
                
            return result
        except Exception as e:
            return {
                "device_type": self.device_type,
                "device_info": self.device_info,
                "operation_successful": False,
                "error": str(e)
            }

def configure_ray_for_gpu():
    """配置并初始化Ray以使用GPU"""
    
    # 设置关键环境变量
    gpu_env = {
        # 常规环境变量
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        
        # AMD ROCm特定环境变量
        "HSA_ENABLE_SDMA": "0",         # 禁用SDMA可能有助于稳定性
        "GPU_MAX_HEAP_SIZE": "100",     # 控制GPU堆大小(%)
        "GPU_SINGLE_ALLOC_PERCENT": "100",  # 允许大型单次内存分配
        
        # 确保GPU可见
        # "HIP_VISIBLE_DEVICES": "0",     # 使用第一个AMD GPU
        # "ROCR_VISIBLE_DEVICES": "0",    # 同上，某些版本需要
    }
    
    # 更新当前进程的环境变量
    for k, v in gpu_env.items():
        os.environ[k] = v
    
    # 检查系统环境
    diagnose_gpu_environment()
    
    # 初始化Ray
    print("\n初始化Ray...")
    ray.init(
        num_gpus=1,                  # 明确声明有1个GPU
        runtime_env={"env_vars": gpu_env},  # 传递环境变量到workers
        ignore_reinit_error=True,    # 忽略重新初始化错误
        include_dashboard=False,     # 简化起见关闭仪表板
    )
    
    # 打印Ray资源信息
    print("\nRay集群资源:")
    print(ray.cluster_resources())
    
    return ray.cluster_resources()

def test_gpu_in_ray():
    """测试Ray框架中的GPU功能"""
    print("\n创建GPU worker...")
    worker = GPUWorker.remote()
    
    print("测试GPU访问...")
    result = ray.get(worker.test_gpu.remote())
    
    print("\nGPU测试结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    return result

def main():
    """主函数"""
    print("启动GPU在Ray框架下的诊断程序...")
    
    # 配置并初始化Ray
    resources = configure_ray_for_gpu()
    
    # 如果Ray成功识别了GPU资源
    if resources.get("GPU", 0) > 0:
        print(f"\n✅ Ray成功识别了 {resources['GPU']} 个GPU资源")
        
        # 测试在Ray中使用GPU
        result = test_gpu_in_ray()
        
        if result["operation_successful"]:
            print(f"\n✅ 成功在Ray worker中使用GPU ({result['device_info']})")
        else:
            print(f"\n❌ 在Ray worker中使用GPU失败: {result.get('error', 'Unknown error')}")
    else:
        print("\n❌ Ray未能识别任何GPU资源")
    
    # 关闭Ray
    print("\n关闭Ray...")
    ray.shutdown()
    print("完成!")

if __name__ == "__main__":
    main()