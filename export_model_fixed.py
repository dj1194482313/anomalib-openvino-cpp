#!/usr/bin/env python3
"""
PatchCore模型导出脚本
修复CUDA/CPU设备不匹配问题，支持从anomalib checkpoint导出OpenVINO格式
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np
from torch import nn
import torchvision.transforms as transforms

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.warning("OpenVINO not available. Please install openvino-dev.")

try:
    from anomalib.models import get_model
    from anomalib.config import get_configurable_parameters
    from anomalib.utils.callbacks import get_callbacks
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    logging.warning("Anomalib not available. Please install anomalib.")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeviceCompatibilityFixer:
    """设备兼容性修复器"""
    
    @staticmethod
    def move_model_to_cpu(model: nn.Module) -> nn.Module:
        """将模型移动到CPU并确保所有参数在CPU上"""
        try:
            # 强制移动到CPU
            model = model.cpu()
            
            # 确保所有参数都在CPU上
            for name, param in model.named_parameters():
                if param.device.type != 'cpu':
                    param.data = param.data.cpu()
                    
            # 确保所有buffer都在CPU上  
            for name, buffer in model.named_buffers():
                if buffer.device.type != 'cpu':
                    buffer.data = buffer.data.cpu()
                    
            logger.info("Model successfully moved to CPU")
            return model
            
        except Exception as e:
            logger.error(f"Failed to move model to CPU: {e}")
            raise
    
    @staticmethod
    def fix_tensor_devices(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """修复tensor字典中的设备不匹配问题"""
        fixed_dict = {}
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                fixed_dict[key] = tensor.cpu()
            else:
                fixed_dict[key] = tensor
        return fixed_dict


class PatchCoreExporter:
    """PatchCore模型导出器"""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查依赖
        if not ANOMALIB_AVAILABLE:
            raise ImportError("Anomalib is required but not available")
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is required but not available")
            
        logger.info(f"Initialized exporter:")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Output: {self.output_dir}")
    
    def load_model_from_checkpoint(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """从checkpoint加载模型"""
        try:
            logger.info("Loading model from checkpoint...")
            
            # 加载配置
            config = get_configurable_parameters(config_path=self.config_path)
            
            # 强制设置设备为CPU以避免设备不匹配
            config.trainer.accelerator = "cpu"
            config.trainer.devices = 1
            
            # 创建模型
            model = get_model(config)
            
            # 加载checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # 修复设备兼容性问题
            if 'state_dict' in checkpoint:
                state_dict = DeviceCompatibilityFixer.fix_tensor_devices(checkpoint['state_dict'])
            else:
                state_dict = DeviceCompatibilityFixer.fix_tensor_devices(checkpoint)
                
            # 加载权重
            model.load_state_dict(state_dict, strict=False)
            
            # 确保模型在CPU上
            model = DeviceCompatibilityFixer.move_model_to_cpu(model)
            model.eval()
            
            logger.info("Model loaded successfully")
            return model, config
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_dummy_input(self, config: Dict[str, Any]) -> torch.Tensor:
        """创建dummy输入用于模型跟踪"""
        try:
            # 从配置获取输入尺寸
            if hasattr(config.dataset, 'image_size'):
                if isinstance(config.dataset.image_size, (list, tuple)):
                    height, width = config.dataset.image_size
                else:
                    height = width = config.dataset.image_size
            else:
                # 默认尺寸
                height, width = 224, 224
                logger.warning(f"Using default image size: {height}x{width}")
            
            # 创建dummy输入 [batch_size, channels, height, width]
            dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)
            
            logger.info(f"Created dummy input with shape: {dummy_input.shape}")
            return dummy_input
            
        except Exception as e:
            logger.error(f"Failed to create dummy input: {e}")
            # 使用默认值
            return torch.randn(1, 3, 224, 224, dtype=torch.float32)
    
    def export_to_onnx(self, model: nn.Module, config: Dict[str, Any]) -> Path:
        """导出模型到ONNX格式"""
        try:
            logger.info("Exporting model to ONNX...")
            
            onnx_path = self.output_dir / "model.onnx"
            dummy_input = self.create_dummy_input(config)
            
            # 确保输入在CPU上
            dummy_input = dummy_input.cpu()
            
            # 导出ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['anomaly_map', 'pred_score'] if hasattr(model, 'training_outputs') else ['anomaly_map'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'anomaly_map': {0: 'batch_size'}
                } if config.get('optimization', {}).get('export_mode') == 'dynamic' else None
            )
            
            logger.info(f"ONNX model saved to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            raise
    
    def convert_to_openvino(self, onnx_path: Path) -> Path:
        """转换ONNX模型到OpenVINO IR格式"""
        try:
            logger.info("Converting ONNX to OpenVINO IR...")
            
            # 使用OpenVINO Model Optimizer
            from openvino.tools import mo
            
            ir_path = self.output_dir / "model.xml"
            
            # 转换模型
            ov_model = mo.convert_model(
                str(onnx_path),
                input_shape=[1, 3, 224, 224],  # 可以根据需要调整
                compress_to_fp16=False  # 可以设置为True以减小模型大小
            )
            
            # 保存模型
            ov.serialize(ov_model, str(ir_path))
            
            logger.info(f"OpenVINO IR model saved to: {ir_path}")
            return ir_path
            
        except Exception as e:
            logger.error(f"Failed to convert to OpenVINO IR: {e}")
            raise
    
    def generate_metadata(self, config: Dict[str, Any], ir_path: Path) -> Path:
        """生成模型元数据文件"""
        try:
            logger.info("Generating metadata...")
            
            # 构建元数据
            metadata = {
                "image_threshold": getattr(config.metrics.threshold, 'image', 0.5),
                "pixel_threshold": getattr(config.metrics.threshold, 'pixel', 0.5),
                "min": 0.0,  # 这些值通常需要从训练统计中获取
                "max": 1.0,
                "transform": {
                    "transform": {
                        "transforms": [
                            {
                                "height": getattr(config.dataset, 'image_size', [224, 224])[0] if isinstance(getattr(config.dataset, 'image_size', [224, 224]), list) else getattr(config.dataset, 'image_size', 224),
                                "width": getattr(config.dataset, 'image_size', [224, 224])[1] if isinstance(getattr(config.dataset, 'image_size', [224, 224]), list) else getattr(config.dataset, 'image_size', 224)
                            }
                        ]
                    }
                }
            }
            
            # 保存元数据
            metadata_path = ir_path.parent / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to generate metadata: {e}")
            raise
    
    def export(self) -> Dict[str, Path]:
        """执行完整的导出流程"""
        try:
            logger.info("Starting model export...")
            
            # 1. 加载模型
            model, config = self.load_model_from_checkpoint()
            
            # 2. 导出ONNX
            onnx_path = self.export_to_onnx(model, config)
            
            # 3. 转换到OpenVINO
            ir_path = self.convert_to_openvino(onnx_path)
            
            # 4. 生成元数据
            metadata_path = self.generate_metadata(config, ir_path)
            
            result = {
                'onnx': onnx_path,
                'openvino_xml': ir_path,
                'openvino_bin': ir_path.with_suffix('.bin'),
                'metadata': metadata_path
            }
            
            logger.info("Export completed successfully!")
            logger.info("Generated files:")
            for key, path in result.items():
                logger.info(f"  {key}: {path}")
                
            return result
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Export PatchCore model to OpenVINO format")
    parser.add_argument("--config", "-c", type=str, required=True,
                       help="Path to anomalib config file")
    parser.add_argument("--checkpoint", "-k", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for exported model")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 检查输入文件
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        # 创建导出器并执行导出
        exporter = PatchCoreExporter(args.config, args.checkpoint, args.output)
        results = exporter.export()
        
        print("\n✅ Export completed successfully!")
        print(f"📁 Output directory: {args.output}")
        print("📋 Generated files:")
        for key, path in results.items():
            print(f"   {key}: {path.name}")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()