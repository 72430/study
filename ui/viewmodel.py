"""
ViewModel 层 - MVVM模式的核心，负责UI与后端逻辑的绑定与数据同步
集成模型管理器，提供命名验证、模型加载/保存全生命周期管理
"""
import os
import json
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from config.config import TrainingConfig, UIConfig
from core.inferencer import Inferencer, InferenceResult, ModelLoadResult
from core.model_manager import ModelManager, OperationHistory
from ui.training_thread import TrainingThread
from utils.logger import get_logger

logger = get_logger("viewmodel")


class MainViewModel(QObject):
    """
    主界面ViewModel

    职责:
      - 管理UI状态
      - 协调训练/推理后端操作
      - 模型名称验证与唯一性校验
      - 模型加载完整性检查与信息展示
      - 操作历史记录管理
    """

    # ---- 数据路径信号 ----
    data_path_changed = pyqtSignal(str)
    data_count_changed = pyqtSignal(int)
    inference_path_changed = pyqtSignal(str)

    # ---- 参数信号 ----
    params_saved = pyqtSignal(str)

    # ---- 模型加载信号 ----
    model_load_finished = pyqtSignal(object)        # ModelLoadResult
    model_list_updated = pyqtSignal(list)           # 模型列表

    # ---- 训练信号 ----
    training_started = pyqtSignal()
    training_paused = pyqtSignal()
    training_resumed = pyqtSignal()
    training_stopped = pyqtSignal()
    training_progress = pyqtSignal(int, int, float)
    training_losses = pyqtSignal(list)
    training_status = pyqtSignal(str)
    training_finished = pyqtSignal(bool, str)
    training_time = pyqtSignal(float)

    # ---- 推理信号 ----
    inference_image_list = pyqtSignal(list)
    inference_result = pyqtSignal(object)
    inference_batch_done = pyqtSignal(int, int)
    inference_clear = pyqtSignal()

    # ---- 操作历史信号 ----
    operation_history_updated = pyqtSignal(str)

    # ---- 系统信号 ----
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.training_cfg = TrainingConfig()
        self.ui_cfg = UIConfig()

        # 后端对象
        self.trainer = None
        self.inferencer: Optional[Inferencer] = None
        self.training_thread: Optional[TrainingThread] = None

        # 模型管理器（直接操作文件系统）
        self.model_manager = ModelManager(self.training_cfg.checkpoint_dir)
        self.op_history = OperationHistory()

        # 推理缓存
        self._inference_results: List[InferenceResult] = []
        self._inference_image_paths: List[str] = []

        # 参数配置文件路径
        self._param_file = str(
            Path(__file__).resolve().parent.parent / "config" / "user_params.json"
        )

    # ==================== 模型列表管理 ====================

    def list_saved_models(self) -> list:
        """获取所有已保存的模型列表"""
        models = self.model_manager.list_models()
        self.model_list_updated.emit(models)
        return models

    # ==================== 模型加载管理 ====================

    @pyqtSlot(str)
    def load_model_file(self, file_path: str):
        """
        加载选中的模型文件（完整验证链）

        Args:
            file_path: 模型文件路径
        """
        if not os.path.exists(file_path):
            self.error_occurred.emit(f"模型文件不存在: {file_path}")
            return

        # 初始化推理引擎并加载模型
        try:
            if self.inferencer is None:
                self.inferencer = Inferencer(self.training_cfg)

            # 执行带验证的加载
            load_result = self.inferencer.load_model_with_validation(file_path)

            # 发射结果给UI
            self.model_load_finished.emit(load_result)

            if load_result.success:
                self.status_message.emit(load_result.message)
                # 显示模型摘要
                if load_result.summary_text:
                    self.status_message.emit(load_result.summary_text)
                # 更新操作历史
                history = self.inferencer.get_operation_history()
                self.operation_history_updated.emit(history)
            else:
                self.error_occurred.emit(load_result.message)

        except Exception as e:
            self.error_occurred.emit(f"模型加载失败: {str(e)}")

    @pyqtSlot()
    def refresh_model_list(self):
        """刷新模型列表"""
        self.list_saved_models()
        self.status_message.emit("模型列表已刷新")

    def get_model_summary_text(self) -> str:
        """获取当前加载模型的摘要"""
        if self.inferencer:
            if self.inferencer._fb_mode:
                info = self.inferencer.get_model_info()
                if info:
                    return self.inferencer._build_fb_summary(info)
            return self.inferencer.get_model_summary_text()
        return "未加载模型"

    # ==================== 数据导入 ====================

    @pyqtSlot(str)
    def set_data_path(self, path: str):
        """设置训练数据路径"""
        if not path:
            return
        self.training_cfg.data_path = path
        self.data_path_changed.emit(path)

        from core.dataset import TrainDataset
        try:
            dataset = TrainDataset(path, self.training_cfg.image_size)
            self.data_count_changed.emit(len(dataset))
            self.status_message.emit(f"已加载 {len(dataset)} 个OK样本")
        except Exception as e:
            self.error_occurred.emit(f"加载数据失败: {str(e)}")

    @pyqtSlot(str)
    def set_inference_path(self, path: str):
        """设置推理数据路径"""
        if not path:
            return
        self.inference_path_changed.emit(path)

        from core.dataset import InferenceDataset
        try:
            dataset = InferenceDataset(path, self.training_cfg.image_size)
            self._inference_image_paths = dataset.image_paths
            self.inference_image_list.emit(
                [(os.path.basename(p), p) for p in dataset.image_paths]
            )
            self.status_message.emit(f"已加载 {len(dataset)} 张待检测图像")
        except Exception as e:
            self.error_occurred.emit(f"加载推理数据失败: {str(e)}")

    # ==================== 参数管理 ====================

    def get_current_params(self) -> dict:
        """获取当前所有训练参数"""
        return self.training_cfg.to_dict()

    @pyqtSlot(dict)
    def update_params(self, params: dict):
        """更新训练参数"""
        self.training_cfg.from_dict(params)
        logger.info(f"参数已更新: {params}")
        self.status_message.emit("参数已更新")

    def save_params(self, file_path: str = None):
        """训练完成后自动保存参数"""
        path = file_path or self._param_file
        try:
            self.training_cfg.save(path)
        except Exception as e:
            logger.warning(f"参数保存失败（不影响使用）: {str(e)}")

    # ==================== 训练控制 ====================

    @pyqtSlot()
    def start_training(self):
        """一键启动训练（FeatureBank特征库构建）"""
        if self.training_thread and self.training_thread.isRunning():
            self.error_occurred.emit("训练正在进行中")
            return

        if not self.training_cfg.data_path:
            self.error_occurred.emit("请先选择训练数据文件夹")
            return

        # 创建训练线程
        self.training_thread = TrainingThread(self.training_cfg)

        # 绑定信号
        self.training_thread.epoch_finished.connect(self._on_epoch_finished)
        self.training_thread.loss_updated.connect(self.training_losses.emit)
        self.training_thread.status_updated.connect(self.training_status.emit)
        self.training_thread.training_finished.connect(self._on_training_finished)
        self.training_thread.error_occurred.connect(self.error_occurred.emit)
        self.training_thread.training_history.connect(self.operation_history_updated.emit)

        # 初始化
        if not self.training_thread.setup():
            return

        # 启动
        self.training_thread.start()
        self.training_started.emit()
        self.status_message.emit("特征库构建已启动")
        self.op_history.add_record("训练启动", "成功", "FeatureBank特征库构建")

    @pyqtSlot()
    def pause_training(self):
        """暂停训练"""
        if self.training_thread:
            self.training_thread.pause()
            self.training_paused.emit()
            self.status_message.emit("训练已暂停")

    @pyqtSlot()
    def resume_training(self):
        """继续训练"""
        if self.training_thread:
            self.training_thread.resume()
            self.training_resumed.emit()
            self.status_message.emit("训练已继续")

    @pyqtSlot()
    def stop_training(self):
        """终止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.quit()
            self.training_thread.wait(3000)
            self.training_stopped.emit()
            self.status_message.emit("训练已终止")
            self.op_history.add_record("训练控制", "终止", "用户手动终止")

    def _on_epoch_finished(self, epoch: int, total: int, loss: float):
        """每个epoch完成时的回调"""
        self.training_progress.emit(epoch, total, loss)
        if self.training_thread and self.training_thread.trainer:
            avg_time = sum(self.training_thread.trainer.epoch_times) / \
                       max(len(self.training_thread.trainer.epoch_times), 1)
            elapsed = epoch * avg_time
            self.training_time.emit(elapsed)

    def _on_training_finished(self, success: bool, message: str):
        """训练完成时的回调"""
        self.training_finished.emit(success, message)
        if success:
            self.save_params()
            # 训练完成后自动初始化推理引擎
            self.init_inferencer()
            # 刷新模型列表
            self.list_saved_models()
        self.status_message.emit(message)

    # ==================== 推理控制 ====================

    @pyqtSlot()
    def init_inferencer(self):
        """初始化推理引擎（查找并加载最新模型）"""
        # 优先加载 FeatureBank 模型
        model_path = os.path.join(
            self.training_cfg.checkpoint_dir, "model_deploy.pth"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                self.training_cfg.checkpoint_dir, "sup_simple_net_best.pth"
            )
        if not os.path.exists(model_path):
            self.error_occurred.emit("未找到训练好的模型文件，请先训练或手动加载模型")
            return False

        try:
            if self.inferencer is None:
                self.inferencer = Inferencer(self.training_cfg)
            result = self.inferencer.load_model_with_validation(model_path)
            self.model_load_finished.emit(result)
            if result.success:
                mode = "FeatureBank" if self.inferencer._fb_mode else "SuperSimpleNet"
                self.status_message.emit(f"推理引擎已加载 [{mode}]: {model_path}")
            return result.success
        except Exception as e:
            self.error_occurred.emit(f"推理引擎初始化失败: {str(e)}")
            return False

    @pyqtSlot(str)
    def infer_single_image(self, image_path: str):
        """对单张图像进行推理"""
        if self.inferencer is None or not self.inferencer._is_loaded:
            if not self.init_inferencer():
                return

        try:
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize(self.training_cfg.image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            is_anomaly, score, time_ms = self.inferencer.infer_single(image_tensor)
            result = InferenceResult(
                file_name=os.path.basename(image_path),
                file_path=image_path,
                is_anomaly=is_anomaly,
                anomaly_score=score,
                inference_time_ms=time_ms,
            )
            self.inference_result.emit(result)
            logger.info(str(result))

        except Exception as e:
            self.error_occurred.emit(f"推理失败: {str(e)}")

    @pyqtSlot()
    def infer_batch_images(self):
        """批量推理已加载的文件夹"""
        if not self._inference_image_paths:
            self.error_occurred.emit("请先选择推理数据文件夹")
            return

        if self.inferencer is None or not self.inferencer._is_loaded:
            if not self.init_inferencer():
                return

        try:
            self.status_message.emit("批量推理进行中...")
            results = self.inferencer.infer_batch(
                os.path.dirname(self._inference_image_paths[0])
            )
            self._inference_results = results

            ok_count = sum(1 for r in results if not r.is_anomaly)
            ng_count = len(results) - ok_count
            self.inference_batch_done.emit(ok_count, ng_count)
            self.status_message.emit(
                f"批量推理完成 | OK: {ok_count} | NG: {ng_count}"
            )

            # 更新操作历史
            history = self.inferencer.get_operation_history()
            self.operation_history_updated.emit(history)

        except Exception as e:
            self.error_occurred.emit(f"批量推理失败: {str(e)}")

    @pyqtSlot()
    def clear_inference_results(self):
        """清空推理结果"""
        self._inference_results = []
        self._inference_image_paths = []
        self.inference_clear.emit()
        self.status_message.emit("推理结果已清空")

    # ==================== 操作历史 ====================

    def get_operation_history(self) -> str:
        """获取操作历史文本"""
        return self.op_history.get_formatted_history()
