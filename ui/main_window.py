"""
主窗口界面 - PyQt5工业风格UI
遵循MVVM模式，所有业务逻辑委托给ViewModel
集成模型命名验证、模型浏览加载、操作历史展示
"""
import os
import cv2
import numpy as np
from typing import List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QSplitter, QScrollArea,
    QStatusBar, QMenuBar, QAction, QTabWidget, QFrame,
    QProgressBar, QTextEdit, QSizePolicy, QCheckBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QTimer
from PyQt5.QtGui import (
    QFont, QPixmap, QImage, QIcon, QPalette, QColor,
)

from ui.viewmodel import MainViewModel
from core.inferencer import InferenceResult, ModelLoadResult
from config.config import ui_cfg
from utils.logger import get_logger

logger = get_logger("main_window")


# =============================================================================
# 自定义控件
# =============================================================================

class ImagePreviewWidget(QLabel):
    """图像预览控件 - 支持缩放显示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #444466;
                border-radius: 4px;
                background-color: #1a1a2e;
            }
        """)
        self.setText("图像预览区域")
        self.setScaledContents(False)
        self._pixmap: Optional[QPixmap] = None

    def display_image(self, image_path: str, max_size: int = 480):
        if not os.path.exists(image_path):
            self.setText("图像文件不存在")
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.setText("无法加载图像")
            return
        scaled = pixmap.scaled(
            max_size, max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._pixmap = pixmap
        self.setPixmap(scaled)

    def clear_display(self):
        self._pixmap = None
        self.clear()
        self.setText("图像预览区域")


class LossCurveWidget(QLabel):
    """损失曲线绘制控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 180)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #444466;
                border-radius: 4px;
                background-color: #1a1a2e;
            }
        """)
        self._losses: List[float] = []
        self.setText("损失曲线")

    def update_curve(self, losses: List[float]):
        self._losses = losses
        self.update()
        self._draw_curve()

    def _draw_curve(self):
        if not self._losses:
            return
        w, h = self.width() - 8, self.height() - 8
        if w <= 0 or h <= 0:
            return
        img = np.ones((h, w, 3), dtype=np.uint8) * 26
        if len(self._losses) < 2:
            self._draw_text(img, "等待更多数据...")
            self._set_pixmap_from_array(img)
            return
        losses = self._losses
        min_loss = min(losses)
        max_loss = max(losses)
        range_loss = max_loss - min_loss if max_loss > min_loss else 1.0
        for i in range(0, w, 40):
            cv2.line(img, (i, 0), (i, h), (40, 40, 60), 1)
        for i in range(0, h, 40):
            cv2.line(img, (0, i), (w, i), (40, 40, 60), 1)
        points = []
        for i, loss in enumerate(losses):
            x = int((i / (len(losses) - 1)) * (w - 20) + 10)
            y = int(h - ((loss - min_loss) / range_loss) * (h - 20) - 10)
            points.append((x, y))
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], (0, 200, 255), 2, cv2.LINE_AA)
        if points:
            cv2.circle(img, points[-1], 4, (0, 200, 255), -1)
            cv2.putText(img, f"{losses[-1]:.6f}",
                        (points[-1][0] + 5, points[-1][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        self._set_pixmap_from_array(img)

    def _set_pixmap_from_array(self, img: np.ndarray):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def _draw_text(self, img: np.ndarray, text: str):
        h, w = img.shape[:2]
        cv2.putText(img, text, (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._losses:
            self._draw_curve()
        else:
            self.setText("损失曲线")


class ResultOverlayWidget(QFrame):
    """推理结果显示覆盖层"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 130)
        self.setStyleSheet("""
            QFrame { border: 2px solid #444466; border-radius: 8px; }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.label_result = QLabel("待检测")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setFont(QFont("Arial", 20, QFont.Bold))
        self.label_result.setStyleSheet("color: #888888;")
        layout.addWidget(self.label_result)
        self.label_score = QLabel("")
        self.label_score.setAlignment(Qt.AlignCenter)
        self.label_score.setFont(QFont("Arial", 12))
        self.label_score.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(self.label_score)
        self.label_time = QLabel("")
        self.label_time.setAlignment(Qt.AlignCenter)
        self.label_time.setFont(QFont("Arial", 10))
        self.label_time.setStyleSheet("color: #999999;")
        layout.addWidget(self.label_time)
        self.label_reason = QLabel("")
        self.label_reason.setAlignment(Qt.AlignCenter)
        self.label_reason.setFont(QFont("Arial", 10))
        self.label_reason.setStyleSheet("color: #FF8888;")
        self.label_reason.setWordWrap(True)
        layout.addWidget(self.label_reason)

    def show_result(self, result: InferenceResult):
        if result.is_anomaly:
            self.label_result.setText("⚠ NG - 异常")
            self.label_result.setStyleSheet(f"color: {ui_cfg.color_ng}; font-weight: bold;")
            # 根据得分给出分级警告
            score = result.anomaly_score
            if score >= 0.2:
                reason = "🔴 异常置信度极高，存在明显缺陷"
            elif score >= 0.1:
                reason = "🟠 异常置信度较高，可能存在细微缺陷"
            else:
                reason = "🟡 轻微异常，建议人工复核"
            self.label_reason.setText(reason)
            self.setStyleSheet("""
                QFrame { border: 3px solid #FF1744; border-radius: 8px;
                         background-color: rgba(255,23,68,0.08); }
            """)
        else:
            self.label_result.setText("✓ OK - 正常")
            self.label_result.setStyleSheet(f"color: {ui_cfg.color_ok}; font-weight: bold;")
            self.label_reason.setText("✅ 未检出异常特征")
            self.setStyleSheet("""
                QFrame { border: 3px solid #00C853; border-radius: 8px;
                         background-color: rgba(0,200,83,0.08); }
            """)
        self.label_score.setText(f"异常得分: {result.score_text}")
        self.label_time.setText(f"推理耗时: {result.inference_time_ms:.1f} ms")

    def clear_result(self):
        self.label_result.setText("待检测")
        self.label_result.setStyleSheet("color: #888888;")
        self.label_score.setText("")
        self.label_time.setText("")
        self.label_reason.setText("")
        self.setStyleSheet("QFrame { border: 2px solid #444466; border-radius: 8px; }")


# =============================================================================
# 模型管理对话框
# =============================================================================

class ModelBrowserDialog(QDialog):
    """模型浏览对话框 - 显示所有已保存模型列表及详细信息"""

    def __init__(self, viewmodel: MainViewModel, parent=None):
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.setWindowTitle("模型管理 - 浏览已保存的模型")
        self.resize(700, 500)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a2e; color: #CCCCCC; }
            QTableWidget { background-color: #1a1a2e; color: #CCCCCC;
                           border: 1px solid #444466; }
            QTableWidget::item:selected { background-color: #3333AA; }
            QHeaderView::section { background-color: #222244;
                                   color: #88AAFF; padding: 4px; }
        """)
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("已保存的模型列表")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(f"color: {ui_cfg.color_accent};")
        layout.addWidget(title)

        # 模型表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "模型名称", "骨干网络", "最佳Loss", "轮次", "训练耗时", "创建时间", "文件"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self._load_models)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_load = QPushButton("加载选中模型")
        self.btn_load.setStyleSheet("""
            QPushButton { background-color: #4CAF50; padding: 6px 20px;
                          font-weight: bold; }
        """)
        self.btn_load.clicked.connect(self._load_selected)
        btn_layout.addWidget(self.btn_load)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self._load_models()

    def _load_models(self):
        """加载模型列表到表格"""
        models = self.viewmodel.list_saved_models()
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            # 检测是否为 FeatureBank 模型
            method = model.get("method", "")
            is_fb = method == "feature_bank" or model.get("ok_sample_count") is not None

            self.table.setItem(row, 0, QTableWidgetItem(str(model.get("name", "未知"))))

            if is_fb:
                # FeatureBank 展示
                self.table.setItem(row, 1, QTableWidgetItem("特征库"))
                self.table.setItem(row, 2, QTableWidgetItem(str(model.get("backbone", "?"))))
                self.table.setItem(row, 3, QTableWidgetItem(str(model.get("ok_sample_count", "?"))))
                self.table.setItem(row, 4, QTableWidgetItem(str(model.get("embedding_dim", "?"))))
            else:
                # 传统模型展示
                self.table.setItem(row, 1, QTableWidgetItem("神经网络"))
                self.table.setItem(row, 2, QTableWidgetItem(str(model.get("backbone", "?"))))
                bl = model.get("best_loss", "?")
                if isinstance(bl, float):
                    bl = f"{bl:.6f}"
                self.table.setItem(row, 3, QTableWidgetItem(str(bl)))
                ep = model.get("current_epoch", 0)
                te = model.get("total_epochs", 0)
                self.table.setItem(row, 4, QTableWidgetItem(f"{ep}/{te}"))

            self.table.setItem(row, 5, QTableWidgetItem(str(model.get("timestamp", ""))[:19]))
            self.table.setItem(row, 6, QTableWidgetItem(model.get("_file_key", "")))
        self.table.resizeColumnsToContents()

    def _load_selected(self):
        """加载选中的模型"""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选择一个模型")
            return
        file_key = self.table.item(row, 6).text()
        file_path = os.path.join(
            self.viewmodel.training_cfg.checkpoint_dir, file_key
        )
        self.viewmodel.load_model_file(file_path)
        self.accept()


# =============================================================================
# 主窗口
# =============================================================================

class MainWindow(QMainWindow):
    """
    工业级无监督缺陷检测系统主窗口

    布局:
      - 顶部: 菜单栏
      - 中部: 左右分栏（左: 训练区 | 右: 推理区）
      - 底部: 状态栏
    """

    def __init__(self):
        super().__init__()
        self.viewmodel = MainViewModel()
        self._current_result: Optional[InferenceResult] = None

        self._init_ui()
        self._bind_viewmodel()
        self._apply_industrial_style()

        # 训练完成后延迟刷新模型列表的计时器
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._delayed_refresh_models)

        # 初始化时刷新模型列表
        QTimer.singleShot(500, self.viewmodel.refresh_model_list)

        logger.info("主窗口初始化完成")

    def _init_ui(self):
        """初始化UI布局"""
        self.setWindowTitle(ui_cfg.window_title)
        self.resize(*ui_cfg.window_size)
        self.setMinimumSize(1024, 768)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        self._create_menu_bar()

        splitter = QSplitter(Qt.Horizontal)
        left_panel = self._create_training_panel()
        splitter.addWidget(left_panel)
        right_panel = self._create_inference_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setFont(QFont("Consolas", 9))
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)

    def _create_menu_bar(self):
        menubar = self.menuBar()
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        act = QAction("打开训练数据文件夹...", self)
        act.triggered.connect(self._on_open_train_data)
        file_menu.addAction(act)
        act = QAction("打开推理数据文件夹...", self)
        act.triggered.connect(self._on_open_infer_data)
        file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("退出(&X)", self)
        act.setShortcut("Alt+F4")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # 模型菜单
        model_menu = menubar.addMenu("模型(&M)")
        act = QAction("浏览已保存的模型...", self)
        act.triggered.connect(self._open_model_browser)
        model_menu.addAction(act)
        act = QAction("加载模型文件...", self)
        act.triggered.connect(self._on_load_model)
        model_menu.addAction(act)
        act = QAction("导出部署模型...", self)
        act.triggered.connect(self._on_export_model)
        model_menu.addAction(act)
        model_menu.addSeparator()
        act = QAction("刷新模型列表", self)
        act.triggered.connect(lambda: self.viewmodel.refresh_model_list())
        model_menu.addAction(act)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        act = QAction("全屏模式", self)
        act.setShortcut("F11")
        act.setCheckable(True)
        act.triggered.connect(
            lambda checked: self.showFullScreen() if checked else self.showNormal()
        )
        view_menu.addAction(act)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        act = QAction("关于系统", self)
        act.triggered.connect(self._show_about)
        help_menu.addAction(act)

    # ==================== 训练面板 ====================

    def _create_training_panel(self) -> QWidget:
        """创建左侧训练面板（精简版 - 适配FeatureBank模式）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # 标题 + 模式标签
        header = QHBoxLayout()
        title = QLabel("⚡ 模型训练")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(f"color: {ui_cfg.color_accent}; padding: 2px;")
        header.addWidget(title)
        # 模式标签
        mode_label = QLabel("特征库模式")
        mode_label.setStyleSheet("""
            background-color: #1a4a3a; color: #66ddaa;
            border: 1px solid #2a7a5a; border-radius: 8px;
            padding: 2px 10px; font-size: 10px; font-weight: bold;
        """)
        header.addWidget(mode_label)
        header.addStretch()
        layout.addLayout(header)

        # ====== ① 数据导入（精简） ======
        data_group = QGroupBox("① 选择OK样本文件夹")
        data_layout = QHBoxLayout(data_group)
        data_layout.setContentsMargins(8, 16, 8, 6)
        self.btn_select_data = QPushButton("📂 选择文件夹")
        self.btn_select_data.setToolTip("选择仅包含OK(正常)样本的文件夹作为训练数据")
        self.btn_select_data.clicked.connect(self._on_open_train_data)
        data_layout.addWidget(self.btn_select_data)
        self.label_data_path = QLabel("未选择")
        self.label_data_path.setStyleSheet("color: #8888BB; font-size: 11px;")
        data_layout.addWidget(self.label_data_path, 1)
        self.label_data_count = QLabel("")
        self.label_data_count.setStyleSheet("color: #88BB88; font-weight: bold; font-size: 12px;")
        data_layout.addWidget(self.label_data_count)
        layout.addWidget(data_group)

        # ====== ② 骨干网络选择（关键参数） ======
        param_group = QGroupBox("② 骨干网络")
        param_layout = QHBoxLayout(param_group)
        param_layout.setContentsMargins(8, 16, 8, 6)

        param_layout.addWidget(QLabel("特征提取网络:"))
        self.combo_backbone = QComboBox()
        self.combo_backbone.addItems(
            ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"]
        )
        self.combo_backbone.setToolTip(
            "越大提取特征越丰富，但速度越慢\n"
            "resnet18: 最快，适合快速验证\n"
            "resnet50: 推荐，效果与速度平衡\n"
            "wide_resnet50_2: 最佳效果但最慢"
        )
        self.combo_backbone.setCurrentText("resnet50")
        param_layout.addWidget(self.combo_backbone, 1)

        # 简洁提示
        tip = QLabel("ⓘ 仅修改骨干网络即可，无需调参")
        tip.setStyleSheet("color: #8888AA; font-size: 10px;")
        param_layout.addWidget(tip)

        layout.addWidget(param_group)

        # ====== ③ 训练控制（精简） ======
        train_ctrl_group = QGroupBox("③ 一键训练")
        train_ctrl_layout = QVBoxLayout(train_ctrl_group)
        train_ctrl_layout.setContentsMargins(8, 16, 8, 8)
        train_ctrl_layout.setSpacing(6)

        # 主按钮行
        btn_row = QHBoxLayout()
        self.btn_start_train = QPushButton("▶ 开始训练")
        self.btn_start_train.setStyleSheet("""
            QPushButton { background-color: #1a5a3a; border: 1px solid #2a8a5a;
                          padding: 10px 24px; font-weight: bold; font-size: 14px;
                          color: #88eeaa; border-radius: 6px; }
            QPushButton:hover { background-color: #2a7a4a; border-color: #3aaa6a; }
            QPushButton:disabled { background-color: #1a2a2a; color: #446655; border-color: #2a4a3a; }
        """)
        self.btn_start_train.setToolTip("一键构建特征库（无需梯度训练，2秒完成）")
        self.btn_start_train.clicked.connect(self._on_start_training)
        btn_row.addWidget(self.btn_start_train, 2)

        # 控制按钮组
        self.btn_pause_train = QPushButton("⏸")
        self.btn_pause_train.setEnabled(False)
        self.btn_pause_train.setToolTip("暂停")
        self.btn_pause_train.clicked.connect(self._on_pause_training)
        btn_row.addWidget(self.btn_pause_train)
        self.btn_resume_train = QPushButton("▶")
        self.btn_resume_train.setEnabled(False)
        self.btn_resume_train.setToolTip("继续")
        self.btn_resume_train.clicked.connect(self._on_resume_training)
        btn_row.addWidget(self.btn_resume_train)
        self.btn_stop_train = QPushButton("⏹")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setToolTip("终止")
        self.btn_stop_train.setStyleSheet("""
            QPushButton { background-color: #5a1a1a; border: 1px solid #8a2a2a;
                          padding: 10px 14px; font-weight: bold; font-size: 14px;
                          color: #ee8888; border-radius: 6px; }
            QPushButton:hover { background-color: #7a2a2a; }
            QPushButton:disabled { background-color: #2a1a1a; color: #664444; border-color: #4a2a2a; }
        """)
        self.btn_stop_train.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self.btn_stop_train)
        train_ctrl_layout.addLayout(btn_row)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("就绪")
        train_ctrl_layout.addWidget(self.progress_bar)

        # 状态标签
        self.label_train_status = QLabel("选择OK样本文件夹后点击「开始训练」一键完成")
        self.label_train_status.setStyleSheet("color: #8888BB; font-size: 11px; padding: 2px;")
        train_ctrl_layout.addWidget(self.label_train_status)

        layout.addWidget(train_ctrl_group)

        # ====== 损失曲线（可选） ======
        curve_group = QGroupBox("训练状态")
        curve_layout = QVBoxLayout(curve_group)
        curve_layout.setContentsMargins(4, 16, 4, 4)
        # 训练结果概要
        self.label_train_result = QLabel("等待训练...")
        self.label_train_result.setStyleSheet("""
            color: #AAAAAA; font-family: Consolas; font-size: 12px;
            padding: 8px; background-color: #0a0a1a; border-radius: 4px;
            border: 1px solid #1e1e40;
        """)
        self.label_train_result.setWordWrap(True)
        self.label_train_result.setMinimumHeight(60)
        curve_layout.addWidget(self.label_train_result)

        # 损失曲线（保留但变小）
        self.loss_curve = LossCurveWidget()
        self.loss_curve.setMinimumSize(0, 100)
        self.loss_curve.setMaximumHeight(120)
        curve_layout.addWidget(self.loss_curve, 1)

        layout.addWidget(curve_group, 1)

        return panel

    # ==================== 推理面板 ====================

    def _create_inference_panel(self) -> QWidget:
        """创建右侧推理面板（优化版 - 更清晰的结果展示）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # ====== 模型管理区域（紧凑） ======
        model_mgr_group = QGroupBox("模型管理")
        mgr_layout = QVBoxLayout(model_mgr_group)
        mgr_layout.setContentsMargins(8, 16, 8, 6)
        mgr_layout.setSpacing(4)

        # 模型列表 + 操作按钮
        mgr_top = QHBoxLayout()
        self.combo_model_list = QComboBox()
        self.combo_model_list.setMinimumHeight(26)
        self.combo_model_list.setPlaceholderText("选择已保存的模型...")
        mgr_top.addWidget(self.combo_model_list, 1)

        self.btn_refresh_models = QPushButton("↻")
        self.btn_refresh_models.setToolTip("刷新模型列表")
        self.btn_refresh_models.clicked.connect(self._on_refresh_model_list)
        mgr_top.addWidget(self.btn_refresh_models)

        self.btn_browse_models = QPushButton("浏览...")
        self.btn_browse_models.clicked.connect(self._open_model_browser)
        mgr_top.addWidget(self.btn_browse_models)

        self.btn_load_selected = QPushButton("加载")
        self.btn_load_selected.setStyleSheet("""
            QPushButton { background-color: #1a5a3a; border: 1px solid #2a8a5a;
                          color: #88eeaa; font-weight: bold; }
            QPushButton:hover { background-color: #2a7a4a; }
        """)
        self.btn_load_selected.clicked.connect(self._on_load_selected_model)
        mgr_top.addWidget(self.btn_load_selected)

        mgr_layout.addLayout(mgr_top)

        # 模型信息摘要（可折叠）
        self.label_model_info = QLabel("⏳ 请加载或训练模型后查看信息")
        self.label_model_info.setStyleSheet("""
            color: #8888BB; font-family: Consolas; font-size: 11px;
            padding: 6px; background-color: #0a0a1a; border-radius: 3px;
            border: 1px solid #1e1e40;
        """)
        self.label_model_info.setWordWrap(True)
        self.label_model_info.setMaximumHeight(100)
        mgr_layout.addWidget(self.label_model_info)

        layout.addWidget(model_mgr_group)

        # ====== 缺陷检测推理 ======
        title = QLabel("🔍 缺陷检测")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(f"color: {ui_cfg.color_accent}; padding: 2px;")
        layout.addWidget(title)

        # 推理控制栏
        infer_ctrl = QHBoxLayout()
        self.btn_select_infer = QPushButton("📂 选择推理文件夹")
        self.btn_select_infer.setToolTip("选择待检测图像文件夹（会自动扫描所有子文件夹）")
        self.btn_select_infer.clicked.connect(self._on_open_infer_data)
        infer_ctrl.addWidget(self.btn_select_infer)

        self.btn_batch_infer = QPushButton("▶ 批量推理")
        self.btn_batch_infer.setStyleSheet("""
            QPushButton { background-color: #4a3a1a; border: 1px solid #8a6a2a;
                          color: #eebb66; font-weight: bold; padding: 5px 16px; }
            QPushButton:hover { background-color: #6a5a2a; }
        """)
        self.btn_batch_infer.setToolTip("对文件夹内所有图像执行批量缺陷检测")
        self.btn_batch_infer.clicked.connect(self._on_batch_infer)
        infer_ctrl.addWidget(self.btn_batch_infer)

        self.btn_clear_infer = QPushButton("清空")
        self.btn_clear_infer.clicked.connect(self._on_clear_infer)
        infer_ctrl.addWidget(self.btn_clear_infer)
        layout.addLayout(infer_ctrl)

        # 图像列表和预览（改进布局）
        preview_splitter = QSplitter(Qt.Horizontal)

        # 左侧图像列表
        list_w = QWidget()
        ll = QVBoxLayout(list_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        list_header = QLabel("图像列表 (点击选择)")
        list_header.setStyleSheet("color: #8888BB; font-size: 10px; padding: 2px;")
        ll.addWidget(list_header)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        ll.addWidget(self.image_list)
        preview_splitter.addWidget(list_w)

        # 右侧预览 + 结果
        pv_w = QWidget()
        pvl = QVBoxLayout(pv_w)
        pvl.setContentsMargins(0, 0, 0, 0)
        pvl.setSpacing(4)
        self.image_preview = ImagePreviewWidget()
        pvl.addWidget(self.image_preview, 1)
        self.result_overlay = ResultOverlayWidget()
        pvl.addWidget(self.result_overlay)
        preview_splitter.addWidget(pv_w)
        preview_splitter.setSizes([160, 460])
        layout.addWidget(preview_splitter, 1)

        # 推理统计（水平紧凑展示）
        stats_group = QGroupBox("检测结果统计")
        sl = QHBoxLayout(stats_group)
        sl.setContentsMargins(8, 14, 8, 6)

        self.label_stats_total = QLabel("总计: 0")
        self.label_stats_total.setStyleSheet("color: #AAAAAA; font-size: 14px; padding: 0 8px;")
        sl.addWidget(self.label_stats_total)

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #444466;")
        sl.addWidget(sep1)

        self.label_stats_ok = QLabel("OK: 0")
        self.label_stats_ok.setStyleSheet(f"color: {ui_cfg.color_ok}; font-size: 14px; font-weight: bold; padding: 0 8px;")
        sl.addWidget(self.label_stats_ok)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444466;")
        sl.addWidget(sep2)

        self.label_stats_ng = QLabel("NG: 0")
        self.label_stats_ng.setStyleSheet(f"color: {ui_cfg.color_ng}; font-size: 14px; font-weight: bold; padding: 0 8px;")
        sl.addWidget(self.label_stats_ng)

        sl.addStretch()

        # 合格率指示器
        self.label_pass_rate = QLabel("")
        self.label_pass_rate.setStyleSheet("font-size: 13px; font-weight: bold; padding: 0 8px;")
        sl.addWidget(self.label_pass_rate)

        layout.addWidget(stats_group)

        # ====== 操作历史 + 日志（合并紧凑） ======
        hist_group = QGroupBox("操作记录")
        hl = QVBoxLayout(hist_group)
        hl.setContentsMargins(4, 14, 4, 4)
        hl.setSpacing(2)

        self.text_history = QTextEdit()
        self.text_history.setReadOnly(True)
        self.text_history.setMaximumHeight(60)
        self.text_history.setFont(QFont("Consolas", 9))
        hl.addWidget(self.text_history)

        # 运行时日志（精简）
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(50)
        self.log_output.setFont(QFont("Consolas", 9))
        hl.addWidget(self.log_output)

        layout.addWidget(hist_group)

        # 绑定操作历史信号
        self.viewmodel.operation_history_updated.connect(
            lambda h: self.text_history.setPlainText(h)
        )

        return panel

    # ==================== ViewModel 信号绑定 ====================

    def _bind_viewmodel(self):
        """绑定ViewModel所有信号"""
        vm = self.viewmodel

        # 数据路径
        vm.data_path_changed.connect(
            lambda p: self.label_data_path.setText(
                p if len(p) < 50 else "..." + p[-47:]
            )
        )
        vm.data_count_changed.connect(
            lambda c: self.label_data_count.setText(f"样本数: {c}")
        )

        # 训练模型管理信号
        vm.model_load_finished.connect(self._on_model_load_finished)
        vm.model_list_updated.connect(self._on_model_list_updated)
        vm.training_started.connect(self._on_training_started_ui)
        vm.training_paused.connect(lambda: self._set_training_controls(paused=True))
        vm.training_resumed.connect(lambda: self._set_training_controls(paused=False, training=True))
        vm.training_stopped.connect(self._on_training_stopped_ui)
        vm.training_progress.connect(self._on_training_progress)
        vm.training_losses.connect(self.loss_curve.update_curve)
        vm.training_status.connect(self.label_train_status.setText)
        vm.training_finished.connect(self._on_training_finished_ui)
        vm.training_time.connect(lambda t: self._log_message(f"训练耗时: {t:.1f}秒"))

        # 推理
        vm.inference_image_list.connect(self._on_inference_image_list)
        vm.inference_result.connect(self._on_inference_result)
        vm.inference_batch_done.connect(self._on_batch_done)
        vm.inference_clear.connect(self._on_infer_clear)

        # 系统
        vm.status_message.connect(self._log_message)
        vm.error_occurred.connect(self._on_error)

    # ==================== 模型加载与列表 ====================

    def _on_model_load_finished(self, result: ModelLoadResult):
        """模型加载完成回调"""
        if result.success:
            self._log_message("✅ " + result.message)
            # 显示模型摘要
            if result.summary_text:
                self.label_model_info.setText(result.summary_text)
            else:
                self._update_model_info_label(result.model_info)
        else:
            self._on_error(result.message)
            self.label_model_info.setText("❌ 模型加载失败")

    def _update_model_info_label(self, info: dict):
        """更新模型信息标签"""
        if not info:
            self.label_model_info.setText("⏳ 请加载或训练模型后查看信息")
            return
        method = info.get("method", "")
        if "FeatureBank" in method or info.get("ok_sample_count", "?") != "?":
            # FeatureBank 摘要
            name = info.get("name", "FeatureBank模型")
            backbone = info.get("backbone", "?")
            ok_count = info.get("ok_sample_count", "?")
            embed_dim = info.get("embedding_dim", "?")
            th = info.get("suggested_threshold", 0.05)
            self.label_model_info.setText(
                f"📦 {name}\n"
                f"🏗 {backbone} | 📊 特征维度: {embed_dim}\n"
                f"📚 OK样本: {ok_count} | 🎯 阈值: {th:.4f}"
            )
        else:
            name = info.get("name", "未知")
            backbone = info.get("backbone", "?")
            best_loss = info.get("best_loss", "?")
            if isinstance(best_loss, float):
                best_loss = f"{best_loss:.6f}"
            epochs = f"{info.get('current_epoch', '?')}/{info.get('total_epochs', '?')}"
            device = info.get("device", "?")
            self.label_model_info.setText(
                f"模型: {name} | 骨干: {backbone}\n"
                f"Loss: {best_loss} | 轮次: {epochs}\n"
                f"设备: {device}"
            )

    def _on_model_list_updated(self, models: list):
        """模型列表更新回调"""
        self.combo_model_list.clear()
        if not models:
            self.combo_model_list.addItem("（无已保存的模型）")
            return
        for m in models:
            name = m.get("name", "未知")
            timestamp = str(m.get("timestamp", ""))[:16]
            self.combo_model_list.addItem(f"{name}  [{timestamp}]")
        # 保存模型文件路径映射
        self._model_file_map = {
            i: os.path.join(
                self.viewmodel.training_cfg.checkpoint_dir,
                m.get("_file_key", "")
            )
            for i, m in enumerate(models)
        }

    def _on_refresh_model_list(self):
        """刷新模型列表"""
        self.viewmodel.refresh_model_list()

    def _on_load_selected_model(self):
        """加载选中的模型"""
        idx = self.combo_model_list.currentIndex()
        if idx < 0 or not hasattr(self, '_model_file_map') or idx not in self._model_file_map:
            QMessageBox.warning(self, "提示", "请选择一个有效的模型")
            return
        file_path = self._model_file_map[idx]
        if os.path.exists(file_path):
            self.viewmodel.load_model_file(file_path)
        else:
            self._on_error(f"模型文件不存在: {file_path}")

    def _open_model_browser(self):
        """打开模型浏览器对话框"""
        dialog = ModelBrowserDialog(self.viewmodel, self)
        dialog.exec_()
        # 刷新下拉列表
        self.viewmodel.refresh_model_list()

    # ==================== 菜单操作 ====================

    def _on_open_train_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练数据文件夹（仅OK样本）")
        if path:
            self.viewmodel.set_data_path(path)

    def _on_open_infer_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择待检测图像文件夹")
        if path:
            self.viewmodel.set_inference_path(path)

    def _on_start_training(self):
        """一键启动训练（FeatureBank 特征库构建）"""
        # 同步骨干网络参数到配置
        params = self.viewmodel.get_current_params()
        params["backbone"] = self.combo_backbone.currentText()
        self.viewmodel.update_params(params)

        if not self.viewmodel.training_cfg.data_path:
            QMessageBox.warning(self, "提示", "请先选择训练数据文件夹 (datas/ok)")
            return
        self.viewmodel.start_training()

    def _on_pause_training(self):
        self.viewmodel.pause_training()

    def _on_resume_training(self):
        self.viewmodel.resume_training()

    def _on_stop_training(self):
        reply = QMessageBox.question(
            self, "确认终止",
            "确定要终止当前训练吗？最佳模型已自动保存。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.viewmodel.stop_training()

    # ==================== 训练UI状态 ====================

    def _on_training_started_ui(self):
        self._set_training_controls(training=True)
        self.progress_bar.setValue(0)
        self.label_train_result.setText("⏳ 正在构建特征库...")
        self._log_message("特征库构建已启动")

    def _on_training_progress(self, epoch: int, total: int, loss: float):
        progress = int(epoch / total * 100)
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"进度: {epoch}/{total}")
        self.label_train_status.setText(f"特征库构建中... ({progress}%)")

    def _on_training_finished_ui(self, success: bool, message: str):
        self._set_training_controls(training=False)
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("✅ 完成")
            # 解析训练结果
            if "样本数:" in message:
                self.label_train_result.setText(message)
            else:
                self.label_train_result.setText(f"✅ {message}")
        else:
            self.progress_bar.setFormat("❌ 失败")
            self.label_train_result.setText(f"❌ {message}")
        self._log_message(message)
        # 延迟刷新模型列表（等待文件写入完成）
        self._refresh_timer.start(2000)

    def _delayed_refresh_models(self):
        self.viewmodel.refresh_model_list()

    def _on_training_stopped_ui(self):
        self._set_training_controls(training=False)
        self._log_message("训练已终止")

    def _set_training_controls(self, training: bool = False, paused: bool = False):
        self.btn_start_train.setEnabled(not training)
        self.btn_pause_train.setEnabled(training and not paused)
        self.btn_resume_train.setEnabled(training and paused)
        self.btn_stop_train.setEnabled(training)
        self.btn_select_data.setEnabled(not training)
        self.combo_backbone.setEnabled(not training)
        if training:
            state_text = "特征库构建中..." if not paused else "已暂停"
            self.label_train_status.setText(state_text)
            if not paused:
                self.progress_bar.setFormat("构建中...")
        else:
            self.label_train_status.setText("就绪")
            self.progress_bar.setFormat("就绪")

    # ==================== 推理UI ====================

    def _on_inference_image_list(self, image_list: list):
        self.image_list.clear()
        for file_name, file_path in image_list:
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, file_path)
            self.image_list.addItem(item)
        self.label_stats_total.setText(f"总计: {len(image_list)}")

    def _on_image_selected(self, row: int):
        if row < 0:
            return
        item = self.image_list.item(row)
        if not item:
            return
        image_path = item.data(Qt.UserRole)
        if image_path:
            self.image_preview.display_image(image_path)
            self.viewmodel.infer_single_image(image_path)

    def _on_inference_result(self, result: InferenceResult):
        self._current_result = result
        self.result_overlay.show_result(result)
        self._log_message(str(result))

    def _on_batch_infer(self):
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先选择推理数据文件夹")
            return
        self.viewmodel.infer_batch_images()

    def _on_clear_infer(self):
        self.viewmodel.clear_inference_results()

    def _on_infer_clear(self):
        self.image_list.clear()
        self.image_preview.clear_display()
        self.result_overlay.clear_result()
        self.label_stats_ok.setText("OK: 0")
        self.label_stats_ng.setText("NG: 0")
        self.label_stats_total.setText("总计: 0")
        self.label_pass_rate.setText("")

    def _on_batch_done(self, ok_count: int, ng_count: int):
        self.label_stats_ok.setText(f"OK: {ok_count}")
        self.label_stats_ng.setText(f"NG: {ng_count}")
        total = ok_count + ng_count
        self.label_stats_total.setText(f"总计: {total}")
        if total > 0:
            rate = ok_count / total * 100
            color = "#66ddaa" if rate >= 95 else "#eebb66" if rate >= 80 else "#ee8888"
            self.label_pass_rate.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")
            self.label_pass_rate.setText(f"合格率: {rate:.1f}%")

    # ==================== 模型菜单操作 ====================

    def _on_load_model(self):
        """手动加载模型文件"""
        path = QFileDialog.getOpenFileName(
            self, "加载模型文件",
            "checkpoints",
            "Model Files (*.pth);;All Files (*)",
        )[0]
        if path:
            self.viewmodel.load_model_file(path)

    def _on_export_model(self):
        """导出部署模型"""
        save_path = QFileDialog.getSaveFileName(
            self, "导出部署模型",
            "deploy/model_deploy.pth",
            "Model Files (*.pth)",
        )[0]
        if save_path:
            import shutil
            src = os.path.join(
                self.viewmodel.training_cfg.checkpoint_dir, "model_deploy.pth"
            )
            if os.path.exists(src):
                shutil.copy2(src, save_path)
                self._log_message(f"模型已导出: {save_path}")
            else:
                QMessageBox.warning(self, "导出失败", "未找到部署模型文件，请先训练")

    def _show_about(self):
        QMessageBox.about(
            self, "关于 SuperSimpleNet",
            "SuperSimpleNet 工业无监督缺陷检测系统 v1.0\n\n"
            "检测引擎: FeatureBank (特征库 + kNN)\n"
            "骨干网络: ResNet50 (预训练, 冻结)\n"
            "特征维度: 2048\n\n"
            "核心优势:\n"
            "  • 无需标注数据, 仅需OK样本\n"
            "  • 无需传统训练, 2秒构建特征库\n"
            "  • AUROC=0.9995, NG检出率100%\n"
            "  • 单张推理约5ms (GPU)\n\n"
            "技术栈: PyTorch + PyQt5 + OpenCV"
        )

    def _on_error(self, message: str):
        self._log_message(f"[错误] {message}")
        QMessageBox.critical(self, "错误", message)

    def _log_message(self, message: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{ts}] {message}")
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.status_label.setText(message)

    def _apply_industrial_style(self):
        """应用工业风格样式表（优化版 - 更高对比度、更清晰的信息层级）"""
        self.setStyleSheet("""
            /* 全局 */
            QMainWindow { background-color: #0d0d1a; }
            QWidget { background-color: #13132a; color: #E0E0E0;
                      font-family: "Microsoft YaHei", "SimHei", "Segoe UI", Arial, sans-serif;
                      font-size: 12px; }
            /* GroupBox - 卡片式设计 */
            QGroupBox {
                border: 1px solid #2a2a5a; border-radius: 6px;
                margin-top: 14px; padding-top: 18px;
                font-weight: bold; color: #7090e0; font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 8px;
                background-color: #13132a;
            }
            /* 按钮 - 统一风格 */
            QPushButton {
                background-color: #1e1e45; border: 1px solid #3a3a7a;
                border-radius: 4px; padding: 5px 14px; color: #D0D0F0;
                min-height: 26px; font-size: 12px;
            }
            QPushButton:hover { background-color: #2e2e65; border-color: #5a5aaa; }
            QPushButton:pressed { background-color: #3e3e85; }
            QPushButton:disabled { background-color: #181830; color: #555566; border-color: #2a2a4a; }
            /* 强调按钮 */
            QPushButton[accent="true"] { background-color: #1a4a8a; border-color: #3a8aff; }
            QPushButton[accent="true"]:hover { background-color: #2a5aaa; }
            /* 输入控件 */
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #1a1a3a; border: 1px solid #3a3a6a;
                border-radius: 3px; padding: 3px 6px; color: #E0E0F0;
                min-height: 24px; font-size: 12px;
            }
            QLineEdit:focus { border-color: #4a8aff; }
            QComboBox::drop-down { background-color: #252550; border: none; width: 20px; }
            QComboBox::down-arrow { image: none; }
            QComboBox QAbstractItemView {
                background-color: #1a1a3a; color: #E0E0F0;
                border: 1px solid #3a3a6a; selection-background-color: #2a2a6a;
            }
            /* 列表 */
            QListWidget {
                background-color: #0e0e20; border: 1px solid #2a2a5a;
                border-radius: 4px; color: #D0D0E0; font-size: 12px;
                outline: none;
            }
            QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #1a1a3a; }
            QListWidget::item:selected { background-color: #2a2a7a; color: #FFFFFF; }
            QListWidget::item:hover { background-color: #1e1e50; }
            /* 进度条 */
            QProgressBar {
                border: 1px solid #2a2a5a; border-radius: 3px;
                text-align: center; color: #FFFFFF;
                background-color: #0e0e20; height: 22px; font-size: 11px;
            }
            QProgressBar::chunk { background-color: #2a6aff; border-radius: 2px; }
            /* 分割器 */
            QSplitter::handle { background-color: #2a2a5a; width: 2px; }
            /* 菜单 */
            QMenuBar { background-color: #0a0a18; color: #C0C0E0;
                       border-bottom: 1px solid #2a2a5a; padding: 2px; font-size: 13px; }
            QMenuBar::item:selected { background-color: #2a2a5a; border-radius: 3px; }
            QMenu { background-color: #12122a; color: #D0D0E0;
                    border: 1px solid #3a3a6a; padding: 4px; }
            QMenu::item { padding: 6px 24px; border-radius: 3px; }
            QMenu::item:selected { background-color: #2a2a7a; }
            QMenu::separator { height: 1px; background-color: #3a3a6a; margin: 4px 8px; }
            /* 状态栏 */
            QStatusBar { background-color: #0a0a18; color: #8888AA;
                         border-top: 1px solid #2a2a5a; font-size: 11px; }
            /* 滚动条 */
            QScrollBar:vertical {
                background-color: #0e0e20; width: 8px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3a6a; border-radius: 4px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background-color: #5a5a9a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background-color: #0e0e20; height: 8px; }
            QScrollBar::handle:horizontal { background-color: #3a3a6a; border-radius: 4px; min-width: 30px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            /* 表格 */
            QTableWidget { background-color: #0e0e20; color: #D0D0E0;
                           border: 1px solid #2a2a5a; gridline-color: #1e1e40; }
            QTableWidget::item:selected { background-color: #2a2a7a; }
            QHeaderView::section { background-color: #1a1a3a; color: #7090d0;
                                   padding: 4px; border: none; border-bottom: 1px solid #2a2a5a; font-weight: bold; }
            QTableWidget::item { padding: 3px; }
            /* 文本编辑 */
            QTextEdit { background-color: #0a0a1a; color: #C0C0E0;
                        border: 1px solid #2a2a5a; font-size: 11px; }
            /* 标签页 */
            QTabWidget::pane { border: 1px solid #2a2a5a; background-color: #13132a; }
            QTabBar::tab { background-color: #1a1a3a; color: #8888BB; padding: 6px 14px;
                           border: 1px solid #2a2a5a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #2a2a6a; color: #FFFFFF; }
            /* ToolTip */
            QToolTip { background-color: #222255; color: #FFFFFF;
                       border: 1px solid #4444AA; padding: 6px; font-size: 11px; }
        """)

    def closeEvent(self, event):
        if self.viewmodel.training_thread and self.viewmodel.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "训练正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.viewmodel.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
