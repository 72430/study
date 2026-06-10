"""
SuperSimpleNet 工业无监督缺陷检测系统 - 主入口

启动命令:
    python main.py

环境要求:
    pip install -r requirements.txt
"""
import sys
import os

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow
from utils.logger import LoggerFactory, get_logger


def setup_environment():
    """初始化运行环境"""
    # 创建必要的目录
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "config"), exist_ok=True)

    # 初始化日志系统
    LoggerFactory.get_logger(
        name="SuperSimpleNet",
        log_dir=os.path.join(project_root, "logs"),
        level="DEBUG",
    )

    # 清理过期日志
    LoggerFactory.clean_old_logs(
        os.path.join(project_root, "logs"), max_days=30
    )


def main():
    """系统主入口"""
    # 初始化环境
    setup_environment()
    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("SuperSimpleNet 工业无监督缺陷检测系统 v1.0")
    logger.info("=" * 60)

    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setApplicationName("SuperSimpleNet-Defect-Detection")
    app.setOrganizationName("IndustrialAI")

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    logger.info("系统启动成功")

    # 运行事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
