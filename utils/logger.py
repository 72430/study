"""
日志系统 - 分级日志记录，自动按日期分割，支持容量控制
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path


class LoggerFactory:
    """日志工厂，统一管理全局日志实例"""

    _instances = {}

    @classmethod
    def get_logger(
        cls,
        name: str = "SuperSimpleNet",
        log_dir: str = None,
        level: str = "DEBUG",
        max_bytes: int = 10 * 1024 * 1024,  # 每个日志文件最大10MB
        backup_count: int = 30,              # 保留最近30个文件
    ) -> logging.Logger:
        """
        获取或创建指定名称的日志器

        Args:
            name: 日志器名称，通常使用模块名
            log_dir: 日志文件保存目录，默认位于项目根目录的logs文件夹
            level: 日志级别 DEBUG/INFO/WARNING/ERROR/CRITICAL
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的日志文件数量

        Returns:
            logging.Logger 实例
        """
        if name in cls._instances:
            return cls._instances[name]

        # 创建日志器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        logger.handlers.clear()

        # 日志格式: 时间戳 | 模块名 | 级别 | 操作类型 | 状态信息 | 异常详情
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ---- 控制台输出 ----
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ---- 文件输出（按日志级别分文件） ----
        if log_dir is None:
            log_dir = str(
                Path(__file__).resolve().parent.parent / "logs"
            )

        os.makedirs(log_dir, exist_ok=True)

        # 所有日志写入主文件
        main_handler = RotatingFileHandler(
            filename=os.path.join(
                log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log"
            ),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(formatter)
        logger.addHandler(main_handler)

        # ERROR及以上级别单独写入错误文件
        error_handler = RotatingFileHandler(
            filename=os.path.join(
                log_dir, f"error_{datetime.now().strftime('%Y-%m-%d')}.log"
            ),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        cls._instances[name] = logger
        return logger

    @classmethod
    def clean_old_logs(cls, log_dir: str, max_days: int = 30):
        """
        清理超过指定天数的日志文件

        Args:
            log_dir: 日志目录
            max_days: 保留的最大天数
        """
        if not os.path.exists(log_dir):
            return

        now = datetime.now()
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            if os.path.isfile(file_path):
                # 解析文件名中的日期
                try:
                    # 尝试从文件名提取日期
                    date_str = file_name.split("_")[0].split(".")[0]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if (now - file_date).days > max_days:
                        os.remove(file_path)
                except (ValueError, IndexError):
                    continue


# 便捷访问接口
def get_logger(name: str = "SuperSimpleNet") -> logging.Logger:
    """获取日志器的快捷方式"""
    return LoggerFactory.get_logger(name=name)
