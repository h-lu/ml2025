import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def set_chinese_font():
    """
    设置matplotlib中文字体
    """
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    elif system == 'Linux':
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号 