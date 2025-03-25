import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import os
import numpy as np
from cycler import cycler
import base64
from io import BytesIO
import locale
import matplotlib.font_manager as fm

def configure_matplotlib_fonts():
    """
    根据操作系统配置matplotlib的中文字体支持
    """
    system = platform.system()
    
    # 配置颜色循环
    plt.rcParams['axes.prop_cycle'] = cycler(color=[
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # 设置全局字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 支持SVG输出
    plt.rcParams['svg.fonttype'] = 'none'
    
    # 获取系统区域设置
    current_locale = locale.getlocale()
    is_chinese = current_locale[0] and ('zh_' in current_locale[0] or 'CN' in current_locale[0])
    
    # 获取系统可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 根据不同系统设置中文字体
    if system == 'Windows':
        # Windows系统
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        # macOS系统常见中文字体
        font_candidates = ['PingFang HK', 'PingFang SC', 'PingFang TC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB', 'Apple LiGothic']
    else:  # Linux和其他系统
        # Linux系统
        font_candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 选择第一个可用的字体
    found_font = False
    for font in font_candidates:
        if font in available_fonts:
            print(f"使用中文字体: {font}")
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
            found_font = True
            break
    
    if not found_font:
        print("未找到合适的中文字体，将使用默认字体")
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica'] + plt.rcParams.get('font.sans-serif', [])

def create_figure(figsize=(10, 6), dpi=100):
    """
    创建具有正确中文支持的matplotlib图形
    
    参数:
    figsize: 图形大小元组(宽, 高)，单位为英寸
    dpi: 分辨率
    
    返回:
    fig, ax: 图形和轴对象
    """
    # 确保中文配置已设置
    configure_matplotlib_fonts()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax

def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    保存图形为多种格式
    
    参数:
    fig: matplotlib图形对象
    filename: 输出文件名(不含扩展名)
    dpi: 输出分辨率
    bbox_inches: 边界框设置
    """
    # 保存为PNG
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches=bbox_inches)
    
    # 保存为SVG (矢量格式)
    fig.savefig(f"{filename}.svg", bbox_inches=bbox_inches)

def create_ensemble_learning_svg():
    """
    创建集成学习示意图的SVG代码
    
    返回:
    svg_code: SVG代码字符串
    """
    # 从文件中读取SVG
    svg_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ensemble_learning.svg')
    
    try:
        with open(svg_file_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()
            return svg_code
    except Exception as e:
        print(f"无法读取SVG文件: {e}")
        # 提供一个简单的备用HTML
        return f'''
        <div style="width:400px; height:320px; border:1px solid #ddd; border-radius:5px; display:flex; flex-direction:column; justify-content:center; align-items:center; background-color:#f8f9fa; font-family:'PingFang HK', 'Microsoft YaHei', sans-serif;">
            <div style="font-size:18px; color:#1E88E5; margin-bottom:20px;">集成学习流程图</div>
            <div style="color:#666; text-align:center; padding:0 20px;">
                集成多个模型的预测结果,<br>提高整体性能和泛化能力
            </div>
        </div>
        '''

def fig_to_base64(fig):
    """
    将matplotlib图形转换为base64编码，可直接嵌入HTML
    
    参数:
    fig: matplotlib图形对象
    
    返回:
    base64编码的图像字符串
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

def create_chinese_text_image(text, fontsize=14, width=300, height=50, bg_color='white', text_color='black'):
    """
    创建包含中文文本的图像，避开matplotlib中文显示问题
    
    参数:
    text: 要显示的中文文本
    fontsize: 字体大小
    width, height: 图像尺寸
    bg_color: 背景颜色
    text_color: 文本颜色
    
    返回:
    HTML img标签字符串，包含base64编码的图像
    """
    # 创建一个简单的HTML元素而不是使用matplotlib
    return f'<div style="width:{width}px; height:{height}px; display:flex; align-items:center; justify-content:center; background-color:{bg_color}; color:{text_color}; font-size:{fontsize}px; font-weight:bold; text-align:center;">{text}</div>'

def create_labeled_figure(title, xlabel, ylabel, figsize=(8, 5)):
    """
    创建带有中文标签的图形，处理中文显示问题
    
    参数:
    title: 图形标题
    xlabel: x轴标签
    ylabel: y轴标签
    figsize: 图形尺寸
    
    返回:
    fig, ax: matplotlib图形和轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用英文占位符
    ax.set_title("_title_")
    ax.set_xlabel("_xlabel_")
    ax.set_ylabel("_ylabel_")
    
    # 创建原始图形
    return fig, ax

def add_chinese_labels(fig, ax, title=None, xlabel=None, ylabel=None):
    """
    在图形上添加中文标签图像
    
    参数:
    fig: matplotlib图形对象
    ax: matplotlib轴对象
    title, xlabel, ylabel: 中文标签文本
    
    返回:
    base64编码的图像字符串
    """
    # 清除原始标签
    if title:
        ax.set_title("")
    if xlabel:
        ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel("")
    
    # 转换为base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    # 创建带有标签的HTML
    html = f'<div style="position:relative; display:inline-block;">'
    html += f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
    
    if title:
        html += f'<div style="position:absolute; top:10px; left:0; width:100%; text-align:center; font-weight:bold;">{title}</div>'
    
    if xlabel:
        html += f'<div style="position:absolute; bottom:5px; left:0; width:100%; text-align:center;">{xlabel}</div>'
    
    if ylabel:
        html += f'<div style="position:absolute; top:50%; left:-40px; transform:rotate(-90deg); transform-origin:right center;">{ylabel}</div>'
    
    html += '</div>'
    
    return html

def get_chinese_plot(plot_function, title, xlabel=None, ylabel=None, figsize=(10, 6)):
    """
    生成带有中文标签的图表并返回HTML代码
    
    参数:
    plot_function: 接受ax参数的绘图函数
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    figsize: 图表尺寸
    
    返回:
    HTML代码字符串
    """
    # 确保字体配置已正确设置
    configure_matplotlib_fonts()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 执行绘图函数
    plot_function(ax)
    
    # 转换为base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    # 创建带有中文标签的HTML
    font_family = "'PingFang HK', 'PingFang SC', 'Microsoft YaHei', sans-serif"
    
    html = f'''
    <div style="text-align:center; margin-bottom:20px; font-family:{font_family};">
        <img src="data:image/png;base64,{img_str}" style="max-width:100%;">
        <div style="font-weight:bold; font-size:16px; margin-top:5px; font-family:{font_family};">{title}</div>
    '''
    
    if xlabel or ylabel:
        html += f'<div style="display:flex; justify-content:center; margin-top:5px; font-family:{font_family};">'
        if xlabel:
            html += f'<span style="margin-right:15px;"><b>X轴</b>: {xlabel}</span>'
        if ylabel:
            html += f'<span><b>Y轴</b>: {ylabel}</span>'
        html += '</div>'
    
    html += '</div>'
    
    return html 