#!/usr/bin/env python3
"""
运行Streamlit应用的主入口
用法: python run.py
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def main():
    """启动Streamlit应用"""
    # 获取当前文件的目录
    current_dir = Path(__file__).parent.absolute()
    
    # 检查是否已安装所需依赖
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import scikit_learn
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        print("正在安装依赖...")
        requirements_path = current_dir / "requirements.txt"
        if requirements_path.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        else:
            print("找不到requirements.txt文件，请确保该文件存在")
            return 1
    
    # 设置工作目录为当前脚本所在目录
    os.chdir(current_dir)
    
    # 启动Streamlit应用
    print("启动课件应用...")
    port = 8505  # 使用不同的端口避免冲突
    url = f"http://localhost:{port}"
    
    # 打开浏览器
    webbrowser.open_new(url)
    
    # 运行Streamlit
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
    subprocess.call(streamlit_cmd)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 