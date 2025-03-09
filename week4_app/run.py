#!/usr/bin/env python
"""
第四周：分类算法基础（二）- 决策树与集成学习（随机森林）
交互式课件启动脚本
"""

import os
import sys
import subprocess

def main():
    """启动Streamlit应用"""
    print("启动第四周课件：决策树与随机森林...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行应用
    cmd = ["streamlit", "run", os.path.join(current_dir, "app.py")]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n应用已关闭")
    except Exception as e:
        print(f"启动应用时出错：{e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 