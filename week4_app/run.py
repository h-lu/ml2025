#!/usr/bin/env python
"""
第四周：分类算法基础（二）- 决策树与集成学习（随机森林）
交互式课件启动脚本
"""

import os
import sys
import subprocess
import shutil

def main():
    """启动Streamlit应用"""
    print("启动第四周课件：决策树与随机森林...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 打印调试信息
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {current_dir}")
    print(f"img目录绝对路径: {os.path.join(current_dir, 'img')}")
    print(f"img目录存在: {os.path.exists(os.path.join(current_dir, 'img'))}")
    if os.path.exists(os.path.join(current_dir, 'img')):
        print(f"img目录内容: {os.listdir(os.path.join(current_dir, 'img'))}")
    
    # 将img目录复制到当前工作目录中
    img_dir = os.path.join(current_dir, 'img')
    working_img_dir = os.path.join(os.getcwd(), 'img')
    
    # 如果当前工作目录中不存在img目录，则创建一个符号链接
    if not os.path.exists(working_img_dir) and os.path.exists(img_dir):
        print(f"在当前工作目录 {os.getcwd()} 创建指向 {img_dir} 的符号链接")
        try:
            # 尝试创建符号链接
            os.symlink(img_dir, working_img_dir)
            print(f"符号链接创建成功，指向: {img_dir}")
        except Exception as e:
            print(f"无法创建符号链接: {e}")
            # 如果符号链接失败，则尝试复制目录
            try:
                shutil.copytree(img_dir, working_img_dir)
                print(f"复制img目录成功: {img_dir} -> {working_img_dir}")
            except Exception as e:
                print(f"复制img目录失败: {e}")
    
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