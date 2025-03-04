import os
import sys
import subprocess

def main():
    """运行Streamlit应用程序"""
    # 获取当前文件目录
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # 确保当前目录在sys.path中
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    
    # 设置Streamlit应用的路径
    filename = os.path.join(dir_path, "app.py")
    
    # 运行Streamlit应用
    print("启动机器学习应用程序...")
    print(f"应用路径: {filename}")
    print("打开浏览器访问: http://localhost:8501/")
    
    # 使用subprocess运行streamlit命令
    cmd = ["streamlit", "run", filename, "--server.port=8501"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 