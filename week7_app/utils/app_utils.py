import streamlit as st
import base64
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

def create_custom_header(title, subtitle=None, icon=None):
    """
    创建自定义标题区域
    
    参数:
        title: 主标题
        subtitle: 副标题 (可选)
        icon: 图标字符 (可选)
    """
    if icon:
        st.markdown(f"# {icon} {title}")
    else:
        st.markdown(f"# {title}")
        
    if subtitle:
        st.markdown(f"*{subtitle}*")
    
    st.markdown("---")

def create_info_box(text, type="info"):
    """
    创建信息提示框
    
    参数:
        text: 提示文本
        type: 提示类型 ("info", "success", "warning", "error")
    """
    if type == "info":
        st.info(text)
    elif type == "success":
        st.success(text)
    elif type == "warning":
        st.warning(text)
    elif type == "error":
        st.error(text)

def create_expander(title, content, expanded=False):
    """
    创建可折叠的内容区域
    
    参数:
        title: 标题
        content: 内容
        expanded: 是否默认展开
    """
    with st.expander(title, expanded=expanded):
        st.markdown(content)

def create_tabs(tab_titles, contents):
    """
    创建标签页
    
    参数:
        tab_titles: 标签页标题列表
        contents: 标签页内容函数列表，每个函数接收一个st.tab参数
    """
    tabs = st.tabs(tab_titles)
    for i, tab in enumerate(tabs):
        with tab:
            contents[i](tab)

def display_code(code, language="python"):
    """
    显示代码块
    
    参数:
        code: 代码字符串
        language: 代码语言
    """
    st.code(code, language=language)

def create_quiz(question, options, correct_index, explanation):
    """
    创建互动问答
    
    参数:
        question: 问题
        options: 选项列表
        correct_index: 正确答案的索引
        explanation: 解释
    """
    st.markdown(f"**问题:** {question}")
    
    answer = st.radio("请选择:", options, key=question)
    
    if st.button("提交答案", key=f"btn_{question}"):
        if answer == options[correct_index]:
            st.success("回答正确! 👍")
        else:
            st.error("回答错误 😕")
        
        st.markdown(f"**解释:** {explanation}")

def plot_step_by_step_controls(step, max_steps):
    """
    创建步骤控制器
    
    参数:
        step: 当前步骤 (从1开始)
        max_steps: 最大步骤数
    
    返回:
        next_step: 如果点击"下一步"按钮，则为True
        prev_step: 如果点击"上一步"按钮，则为True
    """
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        prev_step = st.button("上一步", disabled=step <= 1)
    
    with col2:
        st.progress((step - 1) / (max_steps - 1))
        st.text(f"步骤 {step} / {max_steps}")
    
    with col3:
        next_step = st.button("下一步", disabled=step >= max_steps)
    
    return next_step, prev_step

def get_table_download_link(df, filename="data.csv", text="下载数据"):
    """
    获取DataFrame的下载链接
    
    参数:
        df: pandas DataFrame
        filename: 下载的文件名
        text: 链接显示的文本
        
    返回:
        下载链接的HTML字符串
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_figure_as_image(fig, dpi=100):
    """
    将Matplotlib图形转换为图像
    
    参数:
        fig: Matplotlib图形
        dpi: 分辨率
        
    返回:
        PNG图像的字节数据
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def render_latex(latex_str):
    """
    渲染LaTeX公式
    
    参数:
        latex_str: LaTeX公式字符串
    """
    st.latex(latex_str)

def show_algorithm_steps(steps, step_descriptions):
    """
    显示算法步骤
    
    参数:
        steps: 步骤标题列表
        step_descriptions: 步骤描述字典，键为步骤标题
    """
    step = st.selectbox("选择步骤:", steps)
    
    st.markdown(f"### {step}")
    st.markdown(step_descriptions[step])

def create_two_column_metrics(title1, value1, title2, value2):
    """
    创建两列指标显示
    
    参数:
        title1: 第一列标题
        value1: 第一列值
        title2: 第二列标题
        value2: 第二列值
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label=title1, value=value1)
    
    with col2:
        st.metric(label=title2, value=value2)

def euclidean_distance_calculator():
    """
    欧氏距离计算器
    """
    st.subheader("欧氏距离计算器")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("点 1:")
        x1 = st.number_input("x1", value=0.0, key="x1")
        y1 = st.number_input("y1", value=0.0, key="y1")
    
    with col2:
        st.markdown("点 2:")
        x2 = st.number_input("x2", value=1.0, key="x2")
        y2 = st.number_input("y2", value=1.0, key="y2")
    
    if st.button("计算距离"):
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        st.success(f"欧氏距离: {distance:.4f}")
        
        # 可视化
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter([x1, x2], [y1, y2], color=['red', 'blue'], s=100)
        ax.plot([x1, x2], [y1, y2], 'k--')
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'd = {distance:.2f}', 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig) 