import os
import numpy as np
import math
from utils.svg_generator import (
    generate_logistic_function_svg,
    generate_svm_concept_svg,
    generate_model_comparison_svg,
    generate_learning_path_svg
)

def save_svg_to_file(svg_content, filename):
    """将SVG内容保存到文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG已保存到 {filename}")

def main():
    """生成所有概念SVG图并保存"""
    # 创建img目录
    os.makedirs('img/week3/svg', exist_ok=True)
    
    # 生成并保存Sigmoid函数图
    sigmoid_svg = generate_logistic_function_svg()
    save_svg_to_file(sigmoid_svg, 'img/week3/svg/sigmoid_function.svg')
    
    # 生成并保存SVM概念图
    svm_concept_svg = generate_svm_concept_svg()
    save_svg_to_file(svm_concept_svg, 'img/week3/svg/svm_concept.svg')
    
    # 生成并保存模型比较图
    model_comparison_svg = generate_model_comparison_svg()
    save_svg_to_file(model_comparison_svg, 'img/week3/svg/model_comparison.svg')
    
    # 生成并保存学习路径图
    learning_path_svg = generate_learning_path_svg()
    save_svg_to_file(learning_path_svg, 'img/week3/svg/learning_path.svg')
    
    print("所有SVG概念图已成功生成！")

if __name__ == "__main__":
    main() 