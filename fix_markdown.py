#!/usr/bin/env python3
# 修复markdown格式错误
with open('week4_app/pages/evaluation.py', 'r') as f:
    content = f.read()

# 替换错误的markdown格式
content = content.replace("""
             5折交叉验证过程**:""", """
             **5折交叉验证过程**:""")

with open('week4_app/pages/evaluation.py', 'w') as f:
    f.write(content)

print('修复完成') 