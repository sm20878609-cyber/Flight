import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_block = False
for line in lines:
    if "df_result[\"排名\"] = df_result.index + 1" in line and in_block:
        in_block = False
        new_lines.append("    " + line)
        continue
        
    if in_block:
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    elif line.strip() == 'with st.spinner("🤖 AI 正在為您搜羅全球航班並進行智能分析，請稍候..."):':
        in_block = True
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Done!')
