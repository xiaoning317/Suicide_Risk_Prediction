import openai
import pandas as pd

# 设置 OpenAI API 密钥
openai.api_key = ''

# 读取 CSV 文件
file_path ='../chatglm+chatgpt.csv'
df = pd.read_csv(file_path)

# 假设对话内容存储在第 8 列（索引从 0 开始，所以是 7）
texts = df.iloc[:,7].tolist()

# 初始化一个列表来存储转换后的对话
converted_texts = []

# 遍历列表中的每个对话
for text in texts:

    prompt = f"以下内容是专业的心理专家评价的自杀热线中来电者的当前情况，请你以心理专家的角度来评判这个来电者是否会自杀，输出结果输出0～16之间的自然数，" \
             f"0～7表示中低自杀风险，8～16表示高自杀风险。\n这段话的内容为：{text}"
    response = openai.ChatCompletion.create(
        model="GPT-4 Turbo",  # 使用适合聊天的模型
        messages=[
            {"role": "system", "content": "This is a conversation with a hotline caller."},
            {"role": "user", "content": text}
        ],
        max_tokens=512  # 根据需要调整最大令牌数
    )

    # 提取转换后的文本并添加到列表中
    converted_text = response.choices[0].text.strip()
    converted_texts.append(converted_text)
    print(converted_text)

