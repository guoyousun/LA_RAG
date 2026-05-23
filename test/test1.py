import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# =========================
# 配置区域
# =========================
# 建议将 API Key 配置到环境变量中：
# Windows:
# set DASHSCOPE_API_KEY=你的API_KEY
#
# Linux / Mac:
# export DASHSCOPE_API_KEY=你的API_KEY
#
# 如果你不想使用环境变量，也可以直接写：
# api_key = "你的API_KEY"

api_key = os.getenv("QWEN_API_KEY")

if not api_key:
    raise ValueError("未检测到 DASHSCOPE_API_KEY 环境变量，请先配置 API Key")


# =========================
# 初始化客户端
# =========================
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# =========================
# 系统提示词
# =========================
system_prompt = """
你是一名专业的人工智能助手。

请遵循以下要求：
1. 回答要准确、清晰、专业。
2. 优先使用中文回答。
3. 当用户的问题涉及代码时，提供完整、可运行的示例。
4. 当信息不确定时，明确说明。
5. 回答尽量结构化。
"""


# =========================
# 与模型对话函数
# =========================
def chat_with_qwen(user_input):
    """
    调用 Qwen3-Max 模型进行对话
    :param user_input: 用户输入内容
    :return: 模型回复文本
    """

    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0,
            max_tokens=2048,
            stream=False
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"调用模型时发生错误：{str(e)}"


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-Max 对话程序")
    print("输入 quit 或 exit 退出")
    print("=" * 50)

    while True:
        user_input = input("context: lothair ii ( 835 – ) was the king of lotharingia from 855 until his death. he was the second son of emperor lothair i and ermengarde of tours. he was married to teutberga ( died 875 ), daughter of boso the elder. ermengarde of tours ( d. 20 march 851 ) was the daughter of hugh of tours, a member of the etichonen family. in october 821 in thionville, she married the carolingian emperor lothair i of the franks ( 795 – 855 ). \n Qwenstion: When did Lothair Ii's mother die?")

        if user_input.lower() in ["quit", "exit"]:
            print("程序已退出。")
            break

        answer = chat_with_qwen(user_input)

        print("\nQwen3-Max：")
        print(answer)
