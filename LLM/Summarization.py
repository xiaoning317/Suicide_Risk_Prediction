from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os
import glob

# 大模型调用
glm_model_path = 'ChatGLM2-6B/model'
glm_embeddings = AutoTokenizer.from_pretrained(glm_model_path, trust_remote_code=True)
glm_llm = AutoModel.from_pretrained(glm_model_path, trust_remote_code=True)
glm_llm = glm_llm.eval()
glm_llm.float()

role_define = '''我是非专业的心理干预人员，你是一名心理咨询专家，你的任务是辅助我进行心理辅导，通过患者和我的对话文字，预测来患者的自杀倾向。
请务必记住你自己的身份，不要反转。我将提供同一自杀热线中来电者和我的几段对话文字，每一段对话文字都承接之前的对话文字背景。
请你以专业的口吻进行分析并回复：（1）患者当前遇到的心理问题；（2）患者的压力特征；（3）患者的抑郁倾向；（4）患者的自我关注程度；
（5）患者是否有自杀意图的明确表达；（6）患者的当前情感状态；（7）患者的情感状态变化程度；（8）当前患者的自杀倾向程度。
'''

final_define='''当前内容都已输出完毕，请你以患者的身份，以患者第一人称"我"来自我描述，总结以下几点：（1）当前遇到的心理问题；（2）压力特征；（3）抑郁倾向；（4）自我关注程度；
（5）是否有自杀意图的明确表达；（6）当前情感状态；（7）情感状态变化程度；（8）当前自杀倾向程度。

'''

directory_path = '/home/user416/guanghui/data/huilongguan_audio/texts'
docx_files = glob.glob(os.path.join(directory_path, '*.docx'))
docx_files = docx_files[18:]

for docx_file in docx_files:
    print(docx_file)

    # 切分文件
    file_path = docx_file
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # paragraphs = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    neirong = texts[0].page_content
    paragraphs = [neirong[i:i + 500] for i in range(0, len(neirong), 500)]

    # 用于存储生成的回复
    responses = []
    background = []
    for paragraph  in paragraphs:
        index = 0
        question_input = paragraph
        print(len(paragraphs),"lens")
        print("index",index)
        if question_input:
            query = f"{role_define}参考背景为：{background}\n问题为：{question_input}"
            response, history = glm_llm.chat(glm_embeddings, query=query, history=[])
            print(response)
            responses.append(response)
            background.append(response)
            index= index+1

            if index == len(paragraphs) - 1:
                query2 = f"{final_define}参考背景为：{background}\n"
                final_summary, history = glm_llm.chat(glm_embeddings, query=query2, history=[])
                print(final_summary)
                responses.append(final_summary)

    base_filename = os.path.basename(docx_file)
# Split the base filename using '.' as the separator and get the first part
    filenum = base_filename.split('.')[0]

    print(filenum)  # This will print '202004026696'
    output_file = f"output_new/{filenum}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for response in responses:
            f.write(response + "\n")

    print(file_path, "生成的回复已写入到", output_file)
