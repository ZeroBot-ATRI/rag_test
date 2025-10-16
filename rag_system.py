"""
RAG (检索增强生成) 系统
功能：
1. 从向量数据库检索相关知识
2. 使用LLM生成高质量回答
3. 提供完整的问答体验
"""

from sentence_transformers import SentenceTransformer
from milvus_client import MyMilvusClient
from openai import OpenAI
import config
import sys

class PromptTemplate:
    """提示词模板管理器"""
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一个专业的编程学习助手，专注于Python和JAVA技术问答。
你的任务是根据提供的知识库内容，为用户提供准确、详细、易懂的技术解答。

回答要求：
1. 基于提供的参考资料回答问题，保持准确性
2. 如果参考资料中有代码示例，请完整展示
3. 用清晰的语言解释技术概念
4. 如果参考资料不够充分，可以适当补充你的知识，但要明确说明
5. 保持专业、友好的语气
6. 使用Markdown格式组织答案，提升可读性
"""

    # 用户问题模板
    USER_PROMPT_TEMPLATE = """用户问题：{question}

参考资料（来自知识库）：
{context}

请基于以上参考资料回答用户的问题。如果参考资料已经包含完整答案，请直接使用；如果需要补充说明，请在保持准确性的前提下适当展开。
"""

    # 无参考资料时的模板
    NO_CONTEXT_PROMPT = """用户问题：{question}

知识库中未找到直接相关的参考资料。

请基于你的专业知识回答这个问题，并在回答开头说明"知识库中未找到直接相关的内容，以下是基于通用知识的回答"。
"""

    @classmethod
    def format_context(cls, search_results, max_results=3):
        """
        格式化检索结果为上下文
        :param search_results: 检索结果
        :param max_results: 最多使用的结果数量
        :return: 格式化的上下文字符串
        """
        if not search_results or len(search_results) == 0:
            return None
        
        context_parts = []
        for idx, result in enumerate(search_results[:max_results], 1):
            subject = result.get('subject', 'N/A')
            question = result.get('question', 'N/A')
            answer = result.get('answer', 'N/A')
            distance = result.get('distance', 0)
            
            context_parts.append(f"""
【参考资料 {idx}】(相似度: {1-distance:.2%})
学科：{subject}
问题：{question}
答案：{answer}
""")
        
        return "\n".join(context_parts)
    
    @classmethod
    def create_messages(cls, question, context=None):
        """
        创建对话消息列表
        :param question: 用户问题
        :param context: 检索到的上下文
        :return: 消息列表
        """
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT}
        ]
        
        if context:
            user_content = cls.USER_PROMPT_TEMPLATE.format(
                question=question,
                context=context
            )
        else:
            user_content = cls.NO_CONTEXT_PROMPT.format(question=question)
        
        messages.append({"role": "user", "content": user_content})
        
        return messages


class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, db_name='test1016', collection_name='jp_knowledge_qa'):
        """初始化RAG系统"""
        print("正在初始化RAG系统...")
        
        # 初始化向量数据库
        print("1. 连接向量数据库...")
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MyMilvusClient(db_name=db_name)
        
        # 加载Embedding模型
        print("2. 加载Embedding模型...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        
        # 初始化LLM客户端
        print("3. 初始化LLM客户端...")
        self.llm_client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL
        )
        self.llm_model = config.MODEL
        
        print("✓ RAG系统初始化完成！\n")
    
    def retrieve(self, query, top_k=3, subject_filter=None, similarity_threshold=0.7):
        """
        从向量数据库检索相关知识
        :param query: 查询问题
        :param top_k: 返回top k个结果
        :param subject_filter: 学科过滤
        :param similarity_threshold: 相似度阈值（距离越小越相似，这里是最大距离）
        :return: 检索结果列表
        """
        # 生成查询向量
        query_embedding = self.model.encode([query], prompt_name="query")[0]
        
        # 执行搜索
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k * 2 if subject_filter else top_k,
            output_fields=["id", "subject", "question", "answer"]
        )
        
        # 处理结果
        results = []
        for hits in search_results:
            for hit in hits:
                # 过滤学科
                if subject_filter and hit.get('subject') != subject_filter:
                    continue
                
                # 过滤相似度
                distance = hit.get('distance', 1.0)
                if distance > similarity_threshold:
                    continue
                
                results.append(hit)
                
                if len(results) >= top_k:
                    break
            
            if len(results) >= top_k:
                break
        
        return results
    
    def generate(self, messages, stream=False, temperature=0.7, max_tokens=2000):
        """
        使用LLM生成回答
        :param messages: 对话消息列表
        :param stream: 是否流式输出
        :param temperature: 温度参数（0-1，越高越随机）
        :param max_tokens: 最大token数
        :return: 生成的回答
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if stream:
                return response  # 返回流式响应对象
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"LLM调用失败: {e}"
    
    def answer(self, question, subject_filter=None, top_k=3, 
               stream=False, show_context=True, temperature=0.7):
        """
        完整的RAG问答流程
        :param question: 用户问题
        :param subject_filter: 学科过滤
        :param top_k: 检索结果数量
        :param stream: 是否流式输出
        :param show_context: 是否显示检索到的上下文
        :param temperature: LLM温度参数
        :return: 回答内容
        """
        print("=" * 80)
        print(f"问题：{question}")
        if subject_filter:
            print(f"限定学科：{subject_filter}")
        print("=" * 80)
        
        # 1. 检索相关知识
        print("\n[1/3] 正在检索相关知识...")
        results = self.retrieve(question, top_k=top_k, subject_filter=subject_filter)
        
        if results:
            print(f"✓ 找到 {len(results)} 条相关参考资料")
        else:
            print("✗ 未找到相关参考资料，将使用LLM通用知识回答")
        
        # 2. 格式化上下文
        context = PromptTemplate.format_context(results, max_results=top_k)
        
        # 显示检索到的上下文
        if show_context and context:
            print("\n" + "-" * 80)
            print("检索到的参考资料：")
            print("-" * 80)
            for idx, result in enumerate(results[:top_k], 1):
                print(f"\n【{idx}】相似度: {1-result.get('distance', 1):.2%}")
                print(f"学科：{result.get('subject', 'N/A')}")
                print(f"问题：{result.get('question', 'N/A')[:80]}...")
            print("-" * 80)
        
        # 3. 生成回答
        print("\n[2/3] 正在生成回答...")
        messages = PromptTemplate.create_messages(question, context)
        
        print("\n[3/3] 回答内容：")
        print("=" * 80)
        
        if stream:
            # 流式输出
            response_stream = self.generate(messages, stream=True, temperature=temperature)
            full_response = ""
            try:
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        full_response += content
                print()  # 换行
            except Exception as e:
                print(f"\n流式输出错误: {e}")
                return None
            return full_response
        else:
            # 非流式输出
            answer = self.generate(messages, stream=False, temperature=temperature)
            print(answer)
            print("=" * 80)
            return answer
    
    def chat(self):
        """交互式聊天模式"""
        print("\n" + "=" * 80)
        print("RAG 智能问答系统 - 交互模式")
        print("=" * 80)
        print("\n使用说明：")
        print("  - 直接输入问题进行提问")
        print("  - 输入 '/python' 或 '/java' 限定学科")
        print("  - 输入 '/stream' 切换流式/非流式输出")
        print("  - 输入 '/context' 切换是否显示参考资料")
        print("  - 输入 '/clear' 清空学科限定")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("=" * 80 + "\n")
        
        subject_filter = None
        stream_mode = True
        show_context = True
        
        while True:
            try:
                user_input = input("\n💬 您的问题: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("\n感谢使用！再见！👋")
                    break
                
                if user_input.lower() == '/python':
                    subject_filter = "Python学科"
                    print(f"✓ 已限定学科：{subject_filter}")
                    continue
                
                if user_input.lower() == '/java':
                    subject_filter = "JAVA学科"
                    print(f"✓ 已限定学科：{subject_filter}")
                    continue
                
                if user_input.lower() == '/clear':
                    subject_filter = None
                    print("✓ 已清除学科限定")
                    continue
                
                if user_input.lower() == '/stream':
                    stream_mode = not stream_mode
                    print(f"✓ 流式输出：{'开启' if stream_mode else '关闭'}")
                    continue
                
                if user_input.lower() == '/context':
                    show_context = not show_context
                    print(f"✓ 显示参考资料：{'开启' if show_context else '关闭'}")
                    continue
                
                # 执行问答
                self.answer(
                    question=user_input,
                    subject_filter=subject_filter,
                    top_k=3,
                    stream=stream_mode,
                    show_context=show_context,
                    temperature=0.7
                )
                
            except KeyboardInterrupt:
                print("\n\n感谢使用！再见！👋")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback
                traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("RAG 智能问答系统")
    print("=" * 80 + "\n")
    
    # 初始化系统
    rag = RAGSystem()
    
    # 示例问答
    print("\n" + "=" * 80)
    print("示例问答")
    print("=" * 80 + "\n")
    
    # 示例1：Python装饰器
    rag.answer(
        question="如何使用装饰器计算函数运行时间？",
        top_k=2,
        stream=False,
        show_context=True
    )
    
    print("\n" + "=" * 80 + "\n")
    
    # 询问是否进入交互模式
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        rag.chat()
    else:
        print("\n感谢使用！")


if __name__ == "__main__":
    main()
