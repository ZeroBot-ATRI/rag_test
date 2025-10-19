"""
RAG (检索增强生成) 系统 - 混合搜索+重排版本
功能：
1. 混合搜索：稠密向量 + BM25稀疏向量，获取40个候选结果
2. 重排序：对40个候选结果进行重排，得到最终5个结果
3. 使用LLM生成高质量回答
4. 提供完整的问答体验
"""

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from milvus_client import MyMilvusClient
from openai import OpenAI
import jieba
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
    def format_context(cls, search_results, max_results=5):
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
            score = result.get('score', 0)
            
            context_parts.append(f"""
【参考资料 {idx}】(相关性得分: {score:.4f})
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
    """混合搜索 + 重排RAG系统"""
    
    def __init__(self, db_name='test1017', collection_name='jp_knowledge_qa'):
        """初始化RAG系统"""
        print("正在初始化RAG系统（混合搜索+重排版本）...")
        
        # 初始化向量数据库
        print("1. 连接向量数据库...")
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MyMilvusClient(db_name=db_name)
        
        # 加载Embedding模型
        print("2. 加载Embedding模型...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        
        # 加载重排模型
        print("3. 加载重排模型...")
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        # 初始化LLM客户端
        print("4. 初始化LLM客户端...")
        self.llm_client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL
        )
        self.llm_model = config.MODEL
        
        # 初始化BM25（用于生成稀疏向量）
        self.bm25 = None
        self.question_list = None
        self._init_bm25()
        
        print("✓ RAG系统初始化完成！\n")
    
    def _init_bm25(self):
        """初始化BM25模型（从数据库加载所有问题）"""
        print("   正在初始化BM25模型...")
        try:
            # 从数据库加载所有问题
            from rank_bm25 import BM25Okapi
            all_docs = self.client.query(
                self.collection_name,
                filter_expr="id > 0",
                output_fields=["question"],
                limit=10000
            )
            
            self.question_list = [doc['question'] for doc in all_docs]
            tokenized_corpus = [list(jieba.cut(q)) for q in self.question_list]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"   BM25模型初始化完成（语料库大小: {len(self.question_list)}）")
        except Exception as e:
            print(f"   BM25初始化失败: {e}")
            self.bm25 = None
    
    def get_sparse_vector(self, text):
        """
        生成BM25稀疏向量
        :param text: 查询文本
        :return: 稀疏向量（字典格式）
        """
        if self.bm25 is None or self.question_list is None:
            return {}
        
        tokenized_query = list(jieba.cut(text))
        scores = self.bm25.get_scores(tokenized_query)
        
        # 转换为稀疏向量格式
        sparse_vector = {}
        for idx, score in enumerate(scores):
            if score > 0:
                token = self.question_list[idx][:10]  # 使用问题前10个字符作为token
                sparse_vector[hash(token) % 100000] = float(score)
        
        return sparse_vector
    
    def hybrid_retrieve(self, query, top_k=40, subject_filter=None):
        """
        混合搜索：稠密向量 + BM25稀疏向量
        :param query: 查询问题
        :param top_k: 返回top k个结果（默认40）
        :param subject_filter: 学科过滤
        :return: 检索结果列表
        """
        # 生成稠密向量
        dense_vector = self.model.encode([query], prompt_name="query")[0].tolist()
        
        # 生成稀疏向量
        sparse_vector = self.get_sparse_vector(query)
        
        # 构建过滤条件
        filter_expr = None
        if subject_filter:
            filter_expr = f'subject == "{subject_filter}"'
        
        # 执行混合搜索
        try:
            search_results = self.client.hybrid_search(
                collection_name=self.collection_name,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=top_k,
                output_fields=["id", "subject", "question", "answer"],
                filter=filter_expr
            )
            
            # 处理结果
            results = []
            for hits in search_results:
                for hit in hits:
                    results.append({
                        'id': hit.get('id'),
                        'subject': hit.get('subject'),
                        'question': hit.get('question'),
                        'answer': hit.get('answer'),
                        'distance': hit.get('distance', 1.0)
                    })
            
            return results
        except Exception as e:
            # 如果混合搜索失败，降级为普通向量搜索
            print(f"混合搜索失败，降级为普通搜索: {e}")
            return self._fallback_search(query, top_k, subject_filter)
    
    def _fallback_search(self, query, top_k, subject_filter):
        """降级搜索方案：仅使用稠密向量"""
        dense_vector = self.model.encode([query], prompt_name="query")[0].tolist()
        filter_expr = f'subject == "{subject_filter}"' if subject_filter else None
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vector],
            limit=top_k,
            output_fields=["id", "subject", "question", "answer"],
            filter=filter_expr,
            anns_field="dense_vector"
        )
        
        results = []
        for hits in search_results:
            for hit in hits:
                results.append({
                    'id': hit.get('id'),
                    'subject': hit.get('subject'),
                    'question': hit.get('question'),
                    'answer': hit.get('answer'),
                    'distance': hit.get('distance', 1.0)
                })
        
        return results
    
    def rerank(self, query, candidates, top_k=5):
        """
        重排序候选结果
        :param query: 查询问题
        :param candidates: 候选结果列表
        :param top_k: 返回top k个结果（默认5）
        :return: 重排后的结果列表
        """
        if not candidates:
            return []
        
        # 准备重排输入
        pairs = [[query, candidate['question'] + ' ' + candidate['answer'][:200]] 
                 for candidate in candidates]
        
        # 执行重排
        scores = self.reranker.compute_score(pairs)
        
        # 将分数添加到候选结果中
        for i, candidate in enumerate(candidates):
            candidate['score'] = scores[i] if isinstance(scores, list) else scores
        
        # 按分数排序
        reranked_results = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        return reranked_results[:top_k]
    
    def retrieve(self, query, subject_filter=None, hybrid_top_k=40, final_top_k=5):
        """
        完整检索流程：混合搜索 + 重排
        :param query: 查询问题
        :param subject_filter: 学科过滤
        :param hybrid_top_k: 混合搜索返回的候选数量（默认40）
        :param final_top_k: 重排后最终返回的结果数量（默认5）
        :return: 最终检索结果列表
        """
        # 1. 混合搜索获取候选结果
        candidates = self.hybrid_retrieve(query, top_k=hybrid_top_k, subject_filter=subject_filter)
        
        if not candidates:
            return []
        
        # 2. 重排序
        final_results = self.rerank(query, candidates, top_k=final_top_k)
        
        return final_results
    
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
                return response
            else:
                # 兼容不同API站的返回格式
                if isinstance(response, str):
                    return response
                elif hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content
                else:
                    return str(response)
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"LLM调用失败: {e}\n详细错误:\n{error_detail}"
    
    def answer(self, question, subject_filter=None, stream=False, show_context=True, 
               temperature=0.7, hybrid_top_k=40, final_top_k=5):
        """
        完整的RAG问答流程（混合搜索+重排）
        :param question: 用户问题
        :param subject_filter: 学科过滤
        :param stream: 是否流式输出
        :param show_context: 是否显示检索到的上下文
        :param temperature: LLM温度参数
        :param hybrid_top_k: 混合搜索候选数量
        :param final_top_k: 最终返回结果数量
        :return: 回答内容
        """
        print("=" * 80)
        print(f"问题：{question}")
        if subject_filter:
            print(f"限定学科：{subject_filter}")
        print("=" * 80)
        
        # 1. 混合检索 + 重排
        print(f"\n[1/3] 正在检索相关知识（混合搜索Top{hybrid_top_k} -> 重排Top{final_top_k}）...")
        results = self.retrieve(
            query=question,
            subject_filter=subject_filter,
            hybrid_top_k=hybrid_top_k,
            final_top_k=final_top_k
        )
        
        if results:
            print(f"✓ 最终获得 {len(results)} 条高质量参考资料")
        else:
            print("✗ 未找到相关参考资料，将使用LLM通用知识回答")
        
        # 2. 格式化上下文
        context = PromptTemplate.format_context(results, max_results=final_top_k)
        
        # 显示检索到的上下文
        if show_context and results:
            print("\n" + "-" * 80)
            print("检索到的参考资料（重排后）：")
            print("-" * 80)
            for idx, result in enumerate(results[:final_top_k], 1):
                print(f"\n【{idx}】相关性得分: {result.get('score', 0):.4f}")
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
                if response_stream is None:
                    print("\n错误：未获取到流式响应")
                    return None
                    
                for chunk in response_stream:
                    # 兼容不同API站的流式返回格式
                    if isinstance(chunk, str):
                        content = chunk
                        print(content, end='', flush=True)
                        full_response += content
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            print(content, end='', flush=True)
                            full_response += content
                print()  # 换行
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"\n流式输出错误: {e}")
                print(f"详细错误:\n{error_detail}")
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
        print("RAG 智能问答系统 - 交互模式（混合搜索+重排）")
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
                    stream=stream_mode,
                    show_context=show_context,
                    temperature=0.7,
                    hybrid_top_k=40,
                    final_top_k=5
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
    print("RAG 智能问答系统（混合搜索+重排版本）")
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
        stream=False,
        show_context=True,
        hybrid_top_k=40,
        final_top_k=5
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
