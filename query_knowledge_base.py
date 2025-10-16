"""
知识库查询脚本
功能：通过语义搜索查询知识库中的相关问答
"""

from sentence_transformers import SentenceTransformer
from milvus_client import MyMilvusClient
import sys

class KnowledgeQuerySystem:
    def __init__(self, db_name='test1016', collection_name='jp_knowledge_qa'):
        """初始化查询系统"""
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MyMilvusClient(db_name=db_name)
        
        print("正在加载Embedding模型...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("模型加载完成！\n")
    
    def search(self, query, top_k=5, subject_filter=None):
        """
        搜索知识库
        :param query: 查询问题
        :param top_k: 返回top k个结果
        :param subject_filter: 学科过滤（可选："Python学科" 或 "JAVA学科"）
        """
        print(f"查询问题: {query}")
        if subject_filter:
            print(f"限定学科: {subject_filter}")
        print("=" * 80)
        
        # 生成查询向量
        query_embedding = self.model.encode([query], prompt_name="query")[0]
        
        # 构建过滤条件
        filter_expr = None
        if subject_filter:
            filter_expr = f'subject == "{subject_filter}"'
        
        # 执行搜索（Milvus search不支持filter参数，需要在查询后手动过滤）
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k * 2 if subject_filter else top_k,  # 如果有过滤，多取一些再筛选
            output_fields=["id", "subject", "question", "answer"]
        )
        
        # 显示结果
        for i, hits in enumerate(search_results):
            if len(hits) == 0:
                print("未找到相关结果")
                return
            
            # 如果有学科过滤，筛选结果
            if subject_filter:
                hits = [h for h in hits if h.get('subject') == subject_filter][:top_k]
            
            if len(hits) == 0:
                print(f"未找到 {subject_filter} 相关结果")
                return
                
            print(f"\n找到 {len(hits)} 个相关结果:\n")
            for j, hit in enumerate(hits):
                print(f"{'='*80}")
                print(f"结果 {j+1} (相似度: {hit.get('distance', 0):.4f})")
                print(f"{'='*80}")
                print(f"ID: {hit.get('id', 'N/A')}")
                print(f"学科: {hit.get('subject', 'N/A')}")
                print(f"问题: {hit.get('question', 'N/A')}")
                print(f"\n答案:\n{hit.get('answer', 'N/A')[:500]}...")
                print()
    
    def stats(self):
        """显示数据库统计信息"""
        print("=" * 80)
        print("数据库统计信息")
        print("=" * 80)
        
        # 查询Python学科数量
        python_count = self.client.query(
            self.collection_name,
            filter_expr='subject == "Python学科"',
            output_fields=["id"]
        )
        
        # 查询JAVA学科数量
        java_count = self.client.query(
            self.collection_name,
            filter_expr='subject == "JAVA学科"',
            output_fields=["id"]
        )
        
        print(f"Python学科问答数: {len(python_count)}")
        print(f"JAVA学科问答数: {len(java_count)}")
        print(f"总计: {len(python_count) + len(java_count)}")
        print("=" * 80)

def main():
    """主函数"""
    # 初始化查询系统
    query_system = KnowledgeQuerySystem()
    
    # 显示统计信息
    query_system.stats()
    
    # 示例查询
    print("\n" + "=" * 80)
    print("示例查询")
    print("=" * 80 + "\n")
    
    # 查询1：Python装饰器
    query_system.search(
        query="如何使用装饰器？",
        top_k=3
    )
    
    print("\n" + "=" * 80 + "\n")
    
    # 查询2：限定学科的查询
    query_system.search(
        query="什么是反射？",
        top_k=3,
        subject_filter="JAVA学科"
    )
    
    print("\n" + "=" * 80)
    print("\n交互式查询模式（输入 'quit' 退出）")
    print("=" * 80 + "\n")
    
    # 交互式查询
    while True:
        try:
            user_query = input("\n请输入您的问题: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q', '退出']:
                print("感谢使用！")
                break
            
            if not user_query:
                continue
            
            # 询问是否限定学科
            subject = input("限定学科？(1=Python, 2=JAVA, 回车=不限定): ").strip()
            subject_filter = None
            if subject == '1':
                subject_filter = "Python学科"
            elif subject == '2':
                subject_filter = "JAVA学科"
            
            # 执行查询
            query_system.search(user_query, top_k=3, subject_filter=subject_filter)
            
        except KeyboardInterrupt:
            print("\n\n感谢使用！")
            break
        except Exception as e:
            print(f"查询出错: {e}")

if __name__ == "__main__":
    main()
