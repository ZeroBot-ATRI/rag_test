"""
知识库构建脚本
功能：
1. 创建向量数据库表（包含id、学科、问题、答案、向量、时间戳字段）
2. 读取CSV文件
3. 使用embedding模型将问题转换为向量
4. 将数据存入Milvus数据库
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import DataType, FieldSchema, CollectionSchema
from datetime import datetime
from milvus_client import MyMilvusClient
from tqdm import tqdm
import time

class KnowledgeBaseBuilder:
    def __init__(self, db_name='test1016', collection_name='jp_knowledge_qa'):
        """
        初始化知识库构建器
        :param db_name: 数据库名称
        :param collection_name: 集合（表）名称
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MyMilvusClient(db_name=db_name)
        
        # 加载embedding模型
        print("正在加载Embedding模型...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("模型加载完成！")
        
    def create_collection(self, dim=768, drop_if_exists=False):
        """
        创建集合（表）
        :param dim: 向量维度
        :param drop_if_exists: 如果集合已存在是否删除
        """
        # 检查集合是否已存在
        collections = self.client.list_collections()
        if self.collection_name in collections:
            if drop_if_exists:
                print(f"集合 {self.collection_name} 已存在，正在删除...")
                self.client.client.drop_collection(self.collection_name)
                print("删除成功！")
            else:
                print(f"集合 {self.collection_name} 已存在，将使用现有集合")
                return
        
        # 定义字段schema
        print(f"正在创建集合 {self.collection_name}...")
        
        # 使用客户端的create_collection方法（自动创建schema）
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim
        )
        
        print(f"集合 {self.collection_name} 创建成功！")
        print(f"集合结构: {self.client.describe_collection(self.collection_name)}")
        
    def load_csv_data(self, csv_path):
        """
        加载CSV数据
        :param csv_path: CSV文件路径
        :return: DataFrame
        """
        print(f"正在读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条数据")
        print(f"列名: {df.columns.tolist()}")
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n学科分布:\n{df['学科名称'].value_counts()}")
        return df
    
    def generate_embeddings(self, texts, batch_size=32):
        """
        批量生成文本的embedding向量
        :param texts: 文本列表
        :param batch_size: 批处理大小
        :return: embedding向量数组
        """
        print(f"正在生成 {len(texts)} 条文本的Embedding向量...")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="生成向量"):
            batch_texts = texts[i:i+batch_size]
            # 使用query模式生成向量（适合问题检索）
            batch_embeddings = self.model.encode(batch_texts, prompt_name="query")
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"向量生成完成！形状: {embeddings_array.shape}")
        return embeddings_array
    
    def insert_data(self, df, batch_size=100):
        """
        将数据插入到Milvus
        :param df: 包含数据的DataFrame
        :param batch_size: 批量插入大小
        """
        print(f"\n开始处理并插入数据到集合 {self.collection_name}...")
        
        # 生成embedding向量（使用问题字段）
        questions = df['问题'].tolist()
        embeddings = self.generate_embeddings(questions)
        
        # 准备插入数据
        total_count = len(df)
        success_count = 0
        start_id = 1  # 起始ID
        
        for i in tqdm(range(0, total_count, batch_size), desc="插入数据"):
            batch_df = df.iloc[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            # 构建批量数据
            batch_data = []
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                data_item = {
                    "id": start_id + i + idx,  # 添加自增ID
                    "vector": batch_embeddings[idx].tolist(),
                    "subject": row['学科名称'],
                    "question": row['问题'],
                    "answer": row['答案'],
                    "timestamp": int(datetime.now().timestamp())
                }
                batch_data.append(data_item)
            
            try:
                # 插入数据
                self.client.insert(self.collection_name, batch_data)
                success_count += len(batch_data)
            except Exception as e:
                print(f"\n批次 {i//batch_size + 1} 插入失败: {e}")
                continue
        
        print(f"\n数据插入完成！成功插入 {success_count}/{total_count} 条数据")
    
    def verify_data(self, sample_size=5):
        """
        验证插入的数据
        :param sample_size: 抽样查询的数量
        """
        print(f"\n正在验证数据...")
        
        try:
            # 随机查询几条数据（使用id范围限制返回数量）
            query_result = self.client.query(
                self.collection_name,
                filter_expr=f"id > 0 and id <= {sample_size}",
                output_fields=["id", "subject", "question"]
            )
            
            print(f"\n随机抽样 {len(query_result)} 条数据:")
            for item in query_result:
                question = item.get('question', '')
                question_preview = question[:50] if question else 'N/A'
                print(f"ID: {item.get('id')}, 学科: {item.get('subject')}, 问题: {question_preview}...")
                
        except Exception as e:
            print(f"验证失败: {e}")
    
    def test_search(self, query_text="Python中如何使用装饰器？", top_k=3):
        """
        测试向量搜索功能
        :param query_text: 查询文本
        :param top_k: 返回top k个结果
        """
        print(f"\n测试向量搜索...")
        print(f"查询问题: {query_text}")
        
        # 生成查询向量
        query_embedding = self.model.encode([query_text], prompt_name="query")[0]
        
        # 执行搜索
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=["subject", "question", "answer"]
        )
        
        print(f"\n找到最相似的 {top_k} 个结果:")
        for i, hits in enumerate(search_results):
            for j, hit in enumerate(hits):
                print(f"\n结果 {j+1}:")
                print(f"  相似度: {hit.get('distance', 0):.4f}")
                print(f"  学科: {hit.get('subject', 'N/A')}")
                print(f"  问题: {hit.get('question', 'N/A')[:100]}...")
                print(f"  答案: {hit.get('answer', 'N/A')[:200]}...")

def main():
    """主函数"""
    print("=" * 80)
    print("知识库构建系统")
    print("=" * 80)
    
    # 配置参数
    DB_NAME = 'test1016'
    COLLECTION_NAME = 'jp_knowledge_qa'
    CSV_PATH = 'data/JP学科知识问答.csv'
    VECTOR_DIM = 1024  # Qwen3-Embedding-0.6B输出1024维向量
    
    try:
        # 1. 初始化构建器
        builder = KnowledgeBaseBuilder(db_name=DB_NAME, collection_name=COLLECTION_NAME)
        
        # 2. 创建集合（如果已存在则提示）
        builder.create_collection(dim=VECTOR_DIM, drop_if_exists=True)  # 先删除旧集合
        
        # 3. 加载CSV数据
        df = builder.load_csv_data(CSV_PATH)
        
        # 4. 插入数据
        builder.insert_data(df, batch_size=50)
        
        # 5. 验证数据
        builder.verify_data(sample_size=5)
        
        # 6. 测试搜索
        builder.test_search("如何使用装饰器实现函数运行时间计算？", top_k=3)
        
        print("\n" + "=" * 80)
        print("知识库构建完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
