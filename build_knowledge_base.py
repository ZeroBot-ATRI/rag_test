"""
知识库构建脚本（支持混合搜索）
功能：
1. 创建向量数据库表（包含稠密向量、BM25稀疏向量）
2. 读取CSV文件
3. 使用embedding模型将问题转换为稠密向量
4. 使用BM25生成稀疏向量
5. 将数据存入Milvus数据库
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection, connections
from datetime import datetime
from milvus_client import MyMilvusClient
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import jieba
import time
import config

class KnowledgeBaseBuilder:
    def __init__(self, db_name='test1017', collection_name='jp_knowledge_qa'):
        """
        初始化知识库构建器
        :param db_name: 数据库名称
        :param collection_name: 集合（表）名称
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MyMilvusClient(db_name=db_name)
        
        # 连接到Milvus（用于高级操作）
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
            db_name=db_name
        )
        
        # 加载embedding模型
        print("正在加载Embedding模型...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("模型加载完成！")
        
        # BM25模型（后续构建）
        self.bm25 = None
        self.tokenized_corpus = None
        
    def create_collection(self, dim=1024, drop_if_exists=False):
        """
        创建支持混合搜索的集合（稠密向量 + 稀疏向量）
        :param dim: 稠密向量维度
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
        
        print(f"正在创建支持混合搜索的集合 {self.collection_name}...")
        
        # 定义字段schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, description="JP学科知识问答（支持混合搜索）")
        
        # 创建集合
        collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        # 稠密向量索引
        dense_index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="dense_vector", index_params=dense_index_params)
        
        # 稀疏向量索引
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP"
        }
        collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)
        
        collection.load()
        
        print(f"集合 {self.collection_name} 创建成功！")
        
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
    
    def tokenize(self, text):
        """中文分词"""
        return list(jieba.cut(text))
    
    def build_bm25(self, texts):
        """
        构建BM25模型
        :param texts: 文本列表
        """
        print(f"正在构建BM25模型（分词中文文本）...")
        self.tokenized_corpus = [self.tokenize(text) for text in tqdm(texts, desc="分词")]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25模型构建完成！")
    
    def get_sparse_vector(self, doc_idx):
        """
        根据文档索引生成BM25稀疏向量
        :param doc_idx: 文档在语料库中的索引
        :return: 稀疏向量（字典格式）
        """
        if self.tokenized_corpus is None:
            raise ValueError("语料库未构建，请先调用build_bm25方法")
        
        # 获取文档的token
        doc_tokens = self.tokenized_corpus[doc_idx]
        
        # 计算该文档对所有token的BM25分数
        sparse_vector = {}
        for token in set(doc_tokens):
            # 为该文档对这个token的BM25得分
            idf = self.bm25.idf.get(token, 0)
            if idf > 0:
                tf = doc_tokens.count(token)
                # 使用简化的BM25计算
                score = idf * tf
                if score > 0:
                    # 使用token的哈希值作为索引
                    sparse_vector[hash(token) % 100000] = float(score)
        
        return sparse_vector
    
    def generate_embeddings(self, texts, batch_size=32):
        """
        批量生成文本的embedding向量
        :param texts: 文本列表
        :param batch_size: 批处理大小
        :return: embedding向量数组
        """
        print(f"正在生成 {len(texts)} 条文本的Embedding向量...")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="生成稠密向量"):
            batch_texts = texts[i:i+batch_size]
            # 使用query模式生成向量（适合问题检索）
            batch_embeddings = self.model.encode(batch_texts, prompt_name="query")
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"稠密向量生成完成！形状: {embeddings_array.shape}")
        return embeddings_array
    
    def insert_data(self, df, batch_size=100):
        """
        将数据插入到Milvus（包含稠密向量和稀疏向量）
        :param df: 包含数据的DataFrame
        :param batch_size: 批量插入大小
        """
        print(f"\n开始处理并插入数据到集合 {self.collection_name}...")
        
        # 1. 生成稠密向量
        questions = df['问题'].tolist()
        dense_embeddings = self.generate_embeddings(questions)
        
        # 2. 构建BM25模型并生成稀疏向量
        self.build_bm25(questions)
        
        # 3. 准备并插入数据
        total_count = len(df)
        success_count = 0
        start_id = 1
        
        # 获取Collection对象用于插入
        collection = Collection(name=self.collection_name)
        
        for i in tqdm(range(0, total_count, batch_size), desc="插入数据"):
            batch_df = df.iloc[i:i+batch_size]
            batch_dense = dense_embeddings[i:i+batch_size]
            
            # 构建批量数据
            batch_data = {
                "id": [],
                "subject": [],
                "question": [],
                "answer": [],
                "dense_vector": [],
                "sparse_vector": [],
                "timestamp": []
            }
            
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                doc_idx = i + idx
                batch_data["id"].append(start_id + doc_idx)
                batch_data["subject"].append(row['学科名称'])
                batch_data["question"].append(row['问题'])
                batch_data["answer"].append(row['答案'])
                batch_data["dense_vector"].append(batch_dense[idx].tolist())
                batch_data["sparse_vector"].append(self.get_sparse_vector(doc_idx))
                batch_data["timestamp"].append(int(datetime.now().timestamp()))
            
            try:
                collection.insert([
                    batch_data["id"],
                    batch_data["subject"],
                    batch_data["question"],
                    batch_data["answer"],
                    batch_data["dense_vector"],
                    batch_data["sparse_vector"],
                    batch_data["timestamp"]
                ])
                success_count += len(batch_data["id"])
            except Exception as e:
                print(f"\n批次 {i//batch_size + 1} 插入失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        collection.flush()
        print(f"\n数据插入完成！成功插入 {success_count}/{total_count} 条数据")
    
    def verify_data(self, sample_size=5):
        """
        验证插入的数据
        :param sample_size: 抽样查询的数量
        """
        print(f"\n正在验证数据...")
        
        try:
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

def main():
    """主函数"""
    print("=" * 80)
    print("知识库构建系统（混合搜索版本）")
    print("=" * 80)
    
    # 配置参数
    DB_NAME = 'test1017'
    COLLECTION_NAME = 'jp_knowledge_qa'
    CSV_PATH = 'data/JP学科知识问答.csv'
    VECTOR_DIM = 1024  # Qwen3-Embedding-0.6B输出1024维向量
    
    try:
        # 1. 初始化构建器
        builder = KnowledgeBaseBuilder(db_name=DB_NAME, collection_name=COLLECTION_NAME)
        
        # 2. 创建集合
        builder.create_collection(dim=VECTOR_DIM, drop_if_exists=True)
        
        # 3. 加载CSV数据
        df = builder.load_csv_data(CSV_PATH)
        
        # 4. 插入数据（包含稠密向量和稀疏向量）
        builder.insert_data(df, batch_size=50)
        
        # 5. 验证数据
        builder.verify_data(sample_size=5)
        
        print("\n" + "=" * 80)
        print("知识库构建完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
