"""
Milvus客户端封装
"""

from pymilvus import MilvusClient as PyMilvusClient
import config
class MyMilvusClient:
    def __init__(self, db_name='default'):
        self.client = PyMilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}", db_name=db_name)

    def list_collections(self):
        return self.client.list_collections()

    def create_collection(self, collection_name, dimension):
        self.client.create_collection(collection_name, dimension)

    def insert(self, collection_name, vectors):
        self.client.insert(collection_name, vectors)

    def search(self, collection_name, data, limit=10, output_fields=None):
        """向量相似度搜索"""
        return self.client.search(
            collection_name=collection_name,
            data=data,
            limit=limit,
            output_fields=output_fields
        )
    
    def describe_collection(self, collection_name):
        """查看collection的结构"""
        return self.client.describe_collection(collection_name)
    
    def query(self, collection_name, filter_expr, output_fields=None):
        """根据条件查询数据"""
        return self.client.query(collection_name, filter=filter_expr, output_fields=output_fields)
