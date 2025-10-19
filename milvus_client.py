"""
Milvus客户端封装
"""

from pymilvus import MilvusClient as PyMilvusClient
from pymilvus import AnnSearchRequest, RRFRanker
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

    def search(self, collection_name, data, limit=10, output_fields=None, filter=None, anns_field="vector"):
        """向量相似度搜索
        :param filter: 过滤表达式，例如 'subject == "Python学科"'
        :param anns_field: 向量字段名称，默认为"vector"
        """
        search_params = {
            "collection_name": collection_name,
            "data": data,
            "limit": limit,
            "output_fields": output_fields,
            "anns_field": anns_field
        }
        
        # 如果有过滤条件，添加filter参数
        if filter:
            search_params["filter"] = filter
        
        return self.client.search(**search_params)
    
    def hybrid_search(self, collection_name, dense_vector, sparse_vector, limit=10, output_fields=None, filter=None):
        """混合搜索：稠密向量 + 稀疏向量（BM25）
        :param collection_name: 集合名称
        :param dense_vector: 稠密向量
        :param sparse_vector: 稀疏向量（字典格式）
        :param limit: 返回结果数量
        :param output_fields: 输出字段
        :param filter: 过滤条件
        :return: 搜索结果
        """
        # 构建稠密向量搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=filter
        )
        
        # 构建稀疏向量搜索请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit,
            expr=filter
        )
        
        # 使用RRF（Reciprocal Rank Fusion）进行混合搜索
        return self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=output_fields
        )
    
    def describe_collection(self, collection_name):
        """查看collection的结构"""
        return self.client.describe_collection(collection_name)
    
    def query(self, collection_name, filter_expr, output_fields=None, limit=None):
        """根据条件查询数据
        :param collection_name: 集合名称
        :param filter_expr: 过滤表达式
        :param output_fields: 输出字段列表
        :param limit: 返回结果数量限制（可选）
        :return: 查询结果
        """
        query_params = {
            "collection_name": collection_name,
            "filter": filter_expr,
            "output_fields": output_fields
        }
        
        # 如果指定了limit，添加到参数中
        if limit is not None:
            query_params["limit"] = limit
            
        return self.client.query(**query_params)
