"""提取结果的Pydantic模型。

通用模型 — entity_type由域配置驱动，是字符串类型，
而不是具体的Person/Property/Organization子类。
"""

from typing import Any

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """从文档分块中提取的实体。"""

    # 实体名称
    name: str
    # 实体类型，由域配置驱动（如PERSON，ORGANIZATION等）
    entity_type: str
    # 实体属性字典，存储额外信息
    attributes: dict[str, Any] = Field(default_factory=dict)
    # 提取置信度，范围0.0-1.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # 实体出现的原文引用（上下文）
    context: str = ""


class ExtractedRelation(BaseModel):
    """从文档分块中提取的两个实体之间的关系。"""

    # 关系类型
    relation_type: str
    # 源实体名称
    source_entity: str
    # 目标实体名称
    target_entity: str
    # 提取置信度，范围0.0-1.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # 支持此关系的原文引用（证据）
    evidence: str = ""


class ExtractionResult(BaseModel):
    """单个文档分块的完整提取结果。"""

    # 提取出的实体列表
    entities: list[ExtractedEntity] = Field(default_factory=list)
    # 提取出的关系列表
    relations: list[ExtractedRelation] = Field(default_factory=list)
    # 源文档ID
    source_document: str = ""
    # 分块索引（从0开始）
    chunk_index: int = 0


class DocumentExtraction(BaseModel):
    """整个文档的完整提取结果（所有分块合并）。"""

    # 文档ID
    document_id: str
    # 文档文件路径
    document_path: str
    # 处理的分块数
    chunks_processed: int = 0
    # 提取出的实体列表（已去重）
    entities: list[ExtractedEntity] = Field(default_factory=list)
    # 提取出的关系列表
    relations: list[ExtractedRelation] = Field(default_factory=list)
    # 本次提取总成本（美元）
    cost_usd: float = 0.0
    # 使用的LLM模型
    model_used: str = ""
    # 如果提取失败，存储错误信息
    error: str | None = None
    # 增量提取元数据（默认值用于向后兼容已存在的JSON）
    # 使用的域名称
    domain_name: str = ""
    # 使用的分块大小
    chunk_size: int = 0
    # 提取时间戳（ISO格式）
    extracted_at: str = ""
