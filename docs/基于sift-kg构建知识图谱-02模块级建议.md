# 基于 sift-kg 借鉴实现知识图谱构建系统 - 模块级建议

**返回主目录**: [设计文档.md](设计文档.md)  
**上一页**: [基于sift-kg构建知识图谱-01架构借鉴.md](基于sift-kg构建知识图谱-01架构借鉴.md)

---

## 目录

- [1. 配置管理模块建议](#1-配置管理模块建议)
- [2. 文档摄取模块建议](#2-文档摄取模块建议)
- [3. 域配置与Schema建议](#3-域配置与Schema建议)
- [4. 抽取模块建议](#4-抽取模块建议)
- [5. 知识图谱核心建议](#5-知识图谱核心建议)
- [6. 实体消歧模块建议](#6-实体消歧模块建议)

---

## 1. 配置管理模块建议

### 1.1 配置优先级设计

sift-kg 的配置优先级从高到低：

1. **CLI 标志**（最高）- 临时覆盖很方便
2. **环境变量** - 部署时用，敏感信息（API密钥）不写配置文件
3. `.env` 文件 - 本地开发
4. `sift.yaml` 项目配置 - 项目默认配置
5. **默认值**（最低）

这个设计非常成熟，推荐直接借鉴。使用 `pydantic-settings` 可以很容易实现。

### 1.2 关键设计点参考

```python
class SiftConfig(BaseSettings):
    # LLM配置
    llm_model: str
    llm_max_concurrent: int = 4  # 控制并发，避免速率限制
    llm_max_retries: int = 3
    
    # API密钥 - 支持多个提供商
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # ...
    
    # OCR配置
    ocr_enabled: bool = False
    ocr_backend: OcrBackend = OcrBackend.tesseract
    ocr_language: str = "eng"
    
    # 域配置
    domain: str | None = None
    
    # 输出
    output_dir: Path = Path("output")
```

**验证要点**:
- Ollama 本地模型不需要API密钥校验
- 其他提供商必须提供密钥
- 自动创建输出目录

## 2. 文档摄取模块建议

### 2.1 协议设计

使用协议模式（Protocol）定义抽取器接口，便于添加新后端：

```python
class TextExtractor(Protocol):
    def extract_text(self, file_path: Path) -> str: ...
```

任何实现 `extract_text` 的类都可以即插即用，符合开放封闭原则。

### 2.2 OCR 抽象

支持多个OCR后端，统一接口：

| 后端 | 特点 | 需要API | 适用场景 |
|------|------|--------|---------|
| `tesseract` | 默认，本地，开源 | 否 | 通用场景 |
| `easyocr` | 精度较好，内存大 | 否 | 小批量 |
| `paddleocr` | 中文支持好 | 否 | 中文文档 |
| `gcv` | Google Cloud，高精度 | 是 | 生产环境 |

这个设计允许用户根据自己需求选择，很灵活。

### 2.3 自动OCR回退

对PDF自动检测：
- 如果提取得到非空文本 → 直接使用
- 如果提取得到空 → 回退到OCR

对用户透明，支持混合文件夹（既有文本PDF又有扫描PDF）。这个用户体验很好，一定要做。

### 2.4 文本分块

将长文本分割成LLM可处理的块（默认10,000字符）：

设计要点：
- 按段落边界分割，避免切断句子
- 尽量保持块接近但不超过目标大小
- 保留位置信息
- 短文档整个作为一个块

## 3. 域配置与Schema建议

### 3.1 域配置模型设计

```python
class EntityTypeConfig(BaseModel):
    description: str                    # 类型描述，注入prompt
    extraction_hints: list[str]        # 提取提示帮助LLM理解
    canonical_names: list[str]          # 封闭词汇表
    canonical_fallback_type: str | None # 不匹配时回退

class RelationTypeConfig(BaseModel):
    description: str
    source_types: list[str]            # 允许的源实体类型
    target_types: list[str]            # 允许的目标实体类型
    symmetric: bool = False            # 是否对称关系
    review_required: bool = False      # 是否需要人工审核

class DomainConfig(BaseModel):
    name: str
    entity_types: dict[str, EntityTypeConfig]
    relation_types: dict[str, RelationTypeConfig]
    fallback_relation: str | None      # 不匹配的关系回退
    schema_free: bool = False          # 是否自动发现模式
```

这个模型同时支持**预定义封闭schema**和**schema-free自动发现**两种模式。

### 3.2 两种Schema模式对比

| 模式 | 适用场景 | 优势 |
|------|---------|------|
| **schema-free** | 事前不知道语料内容 | LLM自动设计适合的类型，零配置上手 |
| **预定义schema** | 已有分类体系 | 结果可预测，类型一致 |

sift-kg 默认 schema-free 降低了入门门槛，这个设计选择很好。如果用户有明确需求，也支持预定义。

### 3.3 Schema-free 自动发现工作流

1. 第一次运行时，LLM从5个文档样本分析内容
2. LLM设计适合该语料的实体类型和关系类型
3. 保存到 `output/discovered_domain.yaml`
4. 后续运行重用发现的schema，保持一致性
5. 用户可以手动编辑调整

这大大降低了使用门槛，推荐实现。

### 3.4 预打包域

sift-kg 内置几个常用域供用户直接使用：

- `general` - 通用文档分析（PERSON, ORGANIZATION, LOCATION...）
- `osint` - 调查和开源情报（SHELL_COMPANY, FINANCIAL_ACCOUNT...）
- `academic` - 文献综述（CONCEPT, THEORY, METHOD, FINDING...）

这个特性很实用，用户直接用不用自己定义。你可以根据自己的目标用户群体打包相应的域。

## 4. 抽取模块建议

### 4.1 单次调用抽取实体+关系

sift-kg 在**一次LLM调用**中同时抽取实体和关系，而不是分开两次调用。

**权衡**:
- 优点：节省一半成本和延迟
- 缺点：理论上可能比分两次略不准确
- 实践：现代LLM（GPT-4o-mini等）足够强大，一次调用效果很好

对于大多数应用，这个权衡很值得。推荐这个设计。

### 4.2 异步并发设计

- 使用 `asyncio.Semaphore` 控制并发数（默认4）
- 所有块跨文档共享同一个信号量
- 避免并发溢出，平衡速度和速率限制

这个设计很好，比"一个文档一个并发"更高效利用并发槽。

### 4.3 增量处理

检查现有提取结果是否过期，只有以下变化才重新提取：
- 文档内容变化
- 模型变化
- 域配置变化
- 分块大小变化

这大幅降低重复运行成本，支持知识库增量增长。一定要做。

### 4.4 文档上下文

对每个文档，用第一个块生成2-3句文档摘要，注入到后续所有块的prompt中，帮助LLM理解全局上下文。这个小技巧提升抽取质量很有帮助。

### 4.5 LLM客户端封装

- 封装 LiteLLM 统一接口
- 支持所有LiteLLM兼容的提供商
- 跟踪总美元成本，便于预算控制
- JSON解析失败处理回退

使用 LiteLLM 省去了你自己对接多个提供商的麻烦，推荐依赖它。

## 5. 知识图谱核心建议

### 5.1 数据模型设计

节点（实体）属性：
- `id`: `type:name` 格式
- `entity_type`: 实体类型
- `name`: 显示名称
- `confidence`: 置信度
- `source_documents`: 来源文档列表
- `attributes`: 其他属性

边（关系）属性：
- `source`: 源实体ID
- `target`: 目标实体ID
- `relation_type`: 关系类型
- `confidence`: 置信度
- `evidence`: 证据文本
- `source_document`: 来源文档
- `support_count`: 提及次数

使用 NetworkX `MultiDiGraph` 支持多边有向图，同一个实体对之间可以有多种不同类型的关系。

### 5.2 边方向修复

LLM 有时会颠倒源/目标方向。sift-kg 在构建图谱时根据域定义的 `source_types/target_types` 检查并修复方向。这是一个很重要的后处理步骤，提升质量。

### 5.3 社区检测

使用 Louvain 算法检测社区：
- 每个实体分配到一个社区
- 可视化中用不同颜色区分
- 叙事生成中按社区分组
- 帮助理解知识拓扑结构

这个功能成本低（NetworkX内置），收益大，一定要加上。

## 6. 实体消歧模块建议

### 6.1 三层架构回顾

再强调一次，这个设计非常棒：

1. **预去重**（构建时自动）→ 免费处理明显重复
2. **LLM建议** → 批量比较提出合并候选
3. **人工审核** → 用户做最后决定，从不自动合并

### 6.2 可选嵌入聚类

默认按字母顺序分批，问题是 "Robert Smith" 和 "Bob Smith" 可能在不同批次，永远不会被比较。

可选方案：KMeans 聚类对实体名称嵌入，语义相似聚在一起。解决拼写差异大但指代相同的问题。依赖 `sentence-transformers`，可选安装，不强制。

这个是锦上添花的功能，用户可以根据精度需求选择是否启用。

### 6.3 审核工作流

两种审核方式：

**A. 交互式终端审核**:
- 逐个展示DRAFT建议
- 高置信度自动批准（可配置阈值）
- 低置信度自动拒绝（可配置阈值）
- 适合快速审核

**B. 直接编辑 YAML**:
- 打开 `merge_proposals.yaml` 手动改状态
- `DRAFT`/`CONFIRMED`/`REJECTED`
- 适合高精度需求（法律、族谱）

两种方式都支持，满足不同场景，这个设计很好。

### 6.4 图手术引擎

应用合并时要做：

1. 合并成员节点数据到规范节点（保留来源文档）
2. 重写所有指向/来自成员的边到规范节点
3. 移除成员节点
4. 移除合并产生的自环

每一步都要处理清楚，否则图会留下脏数据。

### 6.5 增量设计

- 保留之前审核决策
- 每次运行只添加新建议
- 可以多次迭代，逐渐完善

允许用户随着对领域理解加深，不断改进图谱，这个很重要。

---

**下一页**: [基于sift-kg构建知识图谱-03工程实践建议.md](基于sift-kg构建知识图谱-03工程实践建议.md)
