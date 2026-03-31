# 基于 sift-kg 借鉴实现知识图谱构建系统 - 工程实践建议

**返回主目录**: [设计文档.md](设计文档.md)  
**上一页**: [基于sift-kg构建知识图谱-02模块级建议.md](基于sift-kg构建知识图谱-02模块级建议.md)

---

## 目录

- [1. 依赖选择建议](#1-依赖选择建议)
- [2. 扩展性设计](#2-扩展性设计)
- [3. 不同场景下的定制建议](#3-不同场景下的定制建议)
- [4. 成本控制建议](#4-成本控制建议)
- [5. AI Agent 集成建议](#5-ai-agent-集成建议)
- [6. 常见陷阱避免](#6-常见陷阱避免)

---

## 1. 依赖选择建议

### 1.1 核心依赖推荐

sift-kg 选择的这些依赖都经过实践检验，推荐直接使用：

| 依赖 | 用途 | 评价 |
|------|------|------|
| `typer[all]` | CLI 框架 | 最好用的 Python CLI 框架，自动帮助，补全 |
| `pydantic` + `pydantic-settings` | 数据验证 + 配置管理 | 行业标准，类型安全 |
| `networkx` | 图数据结构和算法 | Python 生态最好，稳定，文档全 |
| `litellm` | LLM 多提供商统一接口 | 支持几乎所有提供商，不用自己对接 |
| `kreuzberg` | 多格式文本提取 | 支持75+格式，比 textract 维护好 |
| `semhash` | 模糊字符串匹配去重 | 预去重效果好 |
| `pyvis` | 交互式可视化 | 简单，开箱即用，生成单独 HTML |

### 1.2 可选依赖

```toml
# optional dependencies
[project.optional-dependencies]
embeddings = ["sentence-transformers", "scikit-learn"]
ocr = ["google-cloud-vision"]
```

把重量级可选依赖（比如 `sentence-transformers` 要拉 PyTorch）做成可选，用户不用可以不装。这能减小基础安装包体积。

## 2. 扩展性设计

sift-kg 的扩展性设计做得很好，推荐借鉴：

### 2.1 添加新的文本提取后端

1. 在 `ingest/base.py` 实现 `TextExtractor` 协议
2. 在 `reader.py` `read_document()` 添加选择逻辑
3. 更新配置验证

不需要修改其他代码，符合开放封闭原则。

### 2.2 添加新的 OCR 后端

1. 在 `ocr.py` 添加后端实现
2. 更新有效后端列表
3. 在 `reader.py` 传递参数

### 2.3 添加新的导出格式

1. 在 `export.py` 添加新的导出函数
2. 在 `cli.py` `export` 命令添加格式选项
3. 通常只需要几十行代码（利用 NetworkX 内置导出）

### 2.4 自定义域

**不需要修改代码**，用户可以自己创建 `domain.yaml`:

```yaml
name: my-custom-domain
fallback_relation: RELATED_TO
entity_types:
  MY_TYPE:
    description: Description of my type
    extraction_hints:
      - Look for these entities...
relation_types:
  MY_RELATION:
    description: Description of relation
    source_types: [MY_TYPE]
    target_types: [OTHER_TYPE]
```

然后使用：`sift extract ./docs --domain path/to/domain.yaml`

让用户不用改代码就能定制，这大大提升了工具的适用性。

## 3. 不同场景下的定制建议

### 3.1 个人笔记 / AI 第二大脑

sift-kg 本身设计就适合这个场景。重点关注：

- 保持默认 schema-free 让 LLM 自动发现
- 增量处理支持不断添加新笔记
- 导出 JSON 供 AI Agent 查询
- 推荐借鉴 sift-kg 在 `.agents/skills/sift-kg` 中的集成方式

### 3.2 法律/调查工作

需要更高准确性：

- 启用 `--embeddings` 聚类改进建议质量
- 建议用户手动编辑 YAML 审核每个合并
- 设置 `review_required: true` 在关键关系类型上
- 保留完整证据来源链接到原文段落

### 3.3 学术文献综述

使用内置 `academic` 域：

- 实体类型：CONCEPT, THEORY, METHOD, SYSTEM, FINDING, RESEARCHER, PUBLICATION
- 关系类型：SUPPORTS, CONTRADICTS, EXTENDS, IMPLEMENTS, USES_METHOD 等
- 社区检测能帮你看到研究课题分组
- 叙事生成按社区输出概念地图说明

### 3.4 企业级多用户系统

sift-kg 设计是单用户 CLI，你需要改造：

- 替换 NetworkX+JSON → Neo4j/PostgreSQL 存储
- 添加用户认证和权限控制
- 保留流水线架构和人机协作设计
- 可能需要异步任务队列处理大文档集合

### 3.5 纯中文知识图谱

定制点：

- OCR 选择 `paddleocr` 后端，中文识别更好
- 预去重需要适配中文（头衔前缀不同）
- LLM 选择中文能力强的模型
- domain 配置用中文描述，LLM 理解更好

## 4. 成本控制建议

LLM 调用花钱，sift-kg 有这些控制措施：

1. **增量缓存** → 只重新提取修改过的文档
2. **一次调用抽取实体+关系** → 节省一半成本
3. **成本跟踪** → 每个步骤显示花费
4. `--max-cost` 选项 → 超过预算停止
5. **schema 缓存** → schema-free 发现一次后缓存重用

这些都要借鉴，用户对成本很敏感。

**推荐实践**:
- 开始先用便宜模型（比如 GPT-4o-mini）探索
- 高置信度自动批准减少审核工作量
- 只对低置信度要求人工审核

## 5. AI Agent 集成建议

sift-kg 天生设计为 AI Agent 的结构化记忆，这里是集成要点：

### 5.1 能给 Agent 提供什么？

- **结构** → 不是文本块，是实体、关系、社区
- **拓扑** → 哪些知识聚在一起，哪些是桥接
- **持久性** → 上下文窗口重置后图谱还在

### 5.2 提供什么 API 给 Agent？

```bash
sift topology          # 输出结构概览 JSON
sift query "entity"    # 查询实体邻域 JSON
sift search "X" --json # 实体查找 JSON
sift info --json       # 项目统计
```

Agent 可以通过这些命令探索图谱，获取结构化知识，不用自己解析整个图谱。

### 5.3 启发式搜索

sift-kg 的技能模块（`.agents/skills/sift-kg`）教 Agent：

1. 从问题中识别关键实体
2. 查询邻域得到相关概念
3. 沿着关系链发现连接
4. 基于图谱结构综合答案

这种方式比纯向量检索能找到更深入的间接连接。

## 6. 常见陷阱避免

### 陷阱 1: 过度自动化

**误区**: 让LLM自动合并所有实体，不人工审核  
**后果**: 错误传播污染整个图谱，很难修复  
**sift-kg方案**: LLM只提建议，用户做最后决定

### 陷阱 2: 强迫用户事前定义完整Schema

**误区**: 必须先定义所有实体类型才能开始  
**后果**: 门槛太高，用户不知道语料有什么，很难定义  
**sift-kg方案**: 默认schema-free，LLM自动发现，用户可事后编辑

### 陷阱 3: 不支持增量更新

**误区**: 加新文档必须全量重新抽取  
**后果**: 随着知识库增大，成本越来越高  
**sift-kg方案**: 缓存抽取结果，只重新提取修改过的文档

### 陷阱 4: 忽略概率聚合

**误区**: 同一关系多次提及只保留最高置信度  
**后果**: 多个独立弱信号被忽略  
**sift-kg方案**: product_complement 聚合，独立证据互相增强

### 陷阱 5: 不对LLM输出做后处理

**误区**: LLM抽出来什么就直接用  
**后果**: 边方向颠倒，类型不对，低质量关系污染图谱  
**sift-kg方案**: 后处理修复方向，标记低置信度关系供审核

### 陷阱 6: 没有来源追溯

**误区**: 只存实体关系，不存哪里抽出来的  
**后果**: 用户无法验证证据，错误没法修正  
**sift-kg方案**: 每个抽取都链接到源文档和具体段落

## 总结

sift-kg 给出了一个**完整的、可运行的、零配置**知识图谱构建管道参考实现。它的核心贡献不是算法，而是**把各个模块正确组装起来，解决了很多工程细节问题**，这些都是你不用再重新摸索的：

- 配置优先级设计合理
- 多格式提取+OCR抽象成熟
- schema-free 和预定义schema都支持
- 异步并发LLM调用控制得当
- 三层去重人机协作平衡了成本和准确性
- 增量处理支持知识库持续增长
- 导出多种标准格式便于进一步分析
- AI Agent 集成友好，可作为第二大脑

你借鉴的时候，保留整体架构，根据自己的应用场景定制特定模块即可。祝你项目顺利！

---

**文档完**
