"""从文档文本中提取实体和关系。

编排完整的提取流水线：文本分块 → 提取实体 →
提取关系 → 合并分块结果 → 持久化到磁盘。

使用异步并发处理多个分块并行处理。
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

# 导入rich库用于进度条显示
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from sift_kg.domains.models import DomainConfig
from sift_kg.extract.llm_client import LLMClient
from sift_kg.extract.models import (
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from sift_kg.extract.prompts import build_combined_prompt
from sift_kg.ingest.chunker import TextChunk, chunk_text
from sift_kg.ingest.reader import read_document

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 默认并发LLM调用数。平衡速度与限流风险。
DEFAULT_CONCURRENCY = 4


def _check_stale(
    existing: DocumentExtraction,
    model: str,
    domain_name: str,
    chunk_size: int,
) -> str | None:
    """检查已存在的提取结果是否已过期。如果过期返回原因字符串，否则返回None。"""
    # 没有元数据 = 增量支持之前的旧提取，需要重新提取
    if not existing.domain_name and not existing.chunk_size:
        return "missing metadata (pre-incremental extraction)"
    if existing.model_used != model:
        return f"model changed ({existing.model_used} → {model})"
    if existing.domain_name != domain_name:
        return f"domain changed ({existing.domain_name} → {domain_name})"
    if existing.chunk_size != chunk_size:
        return f"chunk size changed ({existing.chunk_size} → {chunk_size})"
    # 没有变化，不需要重新提取
    return None


def extract_from_text(
    text: str,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    chunk_size: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
    output_dir: Path | None = None,
    force: bool = False,
) -> DocumentExtraction:
    """从文档文本提取实体和关系。

    纯逻辑 — 无文件I/O。不需要文件测试夹具也可以测试。
    内部运行异步提取以实现并发。
    """
    return asyncio.run(
        _aextract_from_text(
            text,
            doc_id,
            llm,
            domain,
            chunk_size,
            concurrency,
            output_dir=output_dir,
            force=force,
        )
    )


async def _generate_doc_context(llm: LLMClient, first_chunk_text: str) -> str:
    """从第一个分块生成简短的文档摘要。

    每个文档调用一次LLM — 为后续每个分块提供整个文档的上下文，
    帮助LLM理解整个文档是什么（谁在发言，什么案件，什么主题）。
    """
    prompt = (
        "Summarize this document excerpt in 2-3 sentences. "
        "Focus on: what type of document this is, who the key participants are, "
        "and what the main subject matter is. Be specific about names and roles.\n\n"
        f"TEXT:\n{first_chunk_text}\n\nSUMMARY:"
    )
    try:
        # 调用异步LLM生成摘要
        response = await llm.acall(prompt)
        return response.strip()
    except Exception as e:
        # 生成失败不阻塞流程，返回空字符串
        logger.warning(f"文档上下文生成失败: {e}")
        return ""


async def _aextract_from_text(
    text: str,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    chunk_size: int,
    concurrency: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> DocumentExtraction:
    """异步提取 — 并发处理多个文本分块。"""
    # 将文本分块
    chunks = chunk_text(text, chunk_size=chunk_size)
    # 记录调用前成本，计算本次文档成本增量
    cost_before = llm.total_cost_usd
    # 使用信号量控制并发数
    sem = asyncio.Semaphore(concurrency)

    # 无模式域的模式发现
    if domain.schema_free and output_dir is not None:
        # 延迟导入发现模块，避免不必要的依赖
        from sift_kg.domains.discovery import (
            discover_domain,
            load_discovered_domain,
            save_discovered_domain,
        )

        discovered_path = output_dir / "discovered_domain.yaml"
        # 尝试加载已缓存的发现结果
        cached = load_discovered_domain(discovered_path)
        if cached is not None and not force:
            logger.info(f"使用缓存的已发现模式 ({len(cached.entity_types)} 实体类型)")
            domain = cached
        else:
            # 从第一个分块取样进行发现
            samples = [chunks[0].text[:3000]]
            try:
                domain = await discover_domain(samples, llm, domain.system_context or "")
                save_discovered_domain(domain, discovered_path)
                logger.info(f"已发现模式: {len(domain.entity_types)} 实体类型, {len(domain.relation_types)} 关系类型")
            except (RuntimeError, ValueError) as e:
                logger.warning(f"模式发现失败，回退到无模式提取: {e}")

    # 从第一个分块生成文档级上下文
    doc_context = await _generate_doc_context(llm, chunks[0].text)
    if doc_context:
        logger.debug(f"文档 {doc_id} 上下文: {doc_context[:100]}...")

    # 带并发限制的提取函数
    async def _bounded(chunk: TextChunk) -> ExtractionResult:
        async with sem:
            return await _aextract_chunk(chunk, doc_id, llm, domain, doc_context)

    # 并发提取所有分块
    results = await asyncio.gather(*[_bounded(c) for c in chunks])

    # 收集所有分块的实体和关系
    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []
    for result in results:
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)

    # 对实体去重
    unique_entities = _dedupe_entities(all_entities)

    # 返回整个文档的提取结果
    return DocumentExtraction(
        document_id=doc_id,
        document_path="",
        chunks_processed=len(chunks),
        entities=unique_entities,
        relations=all_relations,
        cost_usd=llm.total_cost_usd - cost_before,
        model_used=llm.model,
    )

        discovered_path = output_dir / "discovered_domain.yaml"
        cached = load_discovered_domain(discovered_path)
        if cached is not None and not force:
            logger.info(f"Using cached discovered schema ({len(cached.entity_types)} entity types)")
            domain = cached
        else:
            samples = [chunks[0].text[:3000]]
            try:
                domain = await discover_domain(samples, llm, domain.system_context or "")
                save_discovered_domain(domain, discovered_path)
                logger.info(
                    f"Discovered schema: {len(domain.entity_types)} entity types, {len(domain.relation_types)} relation types"
                )
            except (RuntimeError, ValueError) as e:
                logger.warning(
                    f"Schema discovery failed, falling back to schema-free extraction: {e}"
                )

    # Generate document-level context from the first chunk
    doc_context = await _generate_doc_context(llm, chunks[0].text)
    if doc_context:
        logger.debug(f"Doc context for {doc_id}: {doc_context[:100]}...")

    async def _bounded(chunk: TextChunk) -> ExtractionResult:
        async with sem:
            return await _aextract_chunk(chunk, doc_id, llm, domain, doc_context)

    results = await asyncio.gather(*[_bounded(c) for c in chunks])

    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []
    for result in results:
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)

    unique_entities = _dedupe_entities(all_entities)

    return DocumentExtraction(
        document_id=doc_id,
        document_path="",
        chunks_processed=len(chunks),
        entities=unique_entities,
        relations=all_relations,
        cost_usd=llm.total_cost_usd - cost_before,
        model_used=llm.model,
    )


def extract_document(
    doc_path: Path,
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    chunk_size: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    ocr: bool = False,
    backend: str = "kreuzberg",
    ocr_backend: str = "tesseract",
    ocr_language: str = "eng",
) -> DocumentExtraction:
    """从单个文档文件提取实体和关系。

    读取文件，调用extract_from_text，将结果保存到磁盘。
    支持增量提取 — 如果提取配置匹配则跳过文档。
    使用force=True强制重新提取，无论配置是否变化。
    """
    # 使用文件名作为文档ID
    doc_id = doc_path.stem
    # 构建结果文件路径
    extraction_path = output_dir / "extractions" / f"{doc_id}.json"

    if extraction_path.exists() and not force:
        # 已存在提取结果且不强制重新提取，检查是否过期
        existing = DocumentExtraction(**json.loads(extraction_path.read_text()))
        reason = _check_stale(existing, llm.model, domain.name, chunk_size)
        if reason is None:
            logger.info(f"跳过 {doc_id} (已提取)")
            return existing
        logger.info(f"重新提取 {doc_id}: {reason}")

    logger.info(f"正在提取: {doc_path.name}")

    try:
        # 读取文档文本，支持多种格式和OCR
        text = read_document(
            doc_path, ocr=ocr, backend=backend,
            ocr_backend=ocr_backend, ocr_language=ocr_language,
        )
    except Exception as e:
        # 读取失败，返回带错误信息的结果
        logger.error(f"读取 {doc_path.name} 失败: {e}")
        return DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            error=str(e),
            model_used=llm.model,
        )

    if not text.strip():
        # 文档为空，返回错误
        logger.warning(f"从 {doc_path.name} 获取到空文本")
        return DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            error="Empty document",
            model_used=llm.model,
        )

    # 执行提取
    extraction = extract_from_text(
        text, doc_id, llm, domain, chunk_size, concurrency,
        output_dir=output_dir, force=force,
    )
    # 添加元数据
    extraction.document_path = str(doc_path)
    extraction.domain_name = domain.name
    extraction.chunk_size = chunk_size
    extraction.extracted_at = datetime.now(UTC).isoformat()

    # 确保目录存在
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存提取结果为JSON
    extraction_path.write_text(extraction.model_dump_json(indent=2))

    logger.info(
        f"  {doc_id}: {len(extraction.entities)} 实体, "
        f"{len(extraction.relations)} 关系 ({extraction.chunks_processed} 分块)"
    )
    return extraction


async def _aextract_chunk(
    chunk: TextChunk,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    doc_context: str = "",
) -> ExtractionResult:
    """从单个文本分块提取实体和关系（异步）。

    使用组合提示（1次LLM调用）而不是分别调用实体+关系（2次LLM调用）。
    如果解析失败，返回空结果。
    """
    # 根据域配置构建组合提示
    prompt = build_combined_prompt(chunk.text, doc_id, domain, doc_context=doc_context)
    try:
        # 异步调用LLM并解析JSON
        data = await llm.acall_json(prompt)
    except (RuntimeError, ValueError) as e:
        # 提取失败，记录警告并返回空结果
        logger.warning(f"提取失败 {doc_id} 分块 {chunk.chunk_index}: {e}")
        return ExtractionResult(source_document=doc_id, chunk_index=chunk.chunk_index)

    # 处理提取的实体
    entities = []
    for raw in data.get("entities", []):
        try:
            entities.append(ExtractedEntity(
                name=raw.get("name", ""),
                entity_type=raw.get("entity_type", "UNKNOWN"),
                attributes=raw.get("attributes", {}),
                confidence=float(raw.get("confidence", 0.5)),
                context=raw.get("context", ""),
            ))
        except (ValueError, TypeError, KeyError) as e:
            # 格式错误的实体，跳过并调试
            logger.debug(f"跳过格式错误的实体: {e}")

    # 处理提取的关系
    relations = []
    for raw in data.get("relations", []):
        try:
            relations.append(ExtractedRelation(
                relation_type=raw.get("relation_type", "ASSOCIATED_WITH"),
                source_entity=raw.get("source_entity", ""),
                target_entity=raw.get("target_entity", ""),
                confidence=float(raw.get("confidence", 0.5)),
                evidence=raw.get("evidence", ""),
            ))
        except (ValueError, TypeError, KeyError) as e:
            # 格式错误的关系，跳过并调试
            logger.debug(f"跳过格式错误的关系: {e}")

    # 返回分块提取结果
    return ExtractionResult(
        entities=entities,
        relations=relations,
        source_document=doc_id,
        chunk_index=chunk.chunk_index,
    )


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """根据名称+类型去除文档内重复实体。

    保留置信度最高的条目，但将所有唯一上下文引用收集到
    管道分隔字符串中，以便后续生成叙述时有丰富的源材料。
    """
    # 已见过的实体，key为"类型:小写名称"
    seen: dict[str, ExtractedEntity] = {}
    # 收集所有唯一上下文
    contexts: dict[str, set[str]] = {}
    for e in entities:
        key = f"{e.entity_type}:{e.name.lower().strip()}"
        # 收集所有唯一上下文
        if e.context.strip():
            contexts.setdefault(key, set()).add(e.context.strip())
        if key in seen:
            # 如果当前实体置信度更高，替换
            if e.confidence > seen[key].confidence:
                seen[key] = e
        else:
            # 新实体，添加
            seen[key] = e

    # 将收集的上下文合并回每个实体
    for key, entity in seen.items():
        all_ctx = contexts.get(key, set())
        if len(all_ctx) > 1:
            # 多个上下文，用|||分隔
            entity.context = " ||| ".join(sorted(all_ctx))

    # 返回去重后的实体列表
    return list(seen.values())


def extract_all(
    doc_paths: list[Path],
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    chunk_size: int = 10000,
    force: bool = False,
    ocr: bool = False,
    backend: str = "kreuzberg",
    ocr_backend: str = "tesseract",
    ocr_language: str = "eng",
) -> list[DocumentExtraction]:
    """从多个文档提取实体和关系。

    所有文档的所有分块共享同一个信号量和限流器，
    因此不会在文档之间等待浪费并发slot。
    """
    return asyncio.run(
        _aextract_all(
            doc_paths, llm, domain, output_dir, max_cost, concurrency,
            chunk_size, force, ocr=ocr, backend=backend,
            ocr_backend=ocr_backend, ocr_language=ocr_language,
        )
    )
    )


async def _aextract_all(
    doc_paths: list[Path],
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None,
    concurrency: int,
    chunk_size: int,
    force: bool = False,
    ocr: bool = False,
    backend: str = "kreuzberg",
    ocr_backend: str = "tesseract",
    ocr_language: str = "eng",
) -> list[DocumentExtraction]:
    """跨所有文档异步提取，共享并发限制。"""
    # 信号量控制全局并发数
    sem = asyncio.Semaphore(concurrency)
    extraction_dir = output_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # 在过期检查之前加载缓存发现的模式，这样domain_name匹配
    if domain.schema_free:
        from sift_kg.domains.discovery import load_discovered_domain

        discovered_path = output_dir / "discovered_domain.yaml"
        cached_domain = load_discovered_domain(discovered_path)
        if cached_domain is not None and not force:
            logger.info(f"使用缓存的已发现模式 ({len(cached_domain.entity_types)} 实体类型)")
            domain = cached_domain

    # 提前读取所有文档并准备分块（便宜，无LLM调用）
    doc_work: list[tuple[Path, str, str, list[TextChunk]]] = []
    # 已缓存（跳过）的提取结果列表
    cached: list[DocumentExtraction] = []

    for doc_path in doc_paths:
        doc_id = doc_path.stem
        extraction_path = extraction_dir / f"{doc_id}.json"

        if extraction_path.exists() and not force:
            # 已存在提取结果且不强制重新提取，检查是否过期
            existing = DocumentExtraction(**json.loads(extraction_path.read_text()))
            reason = _check_stale(existing, llm.model, domain.name, chunk_size)
            if reason is None:
                logger.info(f"跳过 {doc_id} (已提取)")
                cached.append(existing)
                continue
            logger.info(f"重新提取 {doc_id}: {reason}")

        logger.info(f"正在读取 {doc_path.name}...")
        try:
            # 读取文档文本
            text = read_document(
                doc_path, ocr=ocr, backend=backend,
                ocr_backend=ocr_backend, ocr_language=ocr_language,
            )
        except Exception as e:
            # 读取失败，添加到缓存结果列表
            logger.error(f"读取 {doc_path.name} 失败: {e}")
            cached.append(DocumentExtraction(
                document_id=doc_id, document_path=str(doc_path),
                error=str(e), model_used=llm.model,
            ))
            continue

        if not text.strip():
            # 文本为空，添加错误结果
            logger.warning(f"从 {doc_path.name} 获取到空文本")
            cached.append(DocumentExtraction(
                document_id=doc_id, document_path=str(doc_path),
                error="Empty document", model_used=llm.model,
            ))
            continue

        # 分块
        chunks = chunk_text(text, chunk_size=chunk_size)
        logger.info(f"  {len(text):,} 字符 → {len(chunks)} 分块")
        doc_work.append((doc_path, doc_id, text, chunks))

    if not doc_work:
        # 没有需要处理的文档，直接返回缓存结果
        return cached

    # 模式发现 — 只有当上面没有加载到缓存域时才运行
    if domain.schema_free:
        from sift_kg.domains.discovery import (
            discover_domain,
            save_discovered_domain,
        )

        discovered_path = output_dir / "discovered_domain.yaml"
        # 从前5个文档的第一个分块取样
        samples = [chunks[0].text[:3000] for _, _, _, chunks in doc_work[:5]]
        try:
            domain = await discover_domain(samples, llm, domain.system_context or "")
            save_discovered_domain(domain, discovered_path)
            logger.info(f"已发现模式: {len(domain.entity_types)} 实体类型, {len(domain.relation_types)} 关系类型")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"模式发现失败，回退到无模式提取: {e}")

    # 为每个文档生成文档级上下文（每个文档1次LLM调用）
    doc_contexts: dict[str, str] = {}
    for _, doc_id, _, chunks in doc_work:
        logger.info(f"正在为 {doc_id} 生成上下文...")
        ctx = await _generate_doc_context(llm, chunks[0].text)
        doc_contexts[doc_id] = ctx
        if ctx:
            logger.info(f"  {ctx[:120]}")

    # 将所有文档的所有分块展平，带上文档ID标记
    all_tasks: list[tuple[str, TextChunk]] = []
    for _, doc_id, _, chunks in doc_work:
        for chunk in chunks:
            all_tasks.append((doc_id, chunk))

    total_chunks = len(all_tasks)
    completed_chunks = 0

    # 使用rich创建进度条，显示当前成本
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("${task.fields[cost]:.3f}"),
    ) as progress:
        ptask = progress.add_task("提取中...", total=total_chunks, cost=0.0)

        async def _bounded(doc_id: str, chunk: TextChunk) -> tuple[str, ExtractionResult]:
            nonlocal completed_chunks
            # 如果已达到最大成本预算，跳过处理
            if max_cost and llm.total_cost_usd >= max_cost:
                return doc_id, ExtractionResult(source_document=doc_id, chunk_index=chunk.chunk_index)
            async with sem:
                result = await _aextract_chunk(
                    chunk, doc_id, llm, domain, doc_context=doc_contexts.get(doc_id, "")
                )
                completed_chunks += 1
                # 更新进度条和当前成本
                progress.update(ptask, completed=completed_chunks, cost=llm.total_cost_usd)
                return doc_id, result

        # 并发执行所有分块提取任务
        chunk_results = await asyncio.gather(
            *[_bounded(doc_id, chunk) for doc_id, chunk in all_tasks]
        )

    # 按文档分组结果
    doc_results: dict[str, list[ExtractionResult]] = {}
    for doc_id, result in chunk_results:
        doc_results.setdefault(doc_id, []).append(result)

    # 为每个文档构建DocumentExtraction并保存
    extractions = list(cached)
    for doc_path, doc_id, _, chunks in doc_work:
        results = doc_results.get(doc_id, [])
        cost_for_doc = sum(
            getattr(r, '_cost', 0.0) for r in results
        )

        # 收集所有分块的实体和关系
        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []
        for r in results:
            all_entities.extend(r.entities)
            all_relations.extend(r.relations)

        # 构建提取结果
        extraction = DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            chunks_processed=len(chunks),
            entities=_dedupe_entities(all_entities),
            relations=all_relations,
            cost_usd=cost_for_doc,
            model_used=llm.model,
            domain_name=domain.name,
            chunk_size=chunk_size,
            extracted_at=datetime.now(UTC).isoformat(),
        )

        # 保存到文件
        extraction_path = extraction_dir / f"{doc_id}.json"
        extraction_path.write_text(extraction.model_dump_json(indent=2))

        logger.info(
            f"  {doc_id}: {len(extraction.entities)} 实体, "
            f"{len(extraction.relations)} 关系 ({extraction.chunks_processed} 分块)"
        )
        extractions.append(extraction)

    if max_cost and llm.total_cost_usd >= max_cost:
        # 已达到预算上限，警告用户
        logger.warning(
            f"预算上限已达到: ${llm.total_cost_usd:.2f} / ${max_cost:.2f}"
        )

    return extractions
