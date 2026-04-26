from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import AppSettings
from app.service import build_services


APP_CACHE_VERSION = "2026-04-26-delete-doc-v2"

st.set_page_config(page_title="日本大学募集要项 RAG 助手", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def load_app(cache_version: str):
    _ = cache_version
    settings = AppSettings.from_env()
    return settings, *build_services(settings)


def _save_uploads(target_dir: Path, uploaded_files) -> int:
    count = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        destination = target_dir / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        count += 1
    return count


def render_sidebar(settings: AppSettings, ingestion, keyword_store) -> None:
    st.sidebar.header("PDF 导入 / 索引管理")

    uploaded = st.sidebar.file_uploader("上传募集要项 PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.sidebar.button("保存上传文件", use_container_width=True):
        count = _save_uploads(settings.pdf_dir, uploaded)
        st.sidebar.success(f"已保存 {count} 个 PDF 到 {settings.pdf_dir}")

    if st.sidebar.button("重建本地索引", use_container_width=True):
        with st.spinner("正在解析 PDF、切块、embedding 并写入索引..."):
            stats = ingestion.rebuild_from_directory(settings.pdf_dir)
        st.sidebar.success(
            f"完成：{stats.pdf_count} 份 PDF / {stats.page_count} 页 / {stats.chunk_count} 个 chunk / OCR {stats.ocr_pages} 页"
        )

    st.sidebar.divider()
    st.sidebar.subheader("当前已索引文档")
    documents = keyword_store.list_documents()
    if not documents:
        st.sidebar.caption("还没有索引内容。")
        return

    for item in documents:
        label = f"{item['pdf_name']} ({item['chunk_count']} chunks)"
        meta = " / ".join(part for part in [item.get("university", ""), item.get("year", "")] if part)
        st.sidebar.write(label)
        if meta:
            st.sidebar.caption(meta)

    st.sidebar.divider()
    st.sidebar.subheader("删除已索引文档")
    options = [item["pdf_name"] for item in documents]
    selected_pdf = st.sidebar.selectbox("选择要从索引移除的 PDF", options=options)

    if st.sidebar.button("从索引移除所选文档", type="secondary", use_container_width=True):
        if not hasattr(ingestion, "remove_document"):
            st.cache_resource.clear()
            st.rerun()
        removed_chunks = ingestion.remove_document(selected_pdf)
        if removed_chunks:
            st.sidebar.success(f"已从索引移除 {selected_pdf}，共删除 {removed_chunks} 个 chunk。")
            st.rerun()
        else:
            st.sidebar.warning(f"{selected_pdf} 当前不在索引中。")

    st.sidebar.caption("说明：这里只会从索引中移除，不会删除 data/pdfs 里的原 PDF 文件。")


def render_chat(assistant) -> None:
    st.title("日本大学募集要项中文问答")
    st.caption("输入中文问题，系统会检索日文募集要项，并用中文回答，同时附上出处页码。")

    question = st.text_input("请输入问题", placeholder="例如：东京大学大学院的出愿时间是什么时候？")
    if not question:
        return

    if st.button("开始检索并回答", type="primary", use_container_width=True):
        with st.spinner("正在改写问题、混合检索、重排并生成答案..."):
            response = assistant.answer(question)

        st.subheader("中文答案")
        st.write(response.answer_zh)

        st.subheader("检索改写")
        st.json(
            {
                "original_zh": response.rewritten_query.original_zh,
                "rewritten_ja": response.rewritten_query.rewritten_ja,
                "expanded_keywords": response.rewritten_query.expanded_keywords,
                "filters": response.rewritten_query.filters,
            }
        )

        st.subheader("证据卡片")
        if not response.citations:
            st.info("没有找到足够证据。")
        for index, item in enumerate(response.retrieved, start=1):
            with st.container(border=True):
                st.markdown(f"**TOP {index}** | `{item.chunk.metadata.pdf_name}` | 第 `{item.chunk.metadata.page}` 页")
                if item.chunk.metadata.section_title:
                    st.caption(item.chunk.metadata.section_title)
                st.write(item.chunk.text_ja)
                score_parts = [
                    f"bm25={item.bm25_score:.4f}" if item.bm25_score is not None else "",
                    f"dense={item.dense_score:.4f}" if item.dense_score is not None else "",
                    f"rrf={item.rrf_score:.4f}" if item.rrf_score is not None else "",
                    f"rerank={item.rerank_score:.4f}" if item.rerank_score is not None else "",
                ]
                st.caption(" / ".join(part for part in score_parts if part))


def main() -> None:
    settings, ingestion, assistant, keyword_store = load_app(APP_CACHE_VERSION)
    render_sidebar(settings, ingestion, keyword_store)
    render_chat(assistant)


if __name__ == "__main__":
    main()
