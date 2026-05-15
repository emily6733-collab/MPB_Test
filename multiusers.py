"""멀티유저 Supabase RAG Streamlit 앱 (users 테이블 로그인, 세션/메시지 user_id 분리).

DB 스키마: `7.MultiService/prompts/multi-users-schema.sql` 참고.
환경: Streamlit Cloud는 `st.secrets`, 로컬은 `.env` (secrets가 있으면 우선).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from postgrest.exceptions import APIError
from supabase import Client, create_client

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-ada-002"
VECTOR_DIM = 1536
BATCH_VECTOR_INSERT = 10

APP_TITLE = "기획예산처 RAG 챗봇"

load_dotenv(dotenv_path=ENV_PATH)


def apply_streamlit_secrets_overrides() -> None:
    """st.secrets 값이 있으면 os.environ을 덮어씁니다 (로컬 .env보다 우선)."""
    try:
        sec = st.secrets
    except (FileNotFoundError, RuntimeError, AttributeError):
        return
    for key in (
        "OPENAI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        try:
            if key not in sec:
                continue
            val = str(sec[key]).strip()
            if val:
                os.environ[key] = val
        except Exception:  # noqa: BLE001
            continue


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = LOG_DIR / log_name
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("multiusers")


logger = _setup_logging()

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def missing_env_keys() -> list[str]:
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SUPABASE_URL", "").strip():
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_ANON_KEY", "").strip() and not os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    ).strip():
        missing.append("SUPABASE_ANON_KEY (또는 로컬 전용 SUPABASE_SERVICE_ROLE_KEY)")
    return missing


def get_supabase_client() -> Client | None:
    url = os.getenv("SUPABASE_URL", "").strip()
    service = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    key = anon or service
    if not url or not key:
        return None
    return create_client(url, key)


def supabase_uses_service_role() -> bool:
    return not bool(os.getenv("SUPABASE_ANON_KEY", "").strip()) and bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=api_key)


def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)


def register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
    lid = login_id.strip()
    if not lid or not password:
        return False, "로그인 ID와 비밀번호를 입력하세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상으로 설정하세요."
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        sb.table("users").insert({"login_id": lid, "password_hash": hashed}).execute()
    except APIError as exc:
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            return False, "이미 사용 중인 로그인 ID입니다."
        logger.warning("회원가입 API 오류: %s", exc)
        return False, f"회원가입 실패: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("회원가입 오류: %s", exc)
        return False, f"회원가입 실패: {exc}"
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def authenticate_user(sb: Client, login_id: str, password: str) -> tuple[str | None, str]:
    lid = login_id.strip()
    if not lid or not password:
        return None, "로그인 ID와 비밀번호를 입력하세요."
    try:
        res = sb.table("users").select("id,password_hash").eq("login_id", lid).limit(1).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("로그인 조회 실패: %s", exc)
        return None, f"로그인 처리 중 오류: {exc}"
    rows = res.data or []
    if not rows:
        return None, "존재하지 않는 로그인 ID입니다."
    row = rows[0]
    stored = str(row["password_hash"])
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except ValueError:
        return None, "저장된 비밀번호 형식이 올바르지 않습니다."
    if not ok:
        return None, "비밀번호가 일치하지 않습니다."
    return str(row["id"]), ""


def verify_session_owned(sb: Client, user_id: str, session_id: str) -> bool:
    res = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def list_sessions(sb: Client, user_id: str) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def insert_session(sb: Client, user_id: str, title: str) -> str:
    row = sb.table("chat_sessions").insert({"title": title, "user_id": user_id}).execute()
    if not row.data:
        raise RuntimeError("세션 생성 응답이 비어 있습니다.")
    return str(row.data[0]["id"])


def update_session_title(sb: Client, user_id: str, session_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def delete_session(sb: Client, user_id: str, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def replace_messages(
    sb: Client, user_id: str, session_id: str, messages: list[dict[str, str]]
) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        rows.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": m["role"],
                "content": m["content"],
                "msg_index": i,
            }
        )
    if rows:
        sb.table("chat_messages").insert(rows).execute()


def load_messages(sb: Client, user_id: str, session_id: str) -> list[dict[str, str]]:
    res = (
        sb.table("chat_messages")
        .select("role,content,msg_index")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("msg_index")
        .execute()
    )
    out: list[dict[str, str]] = []
    for r in res.data or []:
        out.append({"role": str(r["role"]), "content": str(r["content"])})
    return out


def fetch_vector_filenames(sb: Client, user_id: str, session_id: str) -> list[str]:
    if not verify_session_owned(sb, user_id, session_id):
        return []
    res = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = {str(r["file_name"]) for r in (res.data or []) if r.get("file_name")}
    return sorted(names)


def insert_vector_rows(
    sb: Client,
    session_id: str,
    rows: list[dict[str, Any]],
) -> None:
    for i in range(0, len(rows), BATCH_VECTOR_INSERT):
        batch = rows[i : i + BATCH_VECTOR_INSERT]
        sb.table("vector_documents").insert(batch).execute()


def copy_vectors_between_sessions(
    sb: Client, user_id: str, source_session_id: str, target_session_id: str
) -> None:
    if not verify_session_owned(sb, user_id, source_session_id):
        raise PermissionError("원본 세션에 접근할 수 없습니다.")
    if not verify_session_owned(sb, user_id, target_session_id):
        raise PermissionError("대상 세션에 접근할 수 없습니다.")
    page_size = 200
    offset = 0
    while True:
        res = (
            sb.table("vector_documents")
            .select("file_name,content,metadata,embedding")
            .eq("session_id", source_session_id)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        chunk = res.data or []
        if not chunk:
            break
        new_rows: list[dict[str, Any]] = []
        for r in chunk:
            meta = r.get("metadata") or {}
            if isinstance(meta, dict):
                meta = dict(meta)
                meta["session_id"] = target_session_id
            emb = r.get("embedding")
            if emb is None:
                continue
            new_rows.append(
                {
                    "session_id": target_session_id,
                    "file_name": str(r["file_name"]),
                    "content": str(r["content"]),
                    "metadata": meta,
                    "embedding": emb,
                }
            )
        if new_rows:
            insert_vector_rows(sb, target_session_id, new_rows)
        if len(chunk) < page_size:
            break
        offset += page_size


def retrieve_by_rpc(
    sb: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 10,
) -> list[Document]:
    qvec = embeddings.embed_query(query)
    if len(qvec) != VECTOR_DIM:
        raise ValueError(f"임베딩 차원 불일치: 기대 {VECTOR_DIM}, 실제 {len(qvec)}")

    resp = sb.rpc(
        "match_vector_documents",
        {
            "query_embedding": qvec,
            "match_count": k,
            "filter_session_id": session_id,
        },
    ).execute()

    docs: list[Document] = []
    for row in resp.data or []:
        docs.append(
            Document(
                page_content=str(row.get("content", "")),
                metadata={"id": row.get("id"), **(row.get("metadata") or {})},
            )
        )
    return docs


def retrieve_fallback(
    sb: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 10,
) -> list[Document]:
    res = (
        sb.table("vector_documents")
        .select("id,file_name,content,metadata,embedding")
        .eq("session_id", session_id)
        .limit(120)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return []
    qvec = embeddings.embed_query(query)

    def cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        emb = r.get("embedding")
        if not isinstance(emb, list):
            continue
        try:
            s = cos_sim(qvec, [float(x) for x in emb])
        except Exception:
            continue
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    docs: list[Document] = []
    for _, r in scored[:k]:
        docs.append(
            Document(
                page_content=str(r.get("content", "")),
                metadata={
                    "id": r.get("id"),
                    "file_name": r.get("file_name"),
                    **(r.get("metadata") or {}),
                },
            )
        )
    return docs


def retrieve_documents(
    sb: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 10,
) -> list[Document]:
    try:
        return retrieve_by_rpc(sb, session_id, query, embeddings, k=k)
    except (APIError, Exception) as exc:  # noqa: BLE001
        logger.warning("RPC 검색 실패, 대체 검색 사용: %s", exc)
        try:
            return retrieve_fallback(sb, session_id, query, embeddings, k=k)
        except Exception as exc2:  # noqa: BLE001
            logger.warning("대체 검색 실패: %s", exc2)
            return []


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str, context: str, memory_text: str
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def generate_session_title(llm: ChatOpenAI, q1: str, a1: str) -> str:
    trimmed_q = q1[:1200]
    trimmed_a = a1[:2000]
    prompt = (
        "다음은 채팅 세션의 첫 사용자 질문과 첫 어시스턴트 답변입니다.\n"
        "이 대화를 한 줄로 요약한 **세션 제목**만 출력하세요. "
        "따옴표나 설명 없이 제목 텍스트만, 40자 이내 한국어로.\n\n"
        f"[질문]\n{trimmed_q}\n\n[답변]\n{trimmed_a}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
        raw = remove_separators(str(raw).strip())
        return raw[:200] if raw else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("세션 제목 생성 실패: %s", exc)
        return "새 세션"


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("후속 질문 생성 실패: %s", exc)
        return ""
    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def process_pdfs_to_supabase(
    sb: Client,
    session_id: str,
    uploaded_files: list[Any],
    embeddings: OpenAIEmbeddings,
) -> list[str]:
    if not uploaded_files:
        return []
    processed: list[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for uf in uploaded_files:
        fname = Path(uf.name).name
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        for d in pages:
            if not d.metadata:
                d.metadata = {}
            d.metadata["file_name"] = fname
        splits = splitter.split_documents(pages)
        texts = [d.page_content for d in splits]
        if not texts:
            continue
        vecs = embeddings.embed_documents(texts)
        rows: list[dict[str, Any]] = []
        for doc, vec in zip(splits, vecs, strict=False):
            fn = str(doc.metadata.get("file_name") or fname)
            rows.append(
                {
                    "session_id": session_id,
                    "file_name": fn,
                    "content": doc.page_content,
                    "metadata": {
                        k: v
                        for k, v in dict(doc.metadata).items()
                        if isinstance(v, (str, int, float, bool))
                    },
                    "embedding": vec,
                }
            )
        insert_vector_rows(sb, session_id, rows)
        processed.append(fname)
    return processed


def ensure_active_session(sb: Client, user_id: str) -> str:
    sid = st.session_state.get("active_session_id")
    if sid and verify_session_owned(sb, user_id, str(sid)):
        return str(sid)
    new_id = insert_session(sb, user_id, "새 세션")
    st.session_state.active_session_id = new_id
    return new_id


def auto_sync_session(sb: Client, user_id: str) -> None:
    sid = st.session_state.get("active_session_id")
    if not sid:
        return
    if not verify_session_owned(sb, user_id, str(sid)):
        return
    try:
        replace_messages(sb, user_id, str(sid), list(st.session_state.chat_history))
        sb.table("chat_sessions").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", str(sid)).eq("user_id", user_id).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)


def maybe_update_title_from_first_turn(sb: Client, user_id: str, llm: ChatOpenAI) -> None:
    hist = st.session_state.chat_history
    sid = st.session_state.get("active_session_id")
    if not sid or len(hist) < 2:
        return
    if st.session_state.get("title_generated"):
        return
    if not verify_session_owned(sb, user_id, str(sid)):
        return
    first_user = next((m["content"] for m in hist if m["role"] == "user"), "")
    first_asst = next((m["content"] for m in hist if m["role"] == "assistant"), "")
    if not first_user or not first_asst:
        return
    base_ans = first_asst.split("### 💡 다음에 물어볼 수 있는 질문들")[0].strip()
    title = generate_session_title(llm, first_user, base_ans)
    try:
        update_session_title(sb, user_id, str(sid), title)
        st.session_state.session_title_cache = title
        st.session_state.title_generated = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("제목 업데이트 실패: %s", exc)


def duplicate_session_save(sb: Client, user_id: str, llm: ChatOpenAI) -> None:
    old_sid = st.session_state.get("active_session_id")
    hist = list(st.session_state.chat_history)
    if not hist:
        st.warning("저장할 대화가 없습니다.")
        return
    if old_sid and not verify_session_owned(sb, user_id, str(old_sid)):
        st.error("현재 세션에 대한 권한이 없습니다.")
        return
    first_user = next((m["content"] for m in hist if m["role"] == "user"), "대화")
    first_asst = next((m["content"] for m in hist if m["role"] == "assistant"), "")
    base_ans = (first_asst.split("### 💡 다음에 물어볼 수 있는 질문들")[0] if first_asst else "").strip()
    title = generate_session_title(llm, first_user, base_ans or first_user[:80])
    new_sid = insert_session(sb, user_id, title)
    replace_messages(sb, user_id, new_sid, hist)
    if old_sid and verify_session_owned(sb, user_id, str(old_sid)):
        try:
            copy_vectors_between_sessions(sb, user_id, str(old_sid), new_sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("벡터 복사 실패: %s", exc)
    st.session_state.active_session_id = new_sid
    st.session_state.session_title_cache = title
    st.session_state.title_generated = True
    st.session_state.last_sidebar_pick = new_sid
    st.success("새 세션으로 저장되었습니다.")


def load_session_by_id(sb: Client, user_id: str, session_id: str) -> None:
    if not verify_session_owned(sb, user_id, session_id):
        st.error("해당 세션을 불러올 권한이 없습니다.")
        return
    st.session_state.chat_history = load_messages(sb, user_id, session_id)
    st.session_state.conversation_memory = st.session_state.chat_history[-50:]
    st.session_state.active_session_id = session_id
    st.session_state.last_sidebar_pick = session_id
    st.session_state.title_generated = True
    st.session_state.session_title_cache = next(
        (r["title"] for r in list_sessions(sb, user_id) if str(r["id"]) == session_id),
        "",
    )


def clear_chat_ui_state() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.active_session_id = None
    st.session_state.processed_names = []
    st.session_state.title_generated = False
    st.session_state.session_title_cache = ""
    st.session_state.last_sidebar_pick = None
    st.session_state.show_vectordb_files = False
    if "session_select_idx" in st.session_state:
        del st.session_state["session_select_idx"]


def logout_user() -> None:
    st.session_state.current_user_id = None
    st.session_state.current_user_login_id = None
    clear_chat_ui_state()


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "current_user_id": None,
        "current_user_login_id": None,
        "chat_history": [],
        "conversation_memory": [],
        "active_session_id": None,
        "processed_names": [],
        "title_generated": False,
        "session_title_cache": "",
        "last_sidebar_pick": None,
        "show_vectordb_files": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
    )
    load_dotenv(dotenv_path=ENV_PATH)
    apply_streamlit_secrets_overrides()
    _init_state()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    missing = missing_env_keys()
    if missing:
        st.warning(
            "다음 키가 설정되지 않았습니다. Streamlit Cloud는 **Secrets**, 로컬은 프로젝트 루트 **.env**에 "
            "추가해 주세요: **"
            + "**, **".join(missing)
            + "**"
        )

    sb = get_supabase_client()
    if sb is None:
        st.error(
            "Supabase 연결 정보(SUPABASE_URL, SUPABASE_ANON_KEY 또는 SUPABASE_SERVICE_ROLE_KEY)를 "
            "확인해 주세요."
        )

    if sb and supabase_uses_service_role():
        st.warning(
            "⚠️ `SUPABASE_SERVICE_ROLE_KEY`로 연결 중입니다. RLS를 우회하므로 "
            "공개 배포·버전 관리에 넣지 마세요. `multi-session-ref-grants-only.sql` 실행 후 "
            "anon 키만 쓰는 것을 권장합니다."
        )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            f"""
<h1 style="text-align:center; margin:0; font-size:3.2rem !important;">
  <span style="color:#1f77b4;">기획예산처</span>
  <span style="color:#ff8c00;">RAG</span>
  <span style="color:#1f77b4;">챗봇</span>
</h1>
<p style="text-align:center; color:#666; margin-top:0.5rem;">멀티유저 · 멀티세션 · 대화 저장</p>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    uid = st.session_state.current_user_id
    login_display = st.session_state.current_user_login_id or ""

    with st.sidebar:
        st.markdown("**계정**")
        if not sb:
            st.info("Supabase 설정 후 로그인할 수 있습니다.")
        elif uid:
            st.success(f"로그인: **{login_display}**")
            if st.button("로그아웃"):
                logout_user()
                st.rerun()
        else:
            reg_lid = st.text_input("로그인 ID", key="auth_login_id")
            reg_pw = st.text_input("비밀번호", type="password", key="auth_password")
            c_login, c_reg = st.columns(2)
            with c_login:
                if st.button("로그인"):
                    u, err = authenticate_user(sb, reg_lid, reg_pw)
                    if u:
                        st.session_state.current_user_id = u
                        st.session_state.current_user_login_id = reg_lid.strip()
                        clear_chat_ui_state()
                        st.rerun()
                    else:
                        st.error(err)
            with c_reg:
                if st.button("회원가입"):
                    ok, msg = register_user(sb, reg_lid, reg_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.divider()

        if not sb:
            st.markdown("**세션 관리**")
            st.caption("Supabase URL/키를 설정하면 활성화됩니다.")
            sessions = []
            sid_list: list[str] = []
            labels: list[str] = []
        elif not uid:
            st.markdown("**세션 관리**")
            st.caption("로그인 후 이용할 수 있습니다.")
            sessions = []
            sid_list = []
            labels = []
        else:
            sessions = list_sessions(sb, str(uid))
            sid_list = [str(s["id"]) for s in sessions]
            labels = [str(s["title"]) for s in sessions]

            st.markdown("**세션 관리**")
            options_ids: list[str | None] = [None] + sid_list
            options_labels = ["➕ 새 대화 (화면 전용)"] + labels
            cur = st.session_state.active_session_id
            default_i = 0
            if cur and cur in sid_list:
                default_i = sid_list.index(cur) + 1
            pick_i = st.selectbox(
                "세션 선택",
                range(len(options_labels)),
                index=min(default_i, len(options_labels) - 1),
                format_func=lambda i: options_labels[i],
                key="session_select_idx",
            )
            picked_sid = options_ids[pick_i]

            if picked_sid != st.session_state.last_sidebar_pick:
                st.session_state.last_sidebar_pick = picked_sid
                if picked_sid is None:
                    st.session_state.active_session_id = None
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.title_generated = False
                    st.session_state.session_title_cache = ""
                else:
                    load_session_by_id(sb, str(uid), str(picked_sid))
                st.rerun()

            if st.button("세션로드"):
                if picked_sid is None:
                    st.warning("먼저 목록에서 불러올 세션을 선택하세요.")
                else:
                    load_session_by_id(sb, str(uid), str(picked_sid))
                    st.success("세션을 불러왔습니다.")
                    st.rerun()

            if st.button("세션저장"):
                try:
                    llm = get_llm()
                    duplicate_session_save(sb, str(uid), llm)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("세션 저장 실패: %s", exc)
                    st.error(f"세션 저장 실패: {exc}")

            if st.button("세션삭제"):
                if not st.session_state.active_session_id:
                    st.warning("삭제할 활성 세션이 없습니다.")
                else:
                    try:
                        delete_session(sb, str(uid), str(st.session_state.active_session_id))
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = []
                        st.session_state.active_session_id = None
                        st.session_state.last_sidebar_pick = None
                        st.session_state.title_generated = False
                        if "session_select_idx" in st.session_state:
                            del st.session_state["session_select_idx"]
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"삭제 실패: {exc}")

            if st.button("화면초기화"):
                clear_chat_ui_state()
                st.success("화면을 초기화했습니다. (DB의 저장 세션은 그대로입니다.)")
                st.rerun()

            if st.button("vectordb"):
                st.session_state.show_vectordb_files = not st.session_state.show_vectordb_files

            if st.session_state.show_vectordb_files:
                aid = st.session_state.active_session_id
                if not aid:
                    st.text("(활성 세션 없음 — PDF 처리 또는 세션 로드 후 확인)")
                else:
                    try:
                        names = fetch_vector_filenames(sb, str(uid), str(aid))
                        st.text(
                            "Vector DB 파일명:\n"
                            + ("\n".join(f"- {n}" for n in names) if names else "(없음)")
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.text(f"조회 실패: {exc}")

            st.markdown("**RAG (PDF)**")
            uploads = st.file_uploader(
                "PDF 업로드",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if st.button("파일 처리하기"):
                if not uploads:
                    st.warning("PDF를 선택해 주세요.")
                elif "OPENAI_API_KEY" in missing:
                    st.error("OPENAI_API_KEY를 설정한 뒤 PDF 임베딩을 사용할 수 있습니다.")
                else:
                    try:
                        sid = ensure_active_session(sb, str(uid))
                        emb = get_embeddings()
                        names = process_pdfs_to_supabase(sb, sid, list(uploads), emb)
                        st.session_state.processed_names = names
                        auto_sync_session(sb, str(uid))
                        st.success("PDF가 벡터 DB에 반영되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("PDF 처리 실패: %s", exc)
                        st.error(f"PDF 처리 오류: {exc}")

            if st.session_state.processed_names:
                st.markdown("**처리된 파일**")
                for n in st.session_state.processed_names:
                    st.text(f"- {n}")

            mem_n = len(st.session_state.conversation_memory)
            aid = st.session_state.active_session_id
            try:
                vs_n = len(fetch_vector_filenames(sb, str(uid), str(aid))) if aid else 0
            except Exception:
                vs_n = 0
            st.text(
                f"모델: {MODEL_NAME}\n"
                f"사용자: {login_display}\n"
                f"활성 세션 ID: {st.session_state.active_session_id or '(없음)'}\n"
                f"처리된 PDF(최근 처리): {len(st.session_state.processed_names)}\n"
                f"벡터 문서 파일 종류 수: {vs_n}\n"
                f"대화 메시지 수: {mem_n}"
            )

    if not sb:
        st.stop()

    if not uid:
        st.info("왼쪽 사이드바에서 **회원가입** 또는 **로그인** 후 대화를 시작할 수 있습니다.")
        st.caption("DB에 `users` 테이블과 `chat_sessions`·`chat_messages`의 `user_id` 컬럼이 필요합니다. (`prompts/multi-users-schema.sql`)")
        st.stop()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    if "OPENAI_API_KEY" in missing:
        st.error("OPENAI_API_KEY가 없어 답변을 생성할 수 없습니다.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        try:
            sid = ensure_active_session(sb, str(uid))
            llm = get_llm()
            emb = get_embeddings()
            mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
            docs = retrieve_documents(sb, str(sid), user_input, emb, k=10)
            if docs:
                context = "\n\n".join(d.page_content for d in docs)
                messages = _build_rag_messages(user_input, context, mem_txt)
                acc = ""
                for chunk in llm.stream(messages):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
            else:
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)

            placeholder.markdown(full_answer)
            follow = _generate_followup_section(llm, user_input, full_answer)
            if follow:
                full_answer += follow
                placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

        try:
            llm2 = get_llm()
            maybe_update_title_from_first_turn(sb, str(uid), llm2)
            auto_sync_session(sb, str(uid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("대화 후 저장 실패: %s", exc)

    st.rerun()


if __name__ == "__main__":
    main()
