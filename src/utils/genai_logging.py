"""
Global logging wrapper for Google GenAI client requests.

This module monkeypatches the GenAI client so every `generate_content`
invocation (sync or async/streaming) is captured to disk with the
request, response, and media payloads. It is opt-in: call
`configure_genai_logging(log_root)` once per process to enable and set
the output directory.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import time
import uuid
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent in some environments
    genai = None  # type: ignore

# Shared state for the monkeypatch
_STATE: Dict[str, Any] = {
    "log_root": None,
    "patched": False,
    "original_init": None,
    "lock": RLock(),
}


def configure_genai_logging(log_root: Optional[Union[str, Path]]) -> Optional[Path]:
    """
    Enable request/response logging for all GenAI client calls.

    Args:
        log_root: Directory where call folders should be written. If None,
            logging is disabled.

    Returns:
        The resolved log root if logging is enabled, otherwise None.
    """
    if genai is None or log_root is None:
        return None

    resolved = Path(log_root)
    resolved.mkdir(parents=True, exist_ok=True)

    with _STATE["lock"]:
        _STATE["log_root"] = resolved
        if not _STATE["patched"]:
            _patch_client_init()
            _STATE["patched"] = True

    return resolved


def get_genai_log_root() -> Optional[Path]:
    """Return the current log root if logging is enabled."""
    return _STATE.get("log_root")


def _patch_client_init() -> None:
    """Monkeypatch genai.Client.__init__ to wrap model services once per instance."""
    if genai is None:
        return

    original_init = genai.Client.__init__  # type: ignore[attr-defined]
    _STATE["original_init"] = original_init

    def _wrapped_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        original_init(self, *args, **kwargs)
        _wrap_client_services(self)

    genai.Client.__init__ = _wrapped_init  # type: ignore[assignment]


def _wrap_client_services(client: Any) -> None:
    """Wrap sync and async model services on the client instance."""
    log_root = _STATE.get("log_root")
    if not log_root:
        return

    try:
        models_service = getattr(client, "models", None)
        if models_service:
            _patch_models_service(models_service)
    except Exception:
        pass

    try:
        aio_client = getattr(client, "aio", None)
        aio_models = getattr(aio_client, "models", None) if aio_client else None
        if aio_models:
            _patch_models_service(aio_models)
    except Exception:
        pass


def _patch_models_service(models_service: Any) -> None:
    """Patch generate_content and generate_content_stream on the provided service."""
    cls = models_service.__class__
    marker = "_genai_logging_patched"
    if getattr(cls, marker, False):
        return

    if hasattr(cls, "generate_content"):
        method = getattr(cls, "generate_content")
        if inspect.iscoroutinefunction(method):
            setattr(cls, "generate_content", _wrap_async_call(method))
        else:
            setattr(cls, "generate_content", _wrap_sync_call(method))

    if hasattr(cls, "generate_content_stream"):
        method = getattr(cls, "generate_content_stream")
        if inspect.iscoroutinefunction(method):
            setattr(cls, "generate_content_stream", _wrap_async_stream_call(method))
        else:
            setattr(cls, "generate_content_stream", _wrap_sync_stream_call(method))

    setattr(cls, marker, True)


def _wrap_sync_call(func: Any) -> Any:
    """Wrap a synchronous generate_content call."""

    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _start_request_log(args, kwargs, is_stream=False)
        try:
            result = func(self, *args, **kwargs)
            _finalize_response_log(ctx, response=result)
            return result
        except Exception as exc:  # pragma: no cover - passthrough for runtime errors
            _finalize_response_log(ctx, error=exc)
            raise

    return _wrapper


def _wrap_async_call(func: Any) -> Any:
    """Wrap an async generate_content call."""

    async def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _start_request_log(args, kwargs, is_stream=False)
        try:
            result = await func(self, *args, **kwargs)
            _finalize_response_log(ctx, response=result)
            return result
        except Exception as exc:  # pragma: no cover - passthrough for runtime errors
            _finalize_response_log(ctx, error=exc)
            raise

    return _wrapper


def _wrap_sync_stream_call(func: Any) -> Any:
    """Wrap a synchronous generate_content_stream call."""

    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _start_request_log(args, kwargs, is_stream=True)
        stream = func(self, *args, **kwargs)
        return _SyncStreamLogger(stream, ctx)

    return _wrapper


def _wrap_async_stream_call(func: Any) -> Any:
    """Wrap an async generate_content_stream call."""

    async def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _start_request_log(args, kwargs, is_stream=True)
        stream = await func(self, *args, **kwargs)
        return _AsyncStreamLogger(stream, ctx)

    return _wrapper


class _AsyncStreamLogger:
    """Async iterator that records streamed chunks to disk."""

    def __init__(self, stream: Any, ctx: Dict[str, Any]) -> None:
        self._stream = stream
        self._ctx = ctx
        self._chunks: List[Any] = []
        self._text_parts: List[str] = []
        self._closed = False

    def __aiter__(self) -> "_AsyncStreamLogger":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._record_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            await self._finalize()
            raise
        except Exception as exc:  # pragma: no cover - passthrough for runtime errors
            await self._finalize(error=exc)
            raise

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            try:
                await self._stream.aclose()
            except Exception:
                pass
        await self._finalize()

    def _record_chunk(self, chunk: Any) -> None:
        self._chunks.append(_to_jsonable(chunk))
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            self._text_parts.append(text)

    async def _finalize(self, error: Optional[Exception] = None) -> None:
        if self._closed:
            return
        self._closed = True
        _finalize_response_log(
            self._ctx,
            chunks=self._chunks,
            aggregated_text="".join(self._text_parts) if self._text_parts else None,
            error=error,
        )


class _SyncStreamLogger:
    """Sync iterator that records streamed chunks to disk."""

    def __init__(self, stream: Any, ctx: Dict[str, Any]) -> None:
        self._stream = stream
        self._ctx = ctx
        self._chunks: List[Any] = []
        self._text_parts: List[str] = []
        self._closed = False

    def __iter__(self) -> "_SyncStreamLogger":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._record_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except Exception as exc:  # pragma: no cover - passthrough for runtime errors
            self._finalize(error=exc)
            raise

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception:
                pass
        self._finalize()

    def _record_chunk(self, chunk: Any) -> None:
        self._chunks.append(_to_jsonable(chunk))
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            self._text_parts.append(text)

    def _finalize(self, error: Optional[Exception] = None) -> None:
        if self._closed:
            return
        self._closed = True
        _finalize_response_log(
            self._ctx,
            chunks=self._chunks,
            aggregated_text="".join(self._text_parts) if self._text_parts else None,
            error=error,
        )


def _start_request_log(args: Tuple[Any, ...], kwargs: Dict[str, Any], is_stream: bool) -> Dict[str, Any]:
    """Create the call folder and write request metadata."""
    log_root: Optional[Path] = _STATE.get("log_root")
    if not log_root:
        return {"enabled": False}

    call_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_{uuid.uuid4().hex[:8]}"
    call_dir = log_root / call_id
    call_dir.mkdir(parents=True, exist_ok=True)

    model, contents, config = _extract_request_parts(args, kwargs)
    request_payload, saved_media = _serialize_request_contents(contents, call_dir)

    caller = _find_caller()
    metadata = {
        "call_id": call_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "is_stream": is_stream,
        "caller": caller,
    }

    _write_json(call_dir / "metadata.json", metadata)
    _write_json(
        call_dir / "request.json",
        {
            "model": model,
            "config": _to_jsonable(config),
            "contents": request_payload,
            "saved_media": saved_media,
            "raw_args": _to_jsonable(args),
            "raw_kwargs": _to_jsonable(kwargs),
        },
    )

    return {
        "enabled": True,
        "call_dir": call_dir,
        "metadata": metadata,
        "start_time": time.time(),
    }


def _finalize_response_log(
    ctx: Dict[str, Any],
    *,
    response: Any = None,
    chunks: Optional[List[Any]] = None,
    aggregated_text: Optional[str] = None,
    error: Optional[Exception] = None,
) -> None:
    """Write response payloads and update metadata."""
    if not ctx.get("enabled"):
        return

    call_dir: Path = ctx["call_dir"]
    metadata = ctx.get("metadata", {}).copy()
    metadata["elapsed_seconds"] = round(time.time() - ctx.get("start_time", time.time()), 3)
    if error is None:
        metadata["status"] = "ok"
    else:
        metadata["status"] = "error"
        metadata["error"] = repr(error)

    _write_json(call_dir / "metadata.json", metadata)

    if response is not None:
        _write_json(call_dir / "response.json", _to_jsonable(response))
        text = getattr(response, "text", None)
        if isinstance(text, str):
            (call_dir / "response_text.txt").write_text(text, encoding="utf-8")

    if chunks is not None:
        _write_json(call_dir / "response_stream.jsonl", chunks, json_lines=True)

    if aggregated_text:
        (call_dir / "response_text.txt").write_text(aggregated_text, encoding="utf-8")


def _extract_request_parts(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Best-effort extraction of model, contents, and config from call arguments."""
    model = kwargs.get("model")
    contents = kwargs.get("contents")
    config = kwargs.get("config") or kwargs.get("generation_config")

    if model is None and len(args) > 0:
        model = args[0]
    if contents is None and len(args) > 1:
        contents = args[1]
    if config is None and len(args) > 2:
        config = args[2]

    return model, contents, config


def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Return an attribute or dict key if present."""
    try:
        if hasattr(obj, key):
            return getattr(obj, key)
    except Exception:
        pass

    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _coerce_bytes(data: Any) -> Optional[bytes]:
    """Best effort conversion to bytes, tolerating common containers."""
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return None
    try:
        return bytes(data)
    except Exception:
        return None


def _serialize_part(
    part: Any, call_dir: Path, *, content_idx: int, part_idx: Optional[int]
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Serialize an individual part (or free-form content item)."""
    saved_media: List[Dict[str, Any]] = []
    idx_label = f"{content_idx}" if part_idx is None else f"{content_idx}_{part_idx}"

    def _record(info: Dict[str, Any]) -> None:
        info.setdefault("index", content_idx)
        info["content_index"] = content_idx
        if part_idx is not None:
            info["part_index"] = part_idx
        saved_media.append(info)

    if isinstance(part, str):
        filename = f"text_part_{idx_label}.txt"
        (call_dir / filename).write_text(part, encoding="utf-8")
        _record({"type": "text", "file": filename, "length": len(part)})
        return {"type": "text", "file": filename}, saved_media

    if isinstance(part, (bytes, bytearray)):
        filename = f"media_part_{idx_label}.bin"
        payload = bytes(part)
        (call_dir / filename).write_bytes(payload)
        _record({"type": "bytes", "file": filename, "size": len(payload)})
        return {"type": "bytes", "file": filename, "size": len(payload)}, saved_media

    inline_data = _get_attr_or_key(part, "inline_data") or _get_attr_or_key(part, "inlineData")
    if inline_data is not None:
        mime_type = _get_attr_or_key(inline_data, "mime_type") or _get_attr_or_key(inline_data, "mimeType")
        data = _get_attr_or_key(inline_data, "data")
        payload = _coerce_bytes(data)
        if payload:
            suffix = _guess_extension(mime_type)
            filename = f"media_part_{idx_label}{suffix}"
            (call_dir / filename).write_bytes(payload)
            _record({"type": "inline_data", "file": filename, "mime_type": mime_type, "size": len(payload)})
            return {"type": "inline_data", "file": filename, "mime_type": mime_type}, saved_media
        size_hint = None
        if isinstance(data, dict):
            size_hint = data.get("size")
        _record({"type": "inline_data", "file": None, "mime_type": mime_type, "size": size_hint})
        return {"type": "inline_data", "file": None, "mime_type": mime_type, "size": size_hint}, saved_media

    text_val = _get_attr_or_key(part, "text")
    if isinstance(text_val, str):
        filename = f"text_part_{idx_label}.txt"
        (call_dir / filename).write_text(text_val, encoding="utf-8")
        _record({"type": "text", "file": filename, "length": len(text_val)})
        return {"type": "text", "file": filename}, saved_media

    return _to_jsonable(part), saved_media


def _serialize_content_item(
    content: Any, call_dir: Path, *, content_idx: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Serialize a Content object that includes parts."""
    saved_media: List[Dict[str, Any]] = []
    serialized_parts: List[Any] = []
    parts = _get_attr_or_key(content, "parts") or []
    for part_idx, part in enumerate(parts):
        serialized_part, media = _serialize_part(part, call_dir, content_idx=content_idx, part_idx=part_idx)
        serialized_parts.append(serialized_part)
        saved_media.extend(media)

    role = _get_attr_or_key(content, "role")
    serialized_content: Dict[str, Any] = {"parts": serialized_parts}
    if role is not None:
        serialized_content["role"] = role
    return serialized_content, saved_media


def _serialize_request_contents(contents: Any, call_dir: Path) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Serialize request contents and persist media to disk.

    Returns:
        A tuple of (serialized_contents, saved_media_metadata).
    """
    serialized: List[Any] = []
    saved_media: List[Dict[str, Any]] = []
    if contents is None:
        return serialized, saved_media

    if not isinstance(contents, Iterable) or isinstance(contents, (str, bytes, bytearray, dict)):
        contents = [contents]

    for idx, item in enumerate(contents):
        looks_like_content = _get_attr_or_key(item, "parts") is not None
        if looks_like_content:
            serialized_item, media = _serialize_content_item(item, call_dir, content_idx=idx)
        else:
            serialized_item, media = _serialize_part(item, call_dir, content_idx=idx, part_idx=None)

        serialized.append(serialized_item)
        saved_media.extend(media)

    return serialized, saved_media


def _guess_extension(mime_type: Optional[str]) -> str:
    """Map common mime types to extensions for saved media parts."""
    if not mime_type:
        return ".bin"
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "application/json": ".json",
    }
    return mapping.get(mime_type.lower(), ".bin")


def _find_caller() -> Dict[str, Any]:
    """Return the first caller outside this module and google.genai."""
    for frame in inspect.stack()[2:]:
        module = inspect.getmodule(frame.frame)
        module_name = module.__name__ if module else None
        if module_name and (module_name.startswith("google.genai") or module_name.endswith("genai_logging")):
            continue
        filename = frame.filename
        if filename and filename.endswith("genai_logging.py"):
            continue
        return {
            "module": module_name,
            "function": frame.function,
            "file": filename,
            "line": frame.lineno,
        }
    return {}


def _write_json(path: Path, payload: Any, json_lines: bool = False) -> None:
    """Write JSON or JSONL safely."""
    try:
        if json_lines and isinstance(payload, list):
            lines = "\n".join(json.dumps(item, default=str) for item in payload)
            path.write_text(lines, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        # Last resort: write a repr fallback to avoid losing the entry entirely.
        path.write_text(repr(payload), encoding="utf-8")


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "size": len(obj)}
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _to_jsonable(obj.__dict__)
    return repr(obj)


__all__ = ["configure_genai_logging", "get_genai_log_root"]
