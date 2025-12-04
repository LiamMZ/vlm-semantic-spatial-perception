"""
GenAI Log Viewer

Textual TUI for browsing GenAI request/response logs emitted by
`configure_genai_logging(...)`. Point it at a `genai_logs/` directory
to cycle through call folders, inspect metadata, prompt text, responses,
and attached media paths.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, ListItem, ListView, RichLog, Static


@dataclass
class LogEntry:
    """Represents a single GenAI call folder."""

    call_id: str
    path: Path
    timestamp: str
    status: str
    caller: str
    prompt_tokens: Optional[int]
    response_tokens: Optional[int]
    media_parts: int
    text_parts: int


def _find_latest_genai_logs() -> Optional[Path]:
    """Locate the most recent genai_logs directory under outputs/demos/."""
    base = Path("outputs")
    candidates: List[Tuple[float, Path]] = []
    for path in base.glob("**/genai_logs"):
        if path.is_dir():
            try:
                candidates.append((path.stat().st_mtime, path))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].resolve()


class GenAILogViewer(App):
    """Textual app for navigating GenAI request/response logs."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 38;
        border: solid $accent;
    }

    #content {
        border: solid $accent;
    }

    #status {
        padding: 0 1;
        height: 3;
        border-bottom: heavy $accent;
    }

    #meta-panel { height: 7; border-bottom: solid $accent; }
    #request-panel { height: 1fr; border-bottom: solid $accent; }
    #response-panel { height: 1fr; border-bottom: solid $accent; }
    #media-panel { height: 0.7fr; border-bottom: solid $accent; }

    #meta-log, #request-log, #response-log, #media-log {
        height: 1fr;
    }

    #help {
        padding: 0 1;
        height: 2;
    }

    #log-list > ListItem {
        border-bottom: solid $accent;
    }

    #meta-panel { height: 7; border-bottom: solid $accent; }
    #request-panel { height: 1fr; border-bottom: solid $accent; }
    #response-panel { height: 1fr; border-bottom: solid $accent; }
    #media-panel { height: 0.7fr; border-bottom: solid $accent; }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload_entries", "Reload"),
        ("left", "prev_entry", "Previous"),
        ("right", "next_entry", "Next"),
        ("m", "toggle_media", "Media"),
    ]

    def __init__(self, log_root: Path):
        super().__init__()
        self.log_root = log_root
        self.project_root = Path(__file__).resolve().parent.parent
        self.entries: List[LogEntry] = []
        self._selected_index: int = 0
        self.dark = False  # Start in light mode by default
        self.show_media: bool = False

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("GenAI Calls", id="sidebar-title")
                yield ListView(id="log-list")
            with Vertical(id="content"):
                yield Static("", id="status")
                with Vertical(id="meta-panel"):
                    yield Static("Metadata", id="meta-title")
                    yield RichLog(id="meta-log", markup=True, wrap=True)
                with Vertical(id="request-panel"):
                    yield Static("Request (prompt & config)", id="request-title")
                    yield RichLog(id="request-log", markup=True, wrap=True)
                with Vertical(id="response-panel"):
                    yield Static("Response Text", id="response-title")
                    yield RichLog(id="response-log", markup=True, wrap=True)
                with Vertical(id="media-panel"):
                    yield Static("Media Files (full paths)", id="media-title")
                    yield RichLog(id="media-log", markup=True, wrap=True)
                yield Static(
                    "Commands: ↑/↓ select • ←/→ previous/next • r reload • q quit",
                    id="help",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Load entries and focus the list on startup."""
        self._reload_entries()
        self._focus_list()
        self._set_media_visibility()

    def action_reload_entries(self) -> None:
        """Reload the log list from disk."""
        self._reload_entries()

    def action_prev_entry(self) -> None:
        """Select previous entry."""
        if not self.entries:
            return
        self._selected_index = (self._selected_index - 1) % len(self.entries)
        self._apply_selection()

    def action_next_entry(self) -> None:
        """Select next entry."""
        if not self.entries:
            return
        self._selected_index = (self._selected_index + 1) % len(self.entries)
        self._apply_selection()

    def action_toggle_media(self) -> None:
        """Show/hide media panel to maximize space for request/response."""
        self.show_media = not self.show_media
        self._set_media_visibility()
        if self.show_media and self.entries:
            self._render_media(self.entries[self._selected_index])

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Update selection when a list item is clicked or highlighted."""
        if event.item.data is None:
            return
        try:
            idx = self.entries.index(event.item.data)
            self._selected_index = idx
            self._display_entry(self.entries[idx])
        except ValueError:
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _reload_entries(self) -> None:
        """Read log entries from disk and repopulate the sidebar."""
        list_view = self.query_one("#log-list", ListView)
        list_view.clear()

        self.entries = self._load_entries()
        if not self.entries:
            self._update_status(f"No entries found under {self.log_root}")
            return

        for entry in self.entries:
            ts = self._format_time(entry.timestamp)
            pt = entry.prompt_tokens if entry.prompt_tokens is not None else "-"
            rt = entry.response_tokens if entry.response_tokens is not None else "-"
            label = (
                f"{ts} | in:{pt} out:{rt} | m{entry.media_parts}/t{entry.text_parts}\n"
                f"caller: {entry.caller or '-'}"
            )
            item = ListItem(Static(label))
            item.data = entry
            list_view.append(item)

        # Keep selection in range
        self._selected_index = min(self._selected_index, len(self.entries) - 1)
        list_view.index = self._selected_index
        self._display_entry(self.entries[self._selected_index])
        self._set_media_visibility()

    def _load_entries(self) -> List[LogEntry]:
        """Load entries from the configured log root."""
        if not self.log_root.exists():
            return []

        entries: List[LogEntry] = []
        for child in sorted(self.log_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not child.is_dir():
                continue
            metadata = self._safe_read_json(child / "metadata.json") or {}
            caller_info = metadata.get("caller") or {}
            caller_str = self._caller_label(caller_info)

            prompt_tokens, response_tokens, media_parts, text_parts = self._extract_counts(child)

            entries.append(
                LogEntry(
                    call_id=metadata.get("call_id", child.name),
                    path=child,
                    timestamp=metadata.get("timestamp_utc", ""),
                    status=metadata.get("status", "unknown"),
                    caller=caller_str,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    media_parts=media_parts,
                    text_parts=text_parts,
                )
            )
        return entries

    def _apply_selection(self) -> None:
        """Sync list selection and render the selected entry."""
        if not self.entries:
            return
        list_view = self.query_one("#log-list", ListView)
        list_view.index = self._selected_index
        self._display_entry(self.entries[self._selected_index])

    def _display_entry(self, entry: LogEntry) -> None:
        """Render metadata, request, response, and media for the entry."""
        self._update_status(f"{entry.call_id} — {entry.path}")
        self._render_metadata(entry)
        self._render_request(entry)
        self._render_response(entry)
        self._render_media(entry)

    def _render_metadata(self, entry: LogEntry) -> None:
        log = self.query_one("#meta-log", RichLog)
        log.clear()
        metadata = self._safe_read_json(entry.path / "metadata.json") or {}
        request = self._safe_read_json(entry.path / "request.json") or {}
        model = request.get("model")
        config = request.get("config") or {}
        non_null_config = {k: v for k, v in config.items() if v not in (None, {}, [])}
        lines = [
            f"[b]Call ID[/b]: {metadata.get('call_id', '')}",
            f"[b]Timestamp[/b]: {metadata.get('timestamp_utc', '')}",
            f"[b]Status[/b]: {metadata.get('status', '')}",
            f"[b]Elapsed[/b]: {metadata.get('elapsed_seconds', '')} s",
            f"[b]Caller[/b]: {entry.caller or metadata.get('caller', '')}",
            f"[b]Directory[/b]: {entry.path}",
        ]
        if model:
            lines.append(f"[b]Model[/b]: {model}")
        if non_null_config:
            lines.append("[b]Config[/b]:")
            for key, value in non_null_config.items():
                lines.append(f"  • {key}: {value}")
        for line in lines:
            log.write(line)

    def _render_request(self, entry: LogEntry) -> None:
        log = self.query_one("#request-log", RichLog)
        log.clear()
        request = self._safe_read_json(entry.path / "request.json") or {}
        saved_media = request.get("saved_media") or []
        media_lookup = {m.get("file"): m for m in saved_media if isinstance(m, dict)}
        contents = request.get("contents") or (request.get("raw_kwargs") or {}).get("contents") or []

        if not contents:
            log.write("No request content.")
            return

        def render_part(part: Any) -> None:
            if part is None:
                return
            # Nested content with role + parts
            if isinstance(part, dict) and "parts" in part:
                role = part.get("role")
                if role:
                    log.write(f"[b]Role[/b]: {role}")
                for sub in part.get("parts") or []:
                    render_part(sub)
                return

            # Inline text
            text_val = None
            if isinstance(part, dict):
                text_val = part.get("text")
                if text_val is None and part.get("type") == "text" and part.get("file"):
                    text_val = self._safe_read_text(entry.path / part.get("file"), fallback=None)
            elif isinstance(part, str):
                text_val = part
            if isinstance(text_val, str):
                # Try to load from file if available
                file_name = None
                if isinstance(part, dict):
                    file_name = part.get("file")
                if file_name:
                    text = self._safe_read_text(entry.path / file_name, fallback=text_val)
                    log.write(f"[b]Text ({file_name})[/b]:")
                else:
                    text = text_val
                    log.write("[b]Text[/b]:")
                for line in text.splitlines():
                    log.write(f"  {line}")
                return

            # Media placeholder
            mime_type = None
            file_name = None
            size_hint = None
            if isinstance(part, dict):
                mime_type = part.get("mime_type") or part.get("type")
                file_name = part.get("file")
                size_hint = part.get("size")
            if isinstance(part, dict) and part.get("inline_data") is not None:
                inline_data = part.get("inline_data") or {}
                mime_type = mime_type or inline_data.get("mime_type")
                file_name = file_name or (inline_data if isinstance(inline_data, str) else None)
            media_hint = mime_type or "media"
            if file_name:
                log.write(f"[b]Media ({media_hint})[/b]: {file_name} (press m to view attachments)")
            else:
                size_str = f" ~{size_hint} bytes" if size_hint else ""
                log.write(f"[b]Media ({media_hint})[/b]: (not captured{size_str})")

        for item in contents:
            render_part(item)

    def _render_response(self, entry: LogEntry) -> None:
        log = self.query_one("#response-log", RichLog)
        log.clear()
        response_text_path = entry.path / "response_text.txt"
        response_json_path = entry.path / "response.json"
        stream_path = entry.path / "response_stream.jsonl"

        if response_text_path.exists():
            log.write(f"[b]Response text ({response_text_path})[/b]:")
            text = self._safe_read_text(response_text_path, fallback="[unable to read response_text.txt]")
            for line in text.splitlines():
                log.write(line)
        elif response_json_path.exists():
            log.write(f"[b]Response JSON ({response_json_path})[/b]:")
            payload = self._safe_read_json(response_json_path) or {}
            pretty = json.dumps(payload, indent=2)
            for line in pretty.splitlines():
                log.write(line)
        else:
            log.write("No response payload found.")

        if stream_path.exists():
            log.write(f"\n[b]Stream chunks[/b]: {stream_path}")

    def _render_media(self, entry: LogEntry) -> None:
        log = self.query_one("#media-log", RichLog)
        log.clear()
        if not self.show_media:
            return
        request = self._safe_read_json(entry.path / "request.json") or {}
        saved_media = request.get("saved_media") or []
        if not saved_media:
            log.write("No media attachments.")
            return

        log.write("[b]Media attachments[/b]:")
        for item in saved_media:
            file_name = item.get("file")
            full_path = entry.path / file_name if file_name else None
            desc = f"{item.get('type')}"
            if item.get("mime_type"):
                desc += f" ({item.get('mime_type')})"
            size_info = item.get("size") or item.get("length")
            if size_info:
                desc += f" — {size_info} bytes"
            log.write(f"  • {desc}")
            if full_path:
                log.write(f"    {self._relative_path(full_path)}")
            elif item.get("file") is None:
                log.write("    (not captured)")

    def _update_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _focus_list(self) -> None:
        try:
            self.query_one("#log-list", ListView).focus()
        except Exception:
            pass

    def _set_media_visibility(self) -> None:
        """Toggle media panel visibility."""
        media_panel = self.query_one("#media-panel", Vertical)
        if self.show_media:
            media_panel.remove_class("hidden")
        else:
            media_panel.add_class("hidden")

    @staticmethod
    def _caller_label(caller_info: Dict) -> str:
        """Produce a compact caller string."""
        module = caller_info.get("module") if isinstance(caller_info, dict) else None
        file_path = caller_info.get("file") if isinstance(caller_info, dict) else None
        function = caller_info.get("function") if isinstance(caller_info, dict) else None

        module_part = ""
        if module:
            module_part = module.split(".")[-1]
        elif file_path:
            module_part = Path(file_path).stem

        func_part = function or ""
        return module_part or func_part or ""

    def _relative_path(self, path: Path) -> str:
        """Return path relative to the project root when possible."""
        try:
            return str(path.relative_to(self.project_root))
        except Exception:
            try:
                return str(path.relative_to(self.project_root.parent))
            except Exception:
                return str(path)

    @staticmethod
    def _format_time(timestamp: str) -> str:
        """Extract hh:mm:ss.mmm from an ISO timestamp."""
        if not timestamp:
            return "--:--:--"
        try:
            # Expecting format like 2025-12-02T22:37:22.204676+00:00
            t_part = timestamp.split("T")[1]
            if "+" in t_part:
                t_part = t_part.split("+")[0]
            if "Z" in t_part:
                t_part = t_part.rstrip("Z")
            # Keep milliseconds only
            if "." in t_part:
                main, frac = t_part.split(".", 1)
                t_part = f"{main}.{frac[:3]}"
            return t_part
        except Exception:
            return timestamp

    def _extract_counts(self, entry_dir: Path) -> Tuple[Optional[int], Optional[int], int, int]:
        """Pull token and media/text counts from request/response payloads."""
        prompt_tokens: Optional[int] = None
        response_tokens: Optional[int] = None
        media_parts = 0
        text_parts = 0

        request = self._safe_read_json(entry_dir / "request.json") or {}
        contents = request.get("contents") or []
        for item in contents:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"inline_data", "bytes"}:
                media_parts += 1
            if item.get("type") == "text":
                text_parts += 1

        # Token usage lives on response payload when provided by the API.
        resp = self._safe_read_json(entry_dir / "response.json") or {}
        usage = resp.get("usage") or resp.get("usage_metadata") or {}
        if isinstance(usage, dict):
            prompt_tokens = (
                usage.get("prompt_token_count")
                or usage.get("input_tokens")
                or usage.get("inputTokenCount")
            )
            response_tokens = (
                usage.get("candidates_token_count")
                or usage.get("output_tokens")
                or usage.get("outputTokenCount")
            )

        # For streaming calls, usage metadata may land in the stream chunks.
        if (prompt_tokens is None or response_tokens is None) and (entry_dir / "response_stream.jsonl").exists():
            try:
                with open(entry_dir / "response_stream.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        usage = chunk.get("usage") or chunk.get("usage_metadata") or {}
                        if isinstance(usage, dict):
                            prompt_tokens = prompt_tokens or (
                                usage.get("prompt_token_count")
                                or usage.get("input_tokens")
                                or usage.get("inputTokenCount")
                            )
                            response_tokens = response_tokens or (
                                usage.get("candidates_token_count")
                                or usage.get("output_tokens")
                                or usage.get("outputTokenCount")
                            )
                        if prompt_tokens is not None and response_tokens is not None:
                            break
            except Exception:
                pass

        return prompt_tokens, response_tokens, media_parts, text_parts

    @staticmethod
    def _safe_read_json(path: Path) -> Optional[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    @staticmethod
    def _safe_read_text(path: Path, fallback: str = "") -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return fallback


def _resolve_log_root_from_cli() -> Path:
    """Resolve the log root from CLI arg or error if none provided."""
    if len(sys.argv) <= 1:
        print("Error: Please provide a genai_logs directory or its parent (world/output dir).")
        print("Example: uv run python examples/genai_viewer.py outputs/demos/<run>/genai_logs")
        sys.exit(1)

    candidate = Path(sys.argv[1]).expanduser().resolve()
    if candidate.is_dir():
        if candidate.name != "genai_logs" and (candidate / "genai_logs").is_dir():
            return (candidate / "genai_logs").resolve()
        if candidate.name == "genai_logs":
            return candidate
        print(f"No genai_logs/ found under: {candidate}")
        sys.exit(1)

    print(f"Provided path is not a directory: {candidate}")
    sys.exit(1)


if __name__ == "__main__":
    log_root = _resolve_log_root_from_cli()
    app = GenAILogViewer(log_root)
    app.run()
