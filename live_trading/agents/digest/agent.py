"""
Digest Agent — Anthropic tool_use loop for EOD and Morning trading digests.

Usage:
    agent = DigestAgent(mode="eod", target_date="2026-03-12")
    result = agent.run()
"""

import json
import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import anthropic
from supabase import create_client

from .prompts import EOD_SYSTEM_PROMPT, MORNING_SYSTEM_PROMPT
from .tools import TOOL_DEFINITIONS, TOOL_DISPATCH, save_digest

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

# Model selection — Sonnet for speed + cost, Opus for depth.
MODEL_MAP = {
    "eod": "claude-sonnet-4-20250514",
    "morning": "claude-sonnet-4-20250514",
}

MAX_TURNS = 25              # Safety valve — Sonnet calls tools one-at-a-time, needs ~18 turns
MAX_TOOL_ERRORS = 3         # Abort if the model keeps calling broken tools
MAX_TOKEN_BUDGET = 300_000  # Total tokens (in + out) before forced stop
TOOL_RESULT_MAX_CHARS = 8000  # Truncate individual tool results beyond this


class DigestAgent:
    """Runs one digest cycle: system prompt → tool calls → save_digest."""

    def __init__(
        self,
        mode: str,
        target_date: str | None = None,
        *,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        anthropic_api_key: str | None = None,
        model: str | None = None,
        dry_run: bool = False,
    ):
        if mode not in ("eod", "morning"):
            raise ValueError(f"mode must be 'eod' or 'morning', got {mode!r}")

        self.mode = mode
        self.target_date = target_date or datetime.now(_ET).strftime("%Y-%m-%d")
        self.dry_run = dry_run
        self.model = model or MODEL_MAP[mode]

        # Stats tracked for metadata
        self._tokens_in = 0
        self._tokens_out = 0
        self._tool_calls = 0
        self._start_time: float = 0.0
        self._digest_saved = False

        # Clients (lazy — created in run())
        self._anthropic_key = anthropic_api_key
        self._sb_url = supabase_url
        self._sb_key = supabase_key
        self._client: anthropic.Anthropic | None = None
        self._sb = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full digest loop. Returns the final digest or error."""
        self._start_time = time.monotonic()
        self._init_clients()

        system_prompt = EOD_SYSTEM_PROMPT if self.mode == "eod" else MORNING_SYSTEM_PROMPT
        messages = [
            {
                "role": "user",
                "content": self._build_kickoff_message(),
            }
        ]

        logger.info(
            f"[DigestAgent] Starting {self.mode} digest for {self.target_date} "
            f"(model={self.model}, dry_run={self.dry_run})"
        )

        consecutive_errors = 0
        total_errors = 0
        final_result = None

        for turn in range(MAX_TURNS):
            # ── Call the model ──────────────────────────────────────────
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except (anthropic.AuthenticationError, anthropic.BadRequestError) as e:
                logger.error(f"[DigestAgent] Non-transient API error: {e}")
                return self._make_error_result(
                    f"api_error_{e.status_code}", str(e), turn
                )
            except anthropic.APIConnectionError as e:
                logger.error(f"[DigestAgent] API connection error: {e}")
                return self._make_error_result(
                    "api_connection_error", str(e), turn
                )
            except anthropic.APIStatusError as e:
                logger.error(
                    f"[DigestAgent] API error after retries: {e.status_code} — {e}"
                )
                return self._make_error_result(
                    f"api_error_{e.status_code}", str(e), turn
                )

            # Track usage
            self._tokens_in += response.usage.input_tokens
            self._tokens_out += response.usage.output_tokens

            # ── Budget check ───────────────────────────────────────────
            total_tokens = self._tokens_in + self._tokens_out
            if total_tokens > MAX_TOKEN_BUDGET:
                logger.warning(
                    f"[DigestAgent] Token budget exceeded: {total_tokens:,} > "
                    f"{MAX_TOKEN_BUDGET:,}. Stopping."
                )
                final_result = {
                    "status": "token_budget_exceeded",
                    "turns": turn + 1,
                    "tool_calls": self._tool_calls,
                    "tokens_in": self._tokens_in,
                    "tokens_out": self._tokens_out,
                    "duration_sec": round(time.monotonic() - self._start_time, 2),
                    "cost_usd": self._estimate_cost(),
                    "digest_saved": self._digest_saved,
                }
                break

            # ── Process response blocks ────────────────────────────────
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # ── Handle stop reasons ────────────────────────────────────
            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in assistant_content if b.type == "text"]
                status = "completed" if self._digest_saved else "completed_no_digest"
                if not self._digest_saved:
                    logger.warning(
                        "[DigestAgent] Model finished without calling save_digest"
                    )
                logger.info(
                    f"[DigestAgent] {status} in {turn + 1} turns, "
                    f"{self._tool_calls} tool calls, "
                    f"{self._tokens_in} tokens in, {self._tokens_out} tokens out"
                )
                final_result = {
                    "status": status,
                    "turns": turn + 1,
                    "tool_calls": self._tool_calls,
                    "tokens_in": self._tokens_in,
                    "tokens_out": self._tokens_out,
                    "duration_sec": round(time.monotonic() - self._start_time, 2),
                    "cost_usd": self._estimate_cost(),
                    "digest_saved": self._digest_saved,
                    "final_text": "\n".join(text_blocks) if text_blocks else None,
                }
                break

            if response.stop_reason == "max_tokens":
                logger.error(
                    f"[DigestAgent] Hit max_tokens on turn {turn + 1}. "
                    f"Response may be truncated. Aborting."
                )
                final_result = {
                    "status": "error",
                    "reason": "max_tokens",
                    "turns": turn + 1,
                    "tool_calls": self._tool_calls,
                    "tokens_in": self._tokens_in,
                    "tokens_out": self._tokens_out,
                    "duration_sec": round(time.monotonic() - self._start_time, 2),
                    "cost_usd": self._estimate_cost(),
                    "digest_saved": self._digest_saved,
                }
                break

            # ── Execute tool calls ─────────────────────────────────────
            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                self._tool_calls += 1
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                logger.info(
                    f"[DigestAgent] Turn {turn + 1} → {tool_name}("
                    f"{json.dumps(tool_input, default=str)[:120]})"
                )

                result = self._execute_tool(tool_name, tool_input)

                # Track errors
                if isinstance(result, dict) and "error" in result:
                    consecutive_errors += 1
                    total_errors += 1
                    logger.warning(
                        f"[DigestAgent] Tool error ({consecutive_errors}/{MAX_TOOL_ERRORS}): "
                        f"{result['error']}"
                    )
                    if consecutive_errors >= MAX_TOOL_ERRORS:
                        logger.error("[DigestAgent] Too many consecutive tool errors, aborting")
                        return self._make_error_result(
                            "too_many_tool_errors",
                            f"{total_errors} total errors, {consecutive_errors} consecutive",
                            turn + 1,
                        )
                else:
                    consecutive_errors = 0

                # Serialize + truncate large results
                result_json = json.dumps(result, default=str)
                if len(result_json) > TOOL_RESULT_MAX_CHARS:
                    original_len = len(result_json)
                    result_json = result_json[:TOOL_RESULT_MAX_CHARS] + (
                        f'\n... [TRUNCATED — full result was {original_len:,} chars]'
                    )
                    logger.info(
                        f"[DigestAgent] Truncated {tool_name} result "
                        f"from {original_len:,} to {TOOL_RESULT_MAX_CHARS:,} chars"
                    )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_json,
                })

            # Feed tool results back
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            # Exhausted MAX_TURNS
            logger.warning(f"[DigestAgent] Hit max turns ({MAX_TURNS})")
            final_result = {
                "status": "max_turns",
                "turns": MAX_TURNS,
                "tool_calls": self._tool_calls,
                "tokens_in": self._tokens_in,
                "tokens_out": self._tokens_out,
                "duration_sec": round(time.monotonic() - self._start_time, 2),
                "cost_usd": self._estimate_cost(),
                "digest_saved": self._digest_saved,
            }

        return final_result

    # ── Tool execution ──────────────────────────────────────────────────

    def _execute_tool(self, name: str, params: dict) -> dict | list:
        """Dispatch a tool call. Special-cases save_digest for metadata injection."""

        if name == "save_digest":
            return self._handle_save_digest(params)

        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}

        # All dispatch functions take (client, **kwargs)
        return fn(self._sb, **params)

    def _handle_save_digest(self, params: dict) -> dict:
        """Inject agent metadata and optionally dry-run the save."""
        duration = round(time.monotonic() - self._start_time, 2)
        cost = self._estimate_cost()

        if self.dry_run:
            logger.info("[DigestAgent] DRY RUN — skipping Supabase save")
            from pathlib import Path
            try:
                digest_dir = Path(__file__).parent.parent.parent / "logs" / "digests"
                digest_dir.mkdir(parents=True, exist_ok=True)
                md_path = digest_dir / f"{params.get('date', 'unknown')}_{params.get('digest_type', 'unknown')}_DRYRUN.md"
                md_path.write_text(params.get("markdown", ""))
                logger.info(f"[DigestAgent] Dry-run markdown: {md_path}")
            except Exception as e:
                logger.warning(f"[DigestAgent] Failed to write dry-run markdown: {e}")

            self._digest_saved = True  # dry_run counts as "digest handled"
            return {
                "saved": False,
                "dry_run": True,
                "cost_usd": cost,
                "duration_sec": duration,
                "tokens_in": self._tokens_in,
                "tokens_out": self._tokens_out,
            }

        result = save_digest(
            self._sb,
            date=params.get("date", self.target_date),
            digest_type=params.get("digest_type", self.mode),
            content=params.get("content", {}),
            markdown=params.get("markdown", ""),
            model=self.model,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            cost_usd=cost,
            duration_sec=duration,
            tool_calls=self._tool_calls,
        )
        if result.get("saved"):
            self._digest_saved = True
        return result

    # ── Internals ───────────────────────────────────────────────────────

    def _make_error_result(self, reason: str, message: str, turn: int) -> dict:
        """Build a standardized error result dict."""
        return {
            "status": "error",
            "reason": reason,
            "message": message,
            "turns": turn,
            "tool_calls": self._tool_calls,
            "tokens_in": self._tokens_in,
            "tokens_out": self._tokens_out,
            "duration_sec": round(time.monotonic() - self._start_time, 2),
            "cost_usd": self._estimate_cost(),
            "digest_saved": self._digest_saved,
        }

    def _init_clients(self):
        """Initialize Anthropic + Supabase clients from env or explicit args."""
        import os

        # Load .env if it exists (for standalone CLI usage)
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            _load_dotenv(env_path)

        api_key = self._anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to live_trading/.env or pass explicitly."
            )
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=3)

        sb_url = self._sb_url or os.environ.get("SUPABASE_URL")
        sb_key = self._sb_key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not sb_url or not sb_key:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_SERVICE_KEY not set. "
                "Add them to live_trading/.env or pass explicitly."
            )
        self._sb = create_client(sb_url, sb_key)

    def _build_kickoff_message(self) -> str:
        """First user message that tells the model what date to analyze."""
        now_et = datetime.now(_ET)
        day_name = datetime.strptime(self.target_date, "%Y-%m-%d").strftime("%A")

        if self.mode == "eod":
            return (
                f"Generate the end-of-day digest for {self.target_date} ({day_name}). "
                f"Current time: {now_et.strftime('%H:%M ET')}. "
                f"Follow your analysis process step by step, calling the tools you need. "
                f"End by calling save_digest with your structured analysis and markdown."
            )
        else:
            return (
                f"Generate the morning briefing for {self.target_date} ({day_name}). "
                f"Current time: {now_et.strftime('%H:%M ET')}. "
                f"Follow your briefing process step by step, calling the tools you need. "
                f"End by calling save_digest with your structured analysis and markdown."
            )

    def _estimate_cost(self) -> float:
        """Rough cost estimate based on current Anthropic pricing."""
        if "opus" in self.model:
            cost_in = self._tokens_in * 15.0 / 1_000_000
            cost_out = self._tokens_out * 75.0 / 1_000_000
        elif "haiku" in self.model:
            cost_in = self._tokens_in * 0.80 / 1_000_000
            cost_out = self._tokens_out * 4.0 / 1_000_000
        else:  # sonnet
            cost_in = self._tokens_in * 3.0 / 1_000_000
            cost_out = self._tokens_out * 15.0 / 1_000_000
        return round(cost_in + cost_out, 4)


def _load_dotenv(path: str):
    """Minimal .env loader — no external dependency needed."""
    import os
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Handle `export VAR=value` syntax
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                # Strip surrounding quotes (single or double)
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if not key:
                    continue
                if not os.environ.get(key):  # don't override existing
                    os.environ[key] = value
    except FileNotFoundError:
        pass  # .env is optional
    except Exception as e:
        logger.warning(f"[DigestAgent] Failed to parse .env at {path}: {e}")
