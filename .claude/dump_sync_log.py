#!/usr/bin/env python3
# ==============================================================================
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
dump_sync_log.py — pretty-print a Claude sync session from a .jsonl event log.

Usage:
    python3 .claude/dump_sync_log.py logs/sync_260312.jsonl
    python3 .claude/dump_sync_log.py logs/sync_260312.jsonl --mode reasoning
    python3 .claude/dump_sync_log.py logs/sync_260312.jsonl --tool-results all

Modes:
    full       Show everything: Claude text, all tool calls + results (default)
    reasoning  Show only Claude prose + subagent calls/results (skip bash/read/edit noise)
    summary    Show only Claude prose (no tool calls or results)

--tool-results:
    agents     Show results only for Agent subagent calls (default)
    all        Show results for every tool call
    none       Never show tool results
"""

import json
import sys
import argparse
import textwrap
from datetime import timedelta

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
MAGENTA= "\033[35m"

def c(color, text):
    return f"{color}{text}{RESET}"

# ── Helpers ───────────────────────────────────────────────────────────────────
def wrap(text, indent=4, width=100):
    lines = text.splitlines()
    out = []
    for line in lines:
        if len(line) <= width - indent:
            out.append(" " * indent + line)
        else:
            wrapped = textwrap.wrap(line, width=width - indent,
                                    break_long_words=False, break_on_hyphens=False)
            out.extend(" " * indent + l for l in (wrapped or [line]))
    return "\n".join(out)

def hr(char="─", width=100):
    return char * width

def fmt_duration(ms):
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"

def fmt_cost(usd):
    return f"${usd:.4f}" if usd is not None else "$?"

def extract_text(content):
    """Extract plain text from a tool_result content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)

# ── Tool categories ───────────────────────────────────────────────────────────
NOISE_TOOLS = {"TodoWrite", "TaskUpdate", "TaskCreate", "TaskList", "TaskGet"}
READ_TOOLS  = {"Read", "Glob", "Grep"}
EXEC_TOOLS  = {"Bash", "Edit", "Write"}

def tool_color(name):
    if name == "Agent":       return MAGENTA
    if name in EXEC_TOOLS:    return YELLOW
    if name in READ_TOOLS:    return CYAN
    if name in NOISE_TOOLS:   return DIM
    return BLUE

# ── Main printer ──────────────────────────────────────────────────────────────
def dump(jsonl_path, mode="full", tool_results_mode="agents"):
    with open(jsonl_path) as f:
        events = [json.loads(l) for l in f if l.strip()]

    # Build id→(name, input) map for labelling tool results
    tool_map = {}   # tool_use_id → {"name": ..., "input": ...}
    for e in events:
        if e.get("type") == "assistant":
            for block in e.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_map[block["id"]] = {
                        "name":  block["name"],
                        "input": block.get("input", {}),
                    }

    # ── Session header ────────────────────────────────────────────────────────
    init = next((e for e in events if e.get("type") == "system"
                                   and e.get("subtype") == "init"), None)
    result = next((e for e in events if e.get("type") == "result"), None)

    model   = init.get("model", "?") if init else "?"
    sess_id = (init.get("session_id", "") if init else "")[:8]

    print()
    print(c(BOLD + CYAN, hr("═")))
    print(c(BOLD + CYAN, f"  SYNC SESSION  {model}  [{sess_id}]"))
    if result:
        dur  = fmt_duration(result.get("duration_ms", 0))
        cost = fmt_cost(result.get("total_cost_usd"))
        turns = result.get("num_turns", "?")
        mu = result.get("modelUsage", {})
        tokens_out = sum(v.get("outputTokens", 0) for v in mu.values())
        tokens_in  = sum(v.get("inputTokens", 0) + v.get("cacheReadInputTokens", 0)
                         for v in mu.values())
        print(c(CYAN, f"  duration={dur}  turns={turns}  "
                       f"cost={cost}  in={tokens_in:,}tok  out={tokens_out:,}tok"))
    print(c(BOLD + CYAN, hr("═")))
    print()

    # ── Walk events ───────────────────────────────────────────────────────────
    for e in events:
        etype = e.get("type")

        # ── system: compact boundary / continuation summary ──────────────────
        if etype == "system" and e.get("subtype") == "compact_boundary":
            print(c(DIM, hr("·")))
            print(c(DIM, "  [context compacted — conversation summary injected]"))
            print(c(DIM, hr("·")))
            print()
            continue

        # ── assistant event ──────────────────────────────────────────────────
        if etype == "assistant":
            for block in e.get("message", {}).get("content", []):

                # Claude's prose text
                if block.get("type") == "text":
                    text = block["text"].strip()
                    if not text:
                        continue
                    print(c(BOLD + GREEN, "▌ CLAUDE"))
                    print(wrap(text, indent=2))
                    print()

                # Tool call
                elif block.get("type") == "tool_use":
                    name  = block["name"]
                    inp   = block.get("input", {})
                    tid   = block["id"]
                    col   = tool_color(name)

                    # summary: no tool calls at all
                    if mode == "summary":
                        continue
                    # reasoning: only show Agent (subagent) calls, skip everything else
                    if mode == "reasoning" and name != "Agent":
                        continue
                    # full: skip only pure noise tools (todo trackers)
                    if mode == "full" and name in NOISE_TOOLS:
                        continue

                    if name == "Agent":
                        # Subagent — always show full prompt
                        desc   = inp.get("description", "")
                        prompt = inp.get("prompt", "")
                        subtype = inp.get("subagent_type", "")
                        model_override = inp.get("model", "")
                        print(c(BOLD + MAGENTA, f"▶ SUBAGENT  {desc}"))
                        if subtype:
                            print(c(MAGENTA, f"  type={subtype}"), end="")
                        if model_override:
                            print(c(MAGENTA, f"  model={model_override}"), end="")
                        print()
                        if prompt:
                            print(c(DIM, "  ── prompt ──"))
                            print(wrap(prompt, indent=4))
                        print()
                    else:
                        # Regular tool call
                        # Build a short human-readable summary of the input
                        if "command" in inp:
                            summary = inp["command"]
                            desc_str = inp.get("description", "")
                            label = desc_str if desc_str else summary
                        elif "file_path" in inp:
                            label = inp["file_path"]
                            if "offset" in inp or "limit" in inp:
                                label += f"  [lines {inp.get('offset',1)}–{inp.get('offset',1)+inp.get('limit',0)-1}]"
                        elif "pattern" in inp:
                            label = inp.get("pattern", "")
                            if "path" in inp:
                                label += f"  in {inp['path']}"
                        elif "prompt" in inp:
                            label = inp["prompt"][:120]
                        else:
                            label = json.dumps(inp)[:120]

                        print(c(col, f"  [{name}]  {label}"))

                        # Show full input if requested (all mode, or for key tools)
                        if (mode == "full" and tool_results_mode == "all"
                                and name not in NOISE_TOOLS):
                            full = json.dumps(inp, indent=6)
                            if len(full) > 2000:
                                full = full[:2000] + "\n      … (truncated)"
                            print(c(DIM, wrap(full, indent=6)))
                        print()

        # ── user event: tool results ─────────────────────────────────────────
        elif etype == "user":
            for block in e.get("message", {}).get("content", []):

                # Injected system text (continuation summaries)
                if block.get("type") == "text":
                    text = block["text"].strip()
                    if text.startswith("This session is being continued"):
                        print(c(DIM, "  [continuation summary — session was compacted]"))
                        print(wrap(text[:600] + ("…" if len(text) > 600 else ""),
                                   indent=4))
                        print()
                    continue

                if block.get("type") != "tool_result":
                    continue

                tid     = block.get("tool_use_id", "")
                tool    = tool_map.get(tid, {})
                tname   = tool.get("name", "?")
                content = extract_text(block.get("content", ""))
                is_err  = block.get("is_error", False)

                if mode == "summary":
                    continue
                if mode == "reasoning" and tname != "Agent":
                    continue
                if tool_results_mode == "none":
                    continue
                if tool_results_mode == "agents" and tname != "Agent":
                    continue
                if tname in NOISE_TOOLS and tool_results_mode != "all":
                    continue

                if tname == "Agent":
                    desc = tool.get("input", {}).get("description", "")
                    print(c(BOLD + MAGENTA, f"◀ SUBAGENT RESULT  {desc}"))
                    if is_err:
                        print(c(RED, "  ERROR"))
                    print(wrap(content, indent=4))
                    print()
                else:
                    col = RED if is_err else tool_color(tname)
                    tag = "ERROR" if is_err else "result"
                    print(c(col, f"  [{tname} {tag}]"))
                    # Trim very long results
                    if len(content) > 1500:
                        content = content[:1500] + "\n  … (truncated)"
                    print(wrap(content, indent=4))
                    print()

        # ── final result ─────────────────────────────────────────────────────
        elif etype == "result":
            print(c(BOLD + CYAN, hr("═")))
            sub   = e.get("subtype", "?")
            err   = e.get("is_error", False)
            col   = RED if err else GREEN
            print(c(BOLD + col, f"  DONE  {sub.upper()}"))
            if e.get("result"):
                print(wrap(e["result"], indent=4))
            print()
            # Cost breakdown per model
            for mname, mu in e.get("modelUsage", {}).items():
                print(c(DIM,
                    f"  {mname}: "
                    f"in={mu.get('inputTokens',0):,} "
                    f"cache_read={mu.get('cacheReadInputTokens',0):,} "
                    f"cache_write={mu.get('cacheCreationInputTokens',0):,} "
                    f"out={mu.get('outputTokens',0):,} "
                    f"cost={fmt_cost(mu.get('costUSD'))}"))
            print(c(BOLD + CYAN, hr("═")))


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print a Claude sync session JSONL log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("jsonl", help="Path to the .jsonl event log")
    parser.add_argument(
        "--mode", choices=["full", "reasoning", "summary"],
        default="full",
        help="full=everything  reasoning=prose+subagents  summary=prose only")
    parser.add_argument(
        "--tool-results", dest="tool_results",
        choices=["agents", "all", "none"],
        default="agents",
        help="Which tool results to show (default: agents only)")
    args = parser.parse_args()

    dump(args.jsonl, mode=args.mode, tool_results_mode=args.tool_results)

if __name__ == "__main__":
    main()
