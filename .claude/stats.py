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
"""Dump stats from Claude Code JSONL session logs.

Usage:
  python3 stats.py                     # all *.jsonl in same directory
  python3 stats.py sync_260316.jsonl   # specific file(s)
  python3 stats.py --dir /path/to/dir  # different directory
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import timedelta
from collections import defaultdict


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_dur(ms: int) -> str:
    if ms < 0:
        return f"-{fmt_dur(-ms)}"
    h, rem = divmod(ms // 1000, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def fmt_cost(usd: float) -> str:
    return f"${usd:.2f}"


# ── pricing (Anthropic public pricing, approximate) ───────────────────────────

PRICING = {
    # Model string as it appears in the JSONL 'model' field
    "claude-opus-4-6": {
        "input":          15.00 / 1e6,
        "output":         75.00 / 1e6,
        "cache_read":      1.50 / 1e6,
        "cache_create_5m": 3.75 / 1e6,
        "cache_create_1h":18.75 / 1e6,
    },
    "claude-haiku-4-5-20251001": {
        "input":           0.80 / 1e6,
        "output":          4.00 / 1e6,
        "cache_read":      0.08 / 1e6,
        "cache_create_5m": 1.00 / 1e6,
        "cache_create_1h": 5.00 / 1e6,
    },
}


def msg_raw_cost(model: str, usage: dict) -> float:
    """Estimate cost of a single API call from its usage block."""
    p = PRICING.get(model)
    if not p:
        return 0.0
    cc = usage.get("cache_creation", {})
    return (
        usage.get("input_tokens", 0)              * p["input"] +
        usage.get("output_tokens", 0)             * p["output"] +
        usage.get("cache_read_input_tokens", 0)   * p["cache_read"] +
        cc.get("ephemeral_5m_input_tokens", 0)    * p["cache_create_5m"] +
        cc.get("ephemeral_1h_input_tokens", 0)    * p["cache_create_1h"]
    )


# ── phase detection ───────────────────────────────────────────────────────────

# Ordered list of (label, text-trigger) for phase detection.
# We scan assistant 'text' content blocks for the first matching trigger.
PHASE_TRIGGERS = [
    ("Phase 1: Merge Upstream",        "## Phase 1"),
    ("Phase 2: Resolve Conflicts",     "## Phase 2"),
    ("Phase 3: Mirror CI",             "## Phase 3"),
    ("Phase 4: Build TensorFlow",      "## Phase 4"),
    ("Phase 5: Check Excluded Tests",  "## Phase 5"),
    ("Phase 6: Run RBE Tests",         "## Phase 6"),
    ("Phase 7: Triage",                "## Phase 7"),
    ("Phase 8: CUDA-only Tags",        "## Phase 8"),
    ("Phase 9",                        "## Phase 9"),
]


def detect_phase(text: str) -> str | None:
    for label, trigger in PHASE_TRIGGERS:
        if trigger in text:
            return label
    return None


# ── core parser ───────────────────────────────────────────────────────────────

class Session:
    """One contiguous init→result block."""
    def __init__(self, init_rec: dict):
        self.model = init_rec.get("model", "")
        self.session_id = init_rec.get("session_id", "")
        self.records: list[dict] = []
        self.result: dict = {}

    # Accumulated token usage across all assistant messages in this session
    @property
    def usage(self):
        u = defaultdict(int)
        for r in self.records:
            if r.get("type") != "assistant":
                continue
            m = r.get("message", {})
            mu = m.get("usage", {})
            u["input"]    += mu.get("input_tokens", 0)
            u["output"]   += mu.get("output_tokens", 0)
            u["cache_r"]  += mu.get("cache_read_input_tokens", 0)
            cc = mu.get("cache_creation", {})
            u["cache_c"]  += cc.get("ephemeral_5m_input_tokens", 0) + cc.get("ephemeral_1h_input_tokens", 0)
        return dict(u)

    @property
    def raw_cost(self) -> float:
        total = 0.0
        for r in self.records:
            if r.get("type") != "assistant":
                continue
            m = r.get("message", {})
            total += msg_raw_cost(m.get("model", ""), m.get("usage", {}))
        return total

    @property
    def tool_calls(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in self.records:
            if r.get("type") != "assistant":
                continue
            for c in r.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    counts[c.get("name", "?")] += 1
        return dict(counts)

    @property
    def skill_calls(self) -> list[dict]:
        out = []
        for r in self.records:
            if r.get("type") != "assistant":
                continue
            for c in r.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use" and c["name"] == "Skill":
                    out.append(c.get("input", {}))
        return out

    @property
    def agent_calls(self) -> list[dict]:
        out = []
        for r in self.records:
            if r.get("type") != "assistant":
                continue
            for c in r.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use" and c["name"] == "Agent":
                    out.append({"input": c.get("input", {}), "id": c.get("id", "")})
        return out

    @property
    def duration_ms(self) -> int:
        return self.result.get("duration_ms", 0)

    @property
    def turns(self) -> int:
        return self.result.get("num_turns", 0)

    @property
    def is_main(self) -> bool:
        return self.turns > 5


def parse_sessions(records: list[dict]) -> list[Session]:
    """Split records into Session objects at each system/init boundary."""
    sessions = []
    current: Session | None = None
    for r in records:
        if r.get("type") == "system" and r.get("subtype") == "init":
            if current is not None:
                sessions.append(current)
            current = Session(r)
        elif current is not None:
            current.records.append(r)
            if r.get("type") == "result":
                current.result = r
    if current is not None:
        sessions.append(current)
    return sessions


def enrich_agent_calls(sessions: list[Session]) -> None:
    """Attach task_notification data to agent_calls."""
    # Build map: tool_use_id → task_notification record
    notif_map = {}
    for s in sessions:
        for r in s.records:
            if r.get("type") == "system" and r.get("subtype") == "task_notification":
                notif_map[r.get("tool_use_id", "")] = r

    for s in sessions:
        for ac in s.agent_calls:
            notif = notif_map.get(ac["id"], {})
            if notif:
                usage = notif.get("usage", {})
                ac["status"]       = notif.get("status", "")
                ac["total_tokens"] = usage.get("total_tokens", 0)
                ac["tool_uses"]    = usage.get("tool_uses", 0)
                ac["duration_ms"]  = usage.get("duration_ms", 0)
                ac["summary"]      = notif.get("summary", "")


# ── phase breakdown ───────────────────────────────────────────────────────────

class PhaseStats:
    def __init__(self, name: str):
        self.name = name
        self.n_msgs = 0
        self.in_tok = self.out_tok = self.cache_r = self.cache_c = 0
        self.n_tools = 0
        self.raw_cost = 0.0

    def add_msg(self, model: str, usage: dict, n_tools: int = 0):
        self.n_msgs += 1
        self.in_tok   += usage.get("input_tokens", 0)
        self.out_tok  += usage.get("output_tokens", 0)
        self.cache_r  += usage.get("cache_read_input_tokens", 0)
        cc = usage.get("cache_creation", {})
        self.cache_c  += cc.get("ephemeral_5m_input_tokens", 0) + cc.get("ephemeral_1h_input_tokens", 0)
        self.n_tools  += n_tools
        self.raw_cost += msg_raw_cost(model, usage)


def compute_phases(session: Session) -> list[PhaseStats]:
    """Segment a main session into phases based on text markers."""
    phases: list[PhaseStats] = [PhaseStats("Pre-flight / Reading Skills")]
    current = phases[0]

    for r in session.records:
        if r.get("type") != "assistant":
            continue
        msg = r.get("message", {})
        model = msg.get("model", "")
        usage = msg.get("usage", {})
        content = msg.get("content", [])

        # Count tool calls in this message
        n_tools = sum(1 for c in content
                      if isinstance(c, dict) and c.get("type") == "tool_use")

        # Check for phase transition in text blocks
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                label = detect_phase(c.get("text", ""))
                if label and label != current.name:
                    new_phase = PhaseStats(label)
                    phases.append(new_phase)
                    current = new_phase
                    break

        if usage:
            current.add_msg(model, usage, n_tools)

    return phases


# ── display ───────────────────────────────────────────────────────────────────

def print_phase_breakdown(phases: list[PhaseStats], actual_total: float):
    """Print cost breakdown by phase, scaled to match the authoritative total."""
    raw_total = sum(p.raw_cost for p in phases)
    scale = actual_total / raw_total if raw_total > 0 else 1.0

    col = "{:<33}  {:>4}  {:>6}  {:>6}  {:>9}  {:>9}  {:>5}  {:>8}"
    print(col.format("Phase", "Msgs", "Input", "Output", "CacheR", "CacheC", "Tools", "Cost"))
    print("─" * 93)
    for p in phases:
        cost = p.raw_cost * scale
        print(col.format(
            p.name[:33],
            p.n_msgs,
            fmt_tok(p.in_tok),
            fmt_tok(p.out_tok),
            fmt_tok(p.cache_r),
            fmt_tok(p.cache_c),
            p.n_tools,
            fmt_cost(cost),
        ))
    print("─" * 93)
    print(col.format("TOTAL", "", "", "", "", "", "", fmt_cost(actual_total)))
    if abs(scale - 1.0) > 0.01:
        print(f"  (pricing scale factor: {scale:.3f} — applied to match authoritative result total)")


def print_file_stats(path: str, verbose: bool = False) -> dict:
    with open(path) as f:
        records = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    sessions = parse_sessions(records)
    enrich_agent_calls(sessions)

    # The authoritative cost/duration is in the last result record's cumulative totals.
    # Each result.total_cost_usd is a RUNNING CUMULATIVE — use the last one as the grand total.
    all_results = [r for r in records if r.get("type") == "result"]
    if not all_results:
        print(f"\nNo result records found in {os.path.basename(path)}")
        return {}

    last_result = all_results[-1]
    actual_total_cost = last_result.get("total_cost_usd", 0.0)
    # duration and turns: sum the INCREMENTAL values (each result.duration_ms is for that sub-session)
    total_duration_ms = sum(r.get("duration_ms", 0) for r in all_results)
    total_turns = sum(r.get("num_turns", 0) for r in all_results)

    # Per-model totals from the last result's modelUsage (also cumulative)
    model_usage = last_result.get("modelUsage", {})

    # Main session vs follow-up sessions
    main_sessions = [s for s in sessions if s.is_main]
    followup_sessions = [s for s in sessions if not s.is_main]

    print(f"\n{'═'*65}")
    print(f"  File: {os.path.basename(path)}")
    print(f"{'═'*65}")
    print(f"  Sessions:    {len(sessions)}  "
          f"({len(main_sessions)} main, {len(followup_sessions)} follow-up)")

    # Duration
    print(f"\n  Duration")
    print(f"    Wall time:   {fmt_dur(total_duration_ms)}")
    print(f"    Turns:       {total_turns}")

    # Cost
    print(f"\n  Cost (authoritative)")
    print(f"    Total:       {fmt_cost(actual_total_cost)}")
    for model, mu in sorted(model_usage.items()):
        cost = mu.get("costUSD", 0.0)
        pct = cost / actual_total_cost * 100 if actual_total_cost else 0
        print(f"    {model:<28} {fmt_cost(cost):>8}  ({pct:.0f}%)")

    # Tokens (from last result's modelUsage — cumulative session totals)
    total_in  = sum(mu.get("inputTokens", 0) for mu in model_usage.values())
    total_out = sum(mu.get("outputTokens", 0) for mu in model_usage.values())
    total_cr  = sum(mu.get("cacheReadInputTokens", 0) for mu in model_usage.values())
    total_cc  = sum(mu.get("cacheCreationInputTokens", 0) for mu in model_usage.values())
    print(f"\n  Tokens")
    print(f"    Input:              {fmt_tok(total_in):>9}")
    print(f"    Output:             {fmt_tok(total_out):>9}")
    print(f"    Cache read:         {fmt_tok(total_cr):>9}")
    print(f"    Cache creation:     {fmt_tok(total_cc):>9}")

    # Tool calls across all sessions
    all_tools: dict[str, int] = defaultdict(int)
    all_skills: list[dict] = []
    all_agents: list[dict] = []
    for s in sessions:
        for name, count in s.tool_calls.items():
            all_tools[name] += count
        all_skills.extend(s.skill_calls)
        all_agents.extend(s.agent_calls)

    total_tool_calls = sum(all_tools.values())
    print(f"\n  Tool calls   ({total_tool_calls} total)")
    for name, count in sorted(all_tools.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 35)
        print(f"    {name:<22} {count:>4}  {bar}")

    if all_skills:
        print(f"\n  Skill calls  ({len(all_skills)} total)")
        skill_counts: dict[str, int] = defaultdict(int)
        for sc in all_skills:
            skill_counts[sc.get("skill", "?")] += 1
        for skill, count in sorted(skill_counts.items(), key=lambda x: -x[1]):
            print(f"    {skill:<22} {count:>4}")
    else:
        print(f"\n  Skill calls  0")

    if all_agents:
        print(f"\n  Agent calls  ({len(all_agents)} total)")
        for ac in all_agents:
            inp = ac.get("input", {})
            atype = inp.get("subagent_type", "")
            desc = inp.get("description", "")[:48]
            status = ac.get("status", "")
            dur_a = ac.get("duration_ms", 0)
            toks = ac.get("total_tokens", 0)
            uses = ac.get("tool_uses", 0)
            status_tag = f"[{status}] " if status else ""
            print(f"    [{atype}] {status_tag}{desc}")
            if dur_a:
                print(f"      duration={fmt_dur(dur_a)}  tokens={fmt_tok(toks)}  tool_uses={uses}")
    else:
        print(f"\n  Agent calls  0")

    # Phase breakdown (main sessions only)
    if main_sessions:
        # Cost attributable to follow-up sessions
        followup_raw = sum(s.raw_cost for s in followup_sessions)
        main_raw = sum(s.raw_cost for s in main_sessions)
        grand_raw = main_raw + followup_raw

        # Scale factor so everything sums to actual_total_cost
        scale = actual_total_cost / grand_raw if grand_raw > 0 else 1.0
        followup_cost = followup_raw * scale

        for i, ms in enumerate(main_sessions):
            label = f"  Phase breakdown — main session {i+1}" if len(main_sessions) > 1 else "  Phase breakdown — main session"
            if len(followup_sessions) > 0:
                ms_actual = ms.raw_cost * scale
            else:
                ms_actual = actual_total_cost
            print(f"\n{label}  ({fmt_cost(ms_actual)}, {ms.turns} turns, {fmt_dur(ms.duration_ms)})")
            phases = compute_phases(ms)
            print_phase_breakdown(phases, ms_actual)

        if followup_sessions and followup_cost > 0.01:
            print(f"\n  Follow-up sessions ({len(followup_sessions)} sub-sessions, ~{fmt_cost(followup_cost)})")
            print(f"    These are short sessions triggered by background task notifications")
            print(f"    (each ~{fmt_cost(followup_cost/len(followup_sessions))}, "
                  f"{sum(s.turns for s in followup_sessions)} turns total)")

    return {
        "file": os.path.basename(path),
        "total_cost": actual_total_cost,
        "total_duration_ms": total_duration_ms,
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
        "n_agents": len(all_agents),
        "n_skills": len(all_skills),
        "model_usage": model_usage,
    }


def print_aggregate(all_file_stats: list[dict]):
    if len(all_file_stats) < 2:
        return

    total_cost = sum(s["total_cost"] for s in all_file_stats)
    total_dur = sum(s["total_duration_ms"] for s in all_file_stats)
    total_turns = sum(s["total_turns"] for s in all_file_stats)
    total_tools = sum(s["total_tool_calls"] for s in all_file_stats)
    total_agents = sum(s["n_agents"] for s in all_file_stats)
    total_skills = sum(s["n_skills"] for s in all_file_stats)

    print(f"\n{'═'*65}")
    print(f"  AGGREGATE  ({len(all_file_stats)} files)")
    print(f"{'═'*65}")
    print(f"\n  {'File':<28}  {'Cost':>8}  {'Wall time':>10}  {'Turns':>6}")
    print("  " + "─" * 58)
    for s in all_file_stats:
        print(f"  {s['file']:<28}  {fmt_cost(s['total_cost']):>8}  "
              f"{fmt_dur(s['total_duration_ms']):>10}  {s['total_turns']:>6}")
    print("  " + "─" * 58)
    print(f"  {'TOTAL':<28}  {fmt_cost(total_cost):>8}  "
          f"{fmt_dur(total_dur):>10}  {total_turns:>6}")

    print(f"\n  Tool calls:  {total_tools}")
    print(f"  Agent calls: {total_agents}")
    print(f"  Skill calls: {total_skills}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*",
                        help="JSONL files to analyze (default: all *.jsonl here)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dir", default=None,
                        help="Directory to search for JSONL files")
    args = parser.parse_args()

    files = args.files
    if not files:
        search_dir = args.dir or os.path.dirname(os.path.abspath(__file__))
        files = sorted(Path(search_dir).glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {search_dir}")
            sys.exit(1)

    all_stats = []
    for f in files:
        path = str(f)
        if not os.path.exists(path):
            print(f"File not found: {path}", file=sys.stderr)
            continue
        s = print_file_stats(path, verbose=args.verbose)
        if s:
            all_stats.append(s)

    if len(all_stats) > 1:
        print_aggregate(all_stats)

    print()


if __name__ == "__main__":
    main()
