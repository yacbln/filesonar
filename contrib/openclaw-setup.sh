#!/usr/bin/env bash
# Set up filesonar for OpenClaw:
#   1. register the filesonar MCP server (if it is not registered yet)
#   2. install the filesonar skill, so the agent searches with filesonar
#      instead of exploring folders with ls/grep
#   3. optionally (--read-only-approvals) let read-only shell commands run
#      without an approval prompt. Anything else - deleting, moving, writing,
#      downloading - still asks, as does any command with a redirection (>).
#
# Usage:  contrib/openclaw-setup.sh [--read-only-approvals] [--agent main]
# Use a specific OpenClaw binary with:  OPENCLAW=/path/to/openclaw contrib/openclaw-setup.sh
set -euo pipefail

OPENCLAW="${OPENCLAW:-openclaw}"
export NODE_NO_WARNINGS=1  # OpenClaw prints an experimental-SQLite warning per call
AGENT="main"
READ_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --read-only-approvals) READ_ONLY=1 ;;
    --agent) AGENT="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
filesonar_bin="$(command -v filesonar || true)"
if [[ -z "$filesonar_bin" ]]; then
  echo "filesonar is not on PATH; install it first (see README)" >&2
  exit 1
fi

echo "== MCP server"
if "$OPENCLAW" mcp show filesonar >/dev/null 2>&1; then
  echo "filesonar is already registered"
else
  "$OPENCLAW" mcp add filesonar --command "$filesonar_bin" --arg serve --timeout 60
fi

echo "== Skill"
workspace="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
mkdir -p "$workspace/skills/filesonar"
cp "$here/../skill/SKILL.md" "$workspace/skills/filesonar/SKILL.md"
echo "installed $workspace/skills/filesonar/SKILL.md"

if [[ "$READ_ONLY" == 1 ]]; then
  echo "== Read-only commands that no longer need approval (agent: $AGENT)"
  # Only programs that cannot change files on their own. Deliberately absent:
  # find (-delete, -exec), sed/awk (in-place edits), curl/wget, cp/mv/rm, editors,
  # interpreters (python, node, bash). Allowlist mode rejects redirections, so
  # "cat x > y" still asks.
  for name in ls cat head tail grep wc file stat du df pwd realpath readlink \
              basename dirname date whoami uname tree; do
    path="$(command -v "$name" 2>/dev/null || true)"
    case "$path" in
      /usr/bin/*|/bin/*)
        # An earlier "allow always" may have stored this program for one exact
        # command line; OpenClaw then treats the pattern as present and skips the
        # add. Replace such entries with one that covers any arguments.
        "$OPENCLAW" approvals allowlist remove --agent "$AGENT" "$path" >/dev/null
        "$OPENCLAW" approvals allowlist add --agent "$AGENT" "$path" >/dev/null
        echo "  $path" ;;
      *) ;;  # not installed, or not a system binary: skip
    esac
  done
fi

echo
echo "Done. Check with:  $OPENCLAW mcp probe filesonar"
[[ "$READ_ONLY" == 1 ]] && echo "                   $OPENCLAW approvals get"
echo "Undo an approval:  $OPENCLAW approvals allowlist remove --agent $AGENT /usr/bin/ls"
