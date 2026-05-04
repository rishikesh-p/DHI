import pytest
import re
from dhi.agent.graph import parse_llm_output

def test_parse_llm_output_basic_bash():
    response = "Here is the command you requested:\n```bash\necho 'hello'\n```\nDone."
    assert parse_llm_output(response) == "echo 'hello'"

def test_parse_llm_output_sh_block():
    response = "```sh\npwd\n```"
    assert parse_llm_output(response) == "pwd"

def test_parse_llm_output_no_backticks_returns_none():
    """Raw text without a code block must NOT be treated as a command.
    This prevents executing arbitrary LLM prose like 'Sure! Try rm -rf /'."""
    response = "ls -la"
    assert parse_llm_output(response) is None

def test_parse_llm_output_prose_returns_none():
    """Multi-line prose without code blocks should return None."""
    response = "Sure, I can help you with that.\nHere is what you should do:\nFirst, open a terminal."
    assert parse_llm_output(response) is None

def test_parse_llm_output_multiple_blocks():
    # It should grab the first block
    response = "```bash\ncd /tmp\n```\nand then\n```bash\nrm -rf *\n```"
    assert parse_llm_output(response) == "cd /tmp"

def test_parse_llm_output_empty():
    assert parse_llm_output("") is None
    assert parse_llm_output("   \n  ") is None

# --- Dangerous command detection tests ---
# These test the regex patterns used in node_executor

DANGEROUS_PATTERNS = [
    r'\brm\b', r'\bmv\b', r'\bdd\b', r'\bmkfs\b',
    r'\bchmod\b', r'\bchown\b', r'\bkillall\b', r'\bpkill\b',
    r'\bshred\b', r'\breboot\b', r'\bshutdown\b', r'\bpoweroff\b',
]

def _is_dangerous(cmd):
    return any(re.search(p, cmd) for p in DANGEROUS_PATTERNS)

def test_dangerous_cmd_at_start():
    assert _is_dangerous("rm -rf /tmp/foo")

def test_dangerous_cmd_in_pipe():
    """Commands inside pipes must be caught."""
    assert _is_dangerous("echo foo | rm -rf /")

def test_dangerous_cmd_in_chain():
    """Commands after && or ; must be caught."""
    assert _is_dangerous("echo foo && rm -rf /")
    assert _is_dangerous("ls; shutdown -h now")

def test_dangerous_cmd_in_subshell():
    assert _is_dangerous("$(rm -rf /)")

def test_safe_cmd_not_flagged():
    """Normal commands should NOT be flagged."""
    assert not _is_dangerous("ls -la")
    assert not _is_dangerous("echo hello")
    assert not _is_dangerous("cat /etc/os-release")
    assert not _is_dangerous("find . -name '*.py'")

def test_safe_cmd_containing_substring():
    """'rm' inside a word like 'format' should NOT trigger."""
    assert not _is_dangerous("echo 'format the drive'")
    assert not _is_dangerous("echo 'inform the user'")
