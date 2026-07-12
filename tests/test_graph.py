import pytest
import re
from dhi.agent.graph import parse_llm_output, should_continue


# -- parse_llm_output --

class TestParseLlmOutput:
    """Test extraction of bash code blocks from LLM responses."""

    def test_basic_bash_block(self):
        """Verify a standard ```bash block is extracted."""
        response = "Here is the command:\n```bash\necho 'hello'\n```\nDone."
        assert parse_llm_output(response) == "echo 'hello'"

    def test_sh_block(self):
        """Verify ```sh blocks are also accepted."""
        assert parse_llm_output("```sh\npwd\n```") == "pwd"

    def test_python_block(self):
        """Verify ```python blocks are accepted."""
        assert parse_llm_output("```python\nprint('hi')\n```") == "print('hi')"

    def test_unlabeled_block(self):
        """Verify unlabeled ``` blocks are accepted."""
        assert parse_llm_output("```\nwhoami\n```") == "whoami"

    def test_multiline_command(self):
        """Verify multi-line commands inside a block are preserved."""
        response = "```bash\ncd /tmp\nls -la\necho done\n```"
        result = parse_llm_output(response)
        assert "cd /tmp" in result
        assert "echo done" in result

    def test_no_backticks_returns_none(self):
        """Verify raw text without a code block is rejected to prevent accidental execution."""
        assert parse_llm_output("ls -la") is None

    def test_prose_returns_none(self):
        """Verify multi-line prose without code blocks returns None."""
        response = "Sure, I can help.\nHere is what you should do:\nFirst, open a terminal."
        assert parse_llm_output(response) is None

    def test_multiple_blocks_returns_first(self):
        """Verify only the first code block is extracted."""
        response = "```bash\ncd /tmp\n```\nand then\n```bash\nrm -rf *\n```"
        assert parse_llm_output(response) == "cd /tmp"

    def test_empty_string(self):
        assert parse_llm_output("") is None

    def test_whitespace_only(self):
        assert parse_llm_output("   \n  ") is None

    def test_none_input(self):
        assert parse_llm_output(None) is None

    def test_empty_code_block(self):
        """Verify an empty code block is not treated as a valid command."""
        result = parse_llm_output("```bash\n\n```")
        assert not result  # Returns '' or None, both falsy.

    def test_block_with_surrounding_prose(self):
        """Verify code is extracted even when surrounded by verbose prose."""
        response = (
            "I'll help you with that. Here's a command that does what you need:\n\n"
            "```bash\nfind . -name '*.py' -type f\n```\n\n"
            "This will recursively search for Python files in the current directory."
        )
        assert parse_llm_output(response) == "find . -name '*.py' -type f"


# -- Dangerous command detection --

DANGEROUS_PATTERNS = [
    r'\brm\b', r'\bmv\b', r'\bdd\b', r'\bmkfs\b',
    r'\bchmod\b', r'\bchown\b', r'\bkillall\b', r'\bpkill\b',
    r'\bshred\b', r'\breboot\b', r'\bshutdown\b', r'\bpoweroff\b',
]


def _is_dangerous(cmd):
    return any(re.search(p, cmd) for p in DANGEROUS_PATTERNS)


class TestDangerousCommandDetection:
    """Test regex patterns that guard against destructive commands."""

    @pytest.mark.parametrize("cmd", [
        "rm -rf /tmp/foo",
        "mv important.txt /dev/null",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda1",
        "chmod 777 /etc/passwd",
        "chown root:root file.txt",
        "killall firefox",
        "pkill -9 python",
        "shred -u secret.txt",
        "reboot",
        "shutdown -h now",
        "poweroff",
    ])
    def test_detects_dangerous_command(self, cmd):
        """Verify each dangerous keyword is detected."""
        assert _is_dangerous(cmd), f"Should flag: {cmd}"

    def test_detects_dangerous_in_pipe(self):
        """Verify dangerous commands inside pipes are caught."""
        assert _is_dangerous("echo foo | rm -rf /")

    def test_detects_dangerous_in_chain(self):
        """Verify dangerous commands after && or ; are caught."""
        assert _is_dangerous("echo foo && rm -rf /")
        assert _is_dangerous("ls; shutdown -h now")

    def test_detects_dangerous_in_subshell(self):
        """Verify dangerous commands inside $() are caught."""
        assert _is_dangerous("$(rm -rf /)")

    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "echo hello",
        "cat /etc/os-release",
        "find . -name '*.py'",
        "grep -r 'pattern' .",
        "wc -l file.txt",
        "df -h",
        "ps aux",
        "uname -r",
    ])
    def test_safe_commands_not_flagged(self, cmd):
        """Verify normal commands are not flagged."""
        assert not _is_dangerous(cmd), f"Should not flag: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "echo 'format the drive'",
        "echo 'inform the user'",
        "echo 'removed items'",
        "echo 'chromium browser'",
    ])
    def test_substrings_not_flagged(self, cmd):
        """Verify dangerous keywords as substrings within words are ignored."""
        assert not _is_dangerous(cmd), f"Should not flag substring match: {cmd}"


# -- should_continue (retry logic) --

class TestShouldContinue:
    """Test the retry and fallback logic in the executor loop."""

    def _state(self, plan="local", error=None, retry_count=0, confidence=1.0):
        """Build a minimal AgentState dict."""
        return {
            "messages": [],
            "input_text": "test",
            "plan": plan,
            "command": "echo test",
            "command_output": "",
            "error": error,
            "retry_count": retry_count,
            "route_confidence": confidence,
        }

    def test_no_error_ends(self):
        """Verify the loop ends when there is no error."""
        assert should_continue(self._state(error=None)) == "end"

    def test_local_retries_within_limit(self):
        """Verify local route retries while under the retry limit."""
        assert should_continue(self._state(error="fail", retry_count=0)) == "retry_local"
        assert should_continue(self._state(error="fail", retry_count=1)) == "retry_local"
        assert should_continue(self._state(error="fail", retry_count=2)) == "retry_local"

    def test_cloud_retries_within_limit(self):
        """Verify cloud route retries while under the retry limit."""
        assert should_continue(self._state(plan="cloud", error="fail", retry_count=0)) == "retry_cloud"
        assert should_continue(self._state(plan="cloud", error="fail", retry_count=1)) == "retry_cloud"
        assert should_continue(self._state(plan="cloud", error="fail", retry_count=2)) == "retry_cloud"

    def test_cloud_exhausted_ends(self):
        """Verify cloud route ends after exhausting retries."""
        assert should_continue(self._state(plan="cloud", error="fail", retry_count=3)) == "end"

    def test_low_confidence_accepts_fallback(self, mocker):
        """Verify low-confidence local routes trigger cloud fallback when user accepts."""
        mocker.patch("dhi.agent.graph.Prompt.ask", return_value="y")
        state = self._state(error="fail", retry_count=2, confidence=0.3)
        result = should_continue(state)
        assert result == "fallback_cloud"

    def test_low_confidence_declines_fallback(self, mocker):
        """Verify declining cloud fallback ends the loop."""
        mocker.patch("dhi.agent.graph.Prompt.ask", return_value="n")
        state = self._state(error="fail", retry_count=2, confidence=0.3)
        result = should_continue(state)
        assert result == "end"
