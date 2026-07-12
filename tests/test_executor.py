import pytest
import subprocess
import os
from dhi.tools.executor import SafeExecutor


# -- Fixtures --

@pytest.fixture
def executor():
    """Provide a SafeExecutor with the project root as workdir."""
    return SafeExecutor()


# -- Basic execution --

class TestBasicExecution:
    def test_echo_returns_stdout(self, executor):
        """Verify a simple echo command succeeds and captures stdout."""
        result = executor.execute("echo 'hello pytest'")
        assert result["success"] is True
        assert "hello pytest" in result["output"]

    def test_multiline_output(self, executor):
        """Verify multi-line stdout is captured completely."""
        result = executor.execute("printf 'line1\\nline2\\nline3'")
        assert result["success"] is True
        assert "line1" in result["output"]
        assert "line3" in result["output"]

    def test_silent_success(self, executor):
        """Verify a command with no output still reports success."""
        result = executor.execute("true")
        assert result["success"] is True
        assert result["output"]  # Should have a fallback message.

    def test_exit_code_zero_with_stderr_warning(self, executor):
        """Verify exit 0 with 'Error:' in stdout is still marked success."""
        result = executor.execute("echo 'Error: non-fatal warning'")
        assert result["success"] is True


# -- Failure modes --

class TestFailureModes:
    def test_nonexistent_path(self, executor):
        """Verify a failing command reports failure with stderr."""
        result = executor.execute("ls /nonexistent_folder_xyz123")
        assert result["success"] is False
        assert "No such file or directory" in result["output"]

    def test_syntax_error(self, executor):
        """Verify a bash syntax error is caught and reported."""
        result = executor.execute("if then fi")
        assert result["success"] is False
        assert "output" in result

    def test_nonzero_exit_without_stderr(self, executor):
        """Verify silent non-zero exit codes are handled (e.g., grep no match)."""
        result = executor.execute("grep 'impossibleXYZ' /dev/null")
        assert result["success"] is False

    def test_timeout(self, mocker):
        """Verify timeout is caught without waiting for the real timeout."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="sleep 120", timeout=60)
        )
        executor = SafeExecutor()
        result = executor.execute("sleep 120")

        assert result["success"] is False
        assert "timed out" in result["output"].lower()

    def test_generic_exception(self, mocker):
        """Verify unexpected exceptions are caught gracefully."""
        mocker.patch(
            "subprocess.run",
            side_effect=OSError("Permission denied")
        )
        executor = SafeExecutor()
        result = executor.execute("echo test")

        assert result["success"] is False
        assert "Permission denied" in result["output"]


# -- Sandbox integrity --

class TestSandboxIntegrity:
    def test_filesystem_is_read_only(self, executor):
        """Verify bwrap enforces read-only root filesystem."""
        result = executor.execute("touch /usr/test_file_should_fail")
        assert result["success"] is False

    def test_home_is_tmp(self, executor):
        """Verify HOME is remapped to /tmp inside the sandbox."""
        result = executor.execute("echo $HOME")
        assert result["success"] is True
        assert "/tmp" in result["output"]


# -- Network sandboxing --

class TestNetworkSandbox:
    def test_network_blocked_by_default(self, executor):
        """Verify network is blocked when requires_network=False."""
        result = executor.execute("curl -s --max-time 2 http://example.com", requires_network=False)
        assert result["success"] is False

    def test_workdir_is_writable(self, executor):
        """Verify the project workdir is writable inside the sandbox."""
        test_file = os.path.join(executor.workdir, ".dhi_test_write")
        result = executor.execute(f"touch {test_file} && rm {test_file}")
        assert result["success"] is True
