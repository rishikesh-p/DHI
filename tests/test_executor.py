import pytest
import subprocess
from dhi.tools.executor import SafeExecutor

def test_executor_basic_command():
    executor = SafeExecutor()
    result = executor.execute("echo 'hello pytest'")
    assert result["success"] is True
    assert "hello pytest" in result["output"]

def test_executor_command_failure():
    executor = SafeExecutor()
    result = executor.execute("ls /nonexistent_folder_xyz123")
    assert result["success"] is False
    assert "No such file or directory" in result["output"]

def test_executor_timeout(mocker):
    # Mock subprocess.run to raise a TimeoutExpired exception so we don't have to wait 10 seconds
    mocker.patch(
        "subprocess.run", 
        side_effect=subprocess.TimeoutExpired(cmd="sleep 15", timeout=10)
    )
    
    executor = SafeExecutor()
    result = executor.execute("sleep 15")
    
    assert result["success"] is False
    assert "Execution timed out" in result["output"]
