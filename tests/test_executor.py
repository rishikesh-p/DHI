import pytest
import subprocess
from dhi.tools.executor import SafeExecutor

def test_executor_basic_command():
    executor = SafeExecutor()
    output = executor.execute("echo 'hello pytest'")
    assert "hello pytest" in output
    assert "Error" not in output

def test_executor_command_failure():
    executor = SafeExecutor()
    output = executor.execute("ls /nonexistent_folder_xyz123")
    assert "Error (Exit Code" in output
    assert "No such file or directory" in output

def test_executor_timeout(mocker):
    # Mock subprocess.run to raise a TimeoutExpired exception so we don't have to wait 10 seconds
    mocker.patch(
        "subprocess.run", 
        side_effect=subprocess.TimeoutExpired(cmd="sleep 15", timeout=10)
    )
    
    executor = SafeExecutor()
    output = executor.execute("sleep 15")
    
    assert "Execution timed out" in output
    assert "Error" in output
