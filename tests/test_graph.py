import pytest
from dhi.agent.graph import parse_llm_output

def test_parse_llm_output_basic_bash():
    response = "Here is the command you requested:\n```bash\necho 'hello'\n```\nDone."
    assert parse_llm_output(response) == "echo 'hello'"

def test_parse_llm_output_sh_block():
    response = "```sh\npwd\n```"
    assert parse_llm_output(response) == "pwd"

def test_parse_llm_output_no_backticks():
    response = "ls -la"
    assert parse_llm_output(response) == "ls -la"

def test_parse_llm_output_multiple_blocks():
    # It should grab the first block
    response = "```bash\ncd /tmp\n```\nand then\n```bash\nrm -rf *\n```"
    assert parse_llm_output(response) == "cd /tmp"

def test_parse_llm_output_empty():
    assert parse_llm_output("") is None
    assert parse_llm_output("   \n  ") is None
