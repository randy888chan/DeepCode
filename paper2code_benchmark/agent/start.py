import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

# 首先加载环境变量配置
from env_config import load_env_config, substitute_path_variables
load_env_config()


from _basic_agent_iterative import basic_agent_iterative  # type: ignore
from _basic_agent_plus import basic_agent_plus  # type: ignore
from _execute import bash, python
from _file_reader import read_file_chunk, search_file  # type: ignore
from inspect_ai import Task, eval, task  # type: ignore
from inspect_ai.dataset import Sample  # type: ignore
from inspect_ai.tool import web_browser  # type: ignore
from templates import additional_notes_template
from utils import get_gpu_generation
# PB constants
CODE_DIR = Path(os.environ["CODE_DIR"])
AGENT_DIR = Path(os.environ["AGENT_DIR"])
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE")
assert WORKSPACE_BASE is not None, "Environment variable 'WORKSPACE_BASE' is not set."

# 环境变量已通过 env_config.py 加载，直接获取即可
MAX_TIME_IN_HOURS = os.environ.get("MAX_TIME_IN_HOURS")
MODEL = os.environ.get("MODEL")
assert MODEL is not None, "Environment variable 'MODEL' is not set."
DISALLOW_SUBMIT = os.environ.get("DISALLOW_SUBMIT", "False").lower() == "true"
ITERATIVE_AGENT = os.environ.get("ITERATIVE_AGENT", "False").lower() == "true"
gpu_generation = get_gpu_generation()
type_of_processor = gpu_generation if gpu_generation else "CPU"

additional_notes = additional_notes_template.substitute(
    type_of_processor=type_of_processor,
    max_time_in_hours=MAX_TIME_IN_HOURS,
    workspace=CODE_DIR,
    workspace_base=WORKSPACE_BASE,
)
with open(f"{WORKSPACE_BASE}/code_only_instructions.txt", "r") as file:
    partial_instructions = file.read()

# 替换指令中的路径变量
partial_instructions = substitute_path_variables(partial_instructions)
instructions = partial_instructions + additional_notes


@task
def pb_task():
    if ITERATIVE_AGENT:
        solver = basic_agent_iterative(
            tools=[bash(), read_file_chunk()],
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT,
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
        )
    else:
        solver = basic_agent_plus(
            tools=[bash(), python(), read_file_chunk(), search_file()] + web_browser(),
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT,
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
        )
    return Task(
        dataset=[Sample(input=instructions)],
        solver=solver,
        sandbox="local",
    )


def run():
    eval(pb_task(), model=MODEL, trace=True, reasoning_effort="high", reasoning_tokens=25600)


if __name__ == "__main__":
    run()
