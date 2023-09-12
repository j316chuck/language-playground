import gzip
import json
import subprocess
from typing import Tuple


def execute_code_in_subprocess(code: str) -> Tuple[bool, str, str]:
    # Run the script in a separate process
    try:
        result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=10)
    except: 
        return False, "", "code timed out"
    # Return the output (stdout + stderr)
    success = result.returncode == 0
    return success, result.stdout, result.stderr

# Function to stream and parse each jsonl line from the file
def stream_jsonl(filename: str):
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
