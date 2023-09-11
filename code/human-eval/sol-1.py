import requests
import json
import gzip
import os

# Define the path to the HumanEval.jsonl.gz file
ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "data", "HumanEval.jsonl.gz")

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

# Load the problems from the file
problems = list(stream_jsonl(HUMAN_EVAL))

# Define the function to call the OpenAI ChatCompletions API
def solve_problem_with_openai(problem):
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. \
                Please solve the following coding challenge. Make sure to only output the completion of the code and not any other text/function signatures."},
            {"role": "user", "content": problem}
        ]
    }
    response = requests.post(endpoint, headers=headers, json=data)
    full_response = response.json()["choices"][0]["message"]["content"]
    import pdb; pdb.set_trace()
    
    # Extract code between the markers
    # code_start = full_response.find("---BEGIN CODE---") + len("---BEGIN CODE---")
    # code_end = full_response.find("---END CODE---")
    # extracted_code = full_response[code_start:code_end].strip()
    return full_response

import subprocess

def execute_code_with_doctest(code: str) -> str:
    # Create a full Python script with the code and doctest
    script = f"""
{code}

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    """

    # Run the script in a separate process
    result = subprocess.run(["python3", "-c", script], capture_output=True, text=True)

    # Return the output (stdout + stderr)
    return result.stdout + result.stderr

# Solve problems using the OpenAI API
problems = [problems[0]]
solutions = {}
for problem in problems:
    problem_statement = problem["prompt"]
    solution = solve_problem_with_openai(problem_statement)
    solution += "\n\n---BEGIN CODE---\n\n"
    solutions[problem_statement] = solution
    test_results = execute_code_with_doctest(solution)
    print(test_results)

print(solutions)

