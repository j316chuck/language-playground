from typing import Tuple 
import os 
import requests 
from utils import execute_code_in_subprocess

SYSTEM_PROMPT = "You are a helpful assistant. Please solve the following coding challenge and wrap your code in ```python ``` blocks."

def canonical_solution(problem) -> str:
    return problem['canonical_solution']

def extract_response(problem, response) -> str: 
    try:
        full_response = response.json()["choices"][0]["message"]["content"]
    except: 
        return "error in extracting response"

    try: 
        return full_response.split("```python")[1].split("```")[0]
    except IndexError:
        print("warning: code for problem {} not wrapped in python blocks".format(problem['task_id']))
        return full_response

def openai_with_solution(problem) -> str:
    solution = problem['canonical_solution']

    problem = str(problem)
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
    }
    response = requests.post(endpoint, headers=headers, json=data)  
    # extract code using ```python CODEBLOCK ```
    return extract_response(problem, response)

def openai_without_solution(problem) -> str:
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem['prompt']}
        ]
    }
    response = requests.post(endpoint, headers=headers, json=data)  
    return extract_response(problem, response)

def openai_with_feedback(problem, total_trials=3, pass_in_sol=False) -> Tuple[str, dict]:
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    if pass_in_sol: 
        content = str(problem)
    else:
        content = problem['prompt']
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    }
    response = requests.post(endpoint, headers=headers, json=data)  
    full_response = extract_response(problem, response)
    solutions = {}

    CHECKER_BEGIN_CODE = 'def check(candidate)'
    checker = CHECKER_BEGIN_CODE + problem['test'].split(CHECKER_BEGIN_CODE)[-1]
    
    num_trials = 0 
    while num_trials < total_trials:
        code =  full_response + '\n' + checker + f'\ncheck({problem["entry_point"]})'
        success, stdout, stderr = execute_code_in_subprocess(code)
        solutions[num_trials] = {}
        solutions[num_trials]['code'] = code
        solutions[num_trials]['stdout'] = stdout
        solutions[num_trials]['stderr'] = stderr    

        if success:
            return full_response, solutions
        
        print("failed {}/{} trials".format(num_trials + 1, total_trials))
        try:
            data['messages'].append({"role": "assistant", "content" : response.json()["choices"][0]["message"]["content"]})
        except Exception as e:
            data['messages'].append({"role": "assistant", "content" : full_response})
        data['messages'].append({"role": "user", "content": f"Your solution is incorrect. Here are the error logs from executing the extracted code: stdout {stdout} and stderr {stderr}. Please take a deep breath, think about what went wrong, and try again. Make sure to mark your answer in ```python``` codeblocks still."})
        response = requests.post(endpoint, headers=headers, json=data)
        full_response = extract_response(problem, response)
        num_trials += 1

    return full_response, solutions
