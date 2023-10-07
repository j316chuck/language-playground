import sys
import os 
from typing import List
sys.path.append("../")
from utils import stream_jsonl, execute_code_in_subprocess

def print_problem(problem_id: int, output_dir="./output/hard_9/"):
    problems = list(stream_jsonl('./data/HumanEval.jsonl.gz'))
    output = f"#problem_id: {problem_id}\n"
    output += problems[problem_id]['prompt'] + '\n'
    output += problems[problem_id]['canonical_solution'] + '\n'
    output += problems[problem_id]['test'] + '\n'
    output += f"check({problems[problem_id]['entry_point']})"
    print(output)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/test_{problem_id}.py", 'w') as f:
        f.write(output)


def print_problems(problem_ids: List[int]):
    for p_id in problem_ids:
        print("*"*80)
        print_problem(p_id)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_ids', type=int, nargs='+')
    args = parser.parse_args()
    print_problems(args.problem_ids)
