import subprocess
import os
import re
import math
import numpy as np
from dtw import dtw

def run_test(reference, queries, segment_size=1):
    file_path = write_temp_data(reference, queries)
    dtwax_scores = launch_DTWax(file_path, segment_size)
    python_scores = launch_python_dtw(file_path)
    return compare_scores(dtwax_scores, python_scores)

def write_temp_data(reference, queries):
    # Get the location for the test datasets
    # (get the absolute path, so this works wherever this script is called from)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'data.temp')
    print("Writing test data to file" + " "*30, end='\r')
    with open(file_path, "w") as outFile:
        outFile.write(" ".join(f"{x:.7f}" for x in reference) + "\n")
        for query in queries:
            outFile.write(" ".join(f"{x:.7f}" for x in query) + "\n")
    return file_path

def launch_DTWax(data_file, segment_size=1):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    build_script = os.path.join(script_dir, "../build/build_DTWax.sh")
    main_executable = os.path.join(script_dir, "../src/main_debug")

    # Build command
    print("Building DTWax" + " "*30, end='\r')
    try:
        subprocess.run(
            [build_script, data_file, str(segment_size)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed for {data_file}.")
        print("stderr:\n", e.stderr.strip())
        return

    # Run the main_debug executable
    print("Running DTWax" + " "*30, end='\r')
    try:
        result = subprocess.run(
            [main_executable, data_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Process and print lines after "Results:"
        result_lines = result.stdout.strip().split("\n")
        found_results = False
        scores = []
        for line in result_lines:
            if "Results:" in line:
                found_results = True
                continue
            if found_results and re.match(r"^\d+\s+\d+\s+\d+\s+\d+", line.strip()):
                scores.append(float(line.strip().split()[-1]))
        return scores
    except subprocess.CalledProcessError as e:
        print("Execution failed.")
        print("stderr:\n", e.stderr.strip())

def launch_python_dtw(file_path):
    print("Running dtw-python" + " "*30, end='\r')
    with open(file_path, "r") as inFile:
        reference = list(map(float, inFile.readline().strip().split()))
        queries = []
        for line in inFile:
            queries.append(list(map(float, line.strip().split())))
    python_scores = [python_dtw_score(reference, query) for query in queries]
    return python_scores

def python_dtw_score(reference, query):
    score = dtw(reference, query, step_pattern='symmetric1', dist_method='sqeuclidean', distance_only=True).distance
    return np.float32(score) # match the 32-bit float precision of DTWax

def compare_scores(dtwax_scores, python_scores):
    print("Analyzing results" + " "*30, end="\n")

    tolerance = 1e-1
    mismatches = sum(
        not math.isclose(dtwax_score, python_score, abs_tol=tolerance)
        for dtwax_score, python_score in zip(dtwax_scores, python_scores)
    )

    if mismatches == 0:
        print("\033[92mTest passed.\033[0m")  # prints in green
    else:
        print("\033[91mTest failed.\033[0m")  # prints in red
        print(f"\tMismatched scores: {mismatches}/{len(dtwax_scores)}")
        print(f"\tdtWax_scores = [{', '.join(map(str, dtwax_scores))}]")
        print(f"\tpython_scores = [{', '.join(map(str, python_scores))}]")

    return mismatches == 0

def compare_scores_get_mismatch(dtwax_scores, python_scores):
    print("Analyzing results" + " "*30, end="\n")

    tolerance = 1e-1
    mismatches = [
        i for i, (dtwax_score, python_score) in enumerate(zip(dtwax_scores, python_scores))
        if not math.isclose(dtwax_score, python_score, abs_tol=tolerance)
    ]

    if not mismatches:
        print("\033[92mTest passed.\033[0m")  # prints in green
        return True, None
    else:
        print("\033[91mTest failed.\033[0m")  # prints in red
        print(f"\tMismatched scores: {len(mismatches)}")
        print(f"\tFirst mismatch at index: {mismatches[0]}")
        print(f"\tDTWax scores: [{', '.join(map(str, dtwax_scores))}]")
        print(f"\tPython scores: [{', '.join(map(str, python_scores))}]")

        return False, mismatches[0]

