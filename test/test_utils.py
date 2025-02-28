import subprocess
import os
import re
import math
import time
import numpy as np
import functools
import multiprocessing as mp
from dtw import dtw


def run_test(reference, queries, segment_size=1):
    data_file = write_temp_data(reference, queries)
    dtwax_scores, _ = launch_DTWax(data_file, segment_size)
    python_scores = launch_python_dtw(data_file)
    return compare_scores(dtwax_scores, python_scores)

# def process_stdout(dtwax_stdout, ref_len, query_len, num_queries):
#     dtw_matrices = [np.full((query_len, ref_len), np.nan) for _ in range(num_queries)]
#     for line in dtwax_stdout:
#         match = re.match(r"\[(\d+),(\d+),(\d+)\]=([\d.]+)", line.strip())
#         if match:
#             x, y, z, val = map(float, match.groups())
#             dtw_matrices[int(x)][int(y), int(z)] = val
#     return dtw_matrices


def process_stdout(dtwax_stdout, ref_len, query_len, num_queries):
    cost_matrices = [np.full((query_len, ref_len), np.nan) for _ in range(num_queries)]
    cost_left_matrices = [np.full((query_len, ref_len), np.nan) for _ in range(num_queries)]
    cost_top_matrices = [np.full((query_len, ref_len), np.nan) for _ in range(num_queries)]
    cost_diag_matrices = [np.full((query_len, ref_len), np.nan) for _ in range(num_queries)]
    for line in dtwax_stdout:
        match = re.match(r"\[(\d+),(\d+),(\d+)\]=([\d.]+)", line.strip())
        if match:
            x, y, z, val = map(float, match.groups())
            cost_matrices[int(x)][int(y), int(z)] = val
        match = re.match(r"\[(\d+),(\d+),(\d+)\]=\(([^,]+),([^,]+),([^,]+)\)", line.strip())
        if match:
            x, y, z, left, top, diag = match.groups()
            cost_left_matrices[int(x)][int(y), int(z)] = np.inf if left == "inf" else float(left)
            cost_top_matrices[int(x)][int(y), int(z)] = np.inf if top == "inf" else float(top)
            cost_diag_matrices[int(x)][int(y), int(z)] = np.inf if diag == "inf" else float(diag)

    return zip(cost_matrices, cost_left_matrices, cost_top_matrices, cost_diag_matrices)


def write_temp_data(reference, queries):
    # Get the location for the test datasets
    # (get the absolute path, so this works wherever this script is called from)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'data.temp')
    print("Writing test data to file")
    with open(file_path, "w") as outFile:
        outFile.write(" ".join(f"{x:.7f}" for x in reference) + "\n")
        for query in queries:
            outFile.write(" ".join(f"{x:.7f}" for x in query) + "\n")
    return file_path


def launch_DTWax(data_file, segment_size=1, query_batch_size=64, debug=False):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    build_script = os.path.join(script_dir, "../build/build_DTWax.sh")
    build_command = [build_script, data_file,
                     "-segment_size", str(segment_size),
                     "-query_batch_size", str(query_batch_size)]
    if debug:
        build_command.append("-debug")
        main_executable = "../src/main_debug"
    else:
        main_executable = "../src/main"
    main_executable = os.path.join(script_dir, main_executable)

    print("Building DTWax")
    try:
        subprocess.run(
            build_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed for {data_file}.")
        print("stderr:\n", e.stderr.strip())
        return

    print("Running DTWax")
    try:
        result = subprocess.run(
            [main_executable, data_file],
            capture_output=True,
            text=True,
            check=True
        )

        # Process and print lines after "Results:"
        dtwax_stdout = result.stdout.strip().split("\n")
        found_results = False
        scores = []
        for line in dtwax_stdout:
            if "Results:" in line:
                found_results = True
                continue
            if found_results and re.match(r"^\d+\s+\d+\s+\d+\s+\d+", line.strip()):
                scores.append(float(line.strip().split()[-1]))
        return scores, dtwax_stdout
    except subprocess.CalledProcessError as e:
        print("Execution failed.")
        print("stderr:\n", e.stderr.strip())


def launch_python_dtw(file_path):
    print("Running dtw-python")
    with open(file_path, "r") as inFile:
        reference = list(map(float, inFile.readline().strip().split()))
        queries = []
        for line in inFile:
            queries.append(list(map(float, line.strip().split())))
            
    partial_func = functools.partial(python_dtw_score, reference)
    with mp.Pool(processes=min(len(queries), mp.cpu_count())) as pool:
        python_results = pool.map(partial_func, queries)
    return python_results


def python_dtw_score(reference, query):
    distance = dtw(reference, query, step_pattern='symmetric1',
                   dist_method='sqeuclidean', distance_only=True).distance
    return np.float32(distance)  # match the 32-bit float precision of DTWax


def python_dtw_score_debug(reference, query):
    alignment = dtw(query, reference, step_pattern='symmetric1',
                    dist_method='sqeuclidean', keep_internals=True)
    return np.float32(alignment.distance), alignment.costMatrix


def compare_scores(dtwax_scores, python_scores):
    print("Analyzing results" + " "*30, end="\n")

    tolerance = 1e-1
    mismatches = sum(
        not math.isclose(dtwax_score, python_score, abs_tol=tolerance)
        for dtwax_score, python_score in zip(dtwax_scores, python_scores)
    )

    if mismatches == 0:
        print("\033[92mTest passed.\033[0m")  # prints in green
        # print(f"\tdtWax_scores = [{', '.join(map(str, dtwax_scores))}]")
        # print(f"\tpython_scores = [{', '.join(map(str, python_scores))}]")
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
        print(f"\tMismatched scores: {len(mismatches)}/{len(dtwax_scores)}")
        print(f"\tFirst mismatch at index: {mismatches[0]}")
        print(f"\tDTWax scores: [{', '.join(map(str, dtwax_scores))}]")
        print(f"\tPython scores: [{', '.join(map(str, python_scores))}]")

        return False, mismatches[0]
