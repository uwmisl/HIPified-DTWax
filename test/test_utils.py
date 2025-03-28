import subprocess
import os
import re
import math
import time
import numpy as np
import functools
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime
from dtw import dtw


def run_test(reference, queries, segment_size=1, thorough=False, save_failures=False):
    data_file = write_data(reference, queries)
    dtwax_results, std_out = launch_DTWax(data_file, segment_size, thorough=thorough)
    python_results = launch_python_dtw(data_file, thorough)
    if thorough:
        dtwax_matrices = process_stdout(std_out, len(reference), len(queries[0]), len(queries))
        dtwax_results = list(zip(dtwax_results, dtwax_matrices))
        return compare_scores_and_matrices(dtwax_results, python_results, save_failures, reference, queries)
    return compare_scores(dtwax_results, python_results)

def process_stdout(dtwax_stdout, ref_len, query_len, num_queries):
    cost_matrices = [np.full((query_len, ref_len), np.nan)
                     for _ in range(num_queries)]
    cost_left_matrices = [np.full((query_len, ref_len), np.nan)
                          for _ in range(num_queries)]
    cost_top_matrices = [np.full((query_len, ref_len), np.nan)
                         for _ in range(num_queries)]
    cost_diag_matrices = [np.full((query_len, ref_len), np.nan)
                          for _ in range(num_queries)]
    for line in dtwax_stdout:
        match = re.match(r"\[(\d+),(\d+),(\d+)\]=([\d.]+)", line.strip())
        if match:
            x, y, z, val = map(float, match.groups())
            cost_matrices[int(x)][int(y), int(z)] = val
        match = re.match(
            r"\[(\d+),(\d+),(\d+)\]=\(([^,]+),([^,]+),([^,]+),([^,]+)\)", line.strip())
        if match:
            x, y, z, val, left, top, diag = match.groups()
            x, y, z = int(x), int(y), int(z)
            cost_matrices[x][y,z] = val
            cost_left_matrices[x][y,z] = np.inf if left == "inf" else float(left)
            cost_top_matrices[x][y,z] = np.inf if top == "inf" else float(top)
            cost_diag_matrices[x][y,z] = np.inf if diag == "inf" else float(diag)

    return list(zip(cost_matrices, cost_left_matrices, cost_top_matrices, cost_diag_matrices))


def write_data(reference, queries, file_path='data.temp'):
    # Get the location for the test datasets
    # (get the absolute path, so this works wherever this script is called from)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, file_path)
    print(f"Writing reference and query data to {file_path}")
    with open(file_path, "w") as outFile:
        outFile.write(" ".join(f"{x:.7f}" for x in reference) + "\n")
        for query in queries:
            outFile.write(" ".join(f"{x:.7f}" for x in query) + "\n")
    return file_path


def launch_DTWax(data_file, segment_size=1, query_batch_size=64, thorough=False):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    build_script = os.path.join(script_dir, "../build/build_DTWax.sh")
    build_command = [build_script, data_file,
                     "-segment_size", str(segment_size),
                     "-query_batch_size", str(query_batch_size)]
    if thorough:
        build_command.append("-debug")
        main_executable = "../src/main_debug"
    else:
        main_executable = "../src/main"
    main_executable = os.path.join(script_dir, main_executable)

    print("Building DTWax", end='\r')
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

    print("Running DTWax  ", end='\r')
    start_time = time.time()
    try:
        result = subprocess.run(
            [main_executable, data_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"DTWax ran in {(time.time() - start_time) * 1000:.2f} ms")

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


def launch_python_dtw(file_path, thorough=False):
    print("Running dtw-python", end='\r')
    start_time = time.time()
    with open(file_path, "r") as inFile:
        reference = list(map(float, inFile.readline().strip().split()))
        queries = []
        for line in inFile:
            queries.append(list(map(float, line.strip().split())))
    if thorough:
        partial_func = functools.partial(python_dtw_score_matrix, reference)
    else:
        partial_func = functools.partial(python_dtw_score, reference)
    with mp.Pool(processes=min(len(queries), mp.cpu_count())) as pool:
        python_results = pool.map(partial_func, queries)
    print(f"dtw-python ran in {(time.time() - start_time) * 1000:.2f} ms")
    return python_results

def python_dtw_score(reference, query):
    distance = dtw(reference, query, step_pattern='symmetric1',
                   dist_method='sqeuclidean', distance_only=True).distance
    return np.float32(distance)  # match the 32-bit float precision of DTWax

def python_dtw_score_matrix(reference, query):
    alignment = dtw(query, reference, step_pattern='symmetric1',
                    dist_method='sqeuclidean', keep_internals=True)
    return np.float32(alignment.distance), alignment.costMatrix

def compare_scores_and_matrices(dtwax_results, python_results, save_failures=False, ref=None, queries=None):
    # Computes the costs_left, costs_top, and costs_diag matrices
    def get_all_python_costs(python_costs):
        M, N = python_costs.shape
        inf_col = np.full((M, 1), np.inf)
        inf_row = np.full((1, N), np.inf)
        python_costs_left = np.hstack((inf_col, python_costs[:, :-1]))
        python_costs_top = np.vstack((inf_row, python_costs[:-1, :]))
        python_costs_diag = np.vstack((inf_row, python_costs[:-1, :]))
        python_costs_diag = np.hstack((inf_col, python_costs_diag[:, :-1]))
        python_costs_diag[0, 0] = 0
        return python_costs, python_costs_left, python_costs_top, python_costs_diag
    
    print("Analyzing results (scores and cost matrices)")
    python_results = [(score, get_all_python_costs(matrix))
                      for score, matrix in python_results]
    comparisons = [compare_score_and_matrix(d, p)
                   for d, p in zip(dtwax_results, python_results)]
    if save_failures:
        for q, passes, d, p in zip(queries, comparisons, dtwax_results, python_results):
            if not passes:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_test_case(ref, q, d, p, directory=timestamp)
                print(f"Saved failing test case to {timestamp}")

    print(f"{np.sum(comparisons)}/{len(comparisons)} queries were correctly processed")
    if np.all(comparisons):
        print("\033[92mTest passed.\033[0m")  # prints in green
    else:
        print("\033[91mTest failed.\033[0m")  # prints in red
    return np.all(comparisons)

def compare_scores(dtwax_scores, python_scores):
    print("Analyzing results (just the scores)")
    mismatches = [
        (dtwax_score, python_score)
        for dtwax_score, python_score in zip(dtwax_scores, python_scores)
        if not compare_score(dtwax_score, python_score)
    ]

    if len(mismatches) == 0:
        print("\033[92mTest passed.\033[0m")  # prints in green
    else:
        print("\033[91mTest failed.\033[0m")  # prints in red
        print(f"\tMismatched scores: {len(mismatches)}/{len(dtwax_scores)}")
        for dtwax_score, python_score in mismatches:
            print(f"DTWAX: {dtwax_score}, Python: {python_score}")

    return len(mismatches) == 0

def compare_score(score_a, score_b, tolerance = 1e-3):
    return math.isclose(score_a, score_b, abs_tol=tolerance)
    
def compare_matrix(matrix_a, matrix_b, tolerance=1e-3):
    # print("Matrix A:", matrix_a)
    # print("Matrix B:", matrix_b)

    mismatched_indices = [
        (i, j)
        for i, (row_a, row_b) in enumerate(zip(matrix_a, matrix_b))
        for j, (a, b) in enumerate(zip(row_a, row_b))
        if not math.isclose(a, b, abs_tol=tolerance)
    ]

    # if len(mismatched_indices) > 0:
    #     print("Mismatched indices:", mismatched_indices[:10])
    # else:
    #     print("All values match within tolerance.")

    return len(mismatched_indices) == 0

def compare_score_and_matrix(a, b, save_failures=False, ref=None, queries=None):
    score_a, matrices_a = a
    score_b, matrices_b = b
    scores_match = compare_score(score_a, score_b)
    matrices_match = np.all([compare_matrix(a, b) for a, b in zip(matrices_a, matrices_b)])
    print(f"{scores_match}, {matrices_match}")
    
    return scores_match and matrices_match


def save_test_case(reference, query, dtwax_results, python_results, directory="."):
    dtwax_score, dtwax_matrices = dtwax_results
    python_score, python_matrices = python_results
    # Save the reference and query into a data file
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(script_dir, directory))
    write_data(reference, [query], f"{directory}/data.txt")
    # os.makedirs(os.path.join(script_dir, f"{directory}"), exist_ok=True)
    # file_path = os.path.join(script_dir, f"{directory}/data.txt")
    # with open(file_path, "w") as outFile:
    #     outFile.write(" ".join(map(str, reference)) + "\n")
    #     outFile.write(" ".join(map(str, query)) + "\n")

    # Save the cost matrices
    file_path = os.path.join(script_dir, f"{directory}/matrices.p")
    with open(file_path, "wb") as f:
        pickle.dump((dtwax_matrices, python_matrices), f)
        
    # Save the figure
    title = f"python:{python_score}, dtwax={dtwax_score}"
    save_diffs_plot(dtwax_matrices, python_matrices, f"{directory}/diffs.png", title)
    
def save_diffs_plot(dtwax_matrices, python_matrices, file_path='cost_matrices_diff.png', plot_title="Matrix differences"):
    diffs = [x == y for x, y in zip(dtwax_matrices, python_matrices)]
    titles = ["Costs", "Costs left", "Costs top", "Costs diag"]
    cmap = mcolors.ListedColormap(['#D1D1D1', '#444444'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, diff, title in zip(axes.flat, diffs, titles):
        im = ax.imshow(diff, cmap=cmap, norm=norm)
        ax.xaxis.tick_top()
        ax.set_title(title)
    legend_patches = [
        mpatches.Patch(color=cmap.colors[1], label='Correct value'),
        mpatches.Patch(color=cmap.colors[0], label='Incorrect value')
    ]

    fig.suptitle(plot_title, y=0.95, fontsize=16)
    fig.legend(handles=legend_patches, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.9))
    fig.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
