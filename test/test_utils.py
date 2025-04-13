import subprocess
import os
import re
import math
import time
import numpy as np
import functools
import struct
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime
from dtw import dtw

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXE_DIR = os.path.join(SCRIPT_DIR, "dtwax_executables")

def run_test(reference, queries, segment_size=1, thorough=False):
    ref_data_file, queries_data_file = write_data(reference, queries)
    dtwax_results, std_out = launch_DTWax(ref_data_file, queries_data_file, segment_size, thorough=thorough)
    python_results = launch_python_dtw(ref_data_file, queries_data_file, thorough)
    if thorough:
        dtwax_matrices = process_stdout(std_out, len(reference), len(queries[0]), len(queries))
        dtwax_results = list(zip(dtwax_results, dtwax_matrices))
        return compare_scores_and_matrices(dtwax_results, python_results)
    return compare_scores(dtwax_results, python_results)

def read_debug_matrix(file_name, x, y, num_queries):
    with open(file_name, "rb") as f:
        data = f.read()

        # Each `float4` consists of 4 float values, so 4 * 4 bytes = 16 bytes per `float4`
        num_elements = len(data) // 16  # Total number of `float4` elements in the file
        print(f"len(data)={len(data)}")
        print(f"num_elements={num_elements}")

        float4_list = []
        for i in range(num_elements):
            # Unpack the 16 bytes into 4 float values
            unpacked = struct.unpack('4f', data[i*16:(i+1)*16])  # '4f' means 4 floats
            float4_list.append(unpacked)

        # Convert to a numpy array and reshape to (num_reads, y, x, 4)
        float4_array = np.array(float4_list, dtype=np.float32).reshape(num_queries, y, x, 4)

        return float4_array

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
        if line.startswith("Writing debug data to"):
            filename = line.split()[-1]
            matrices = read_debug_matrix(filename, ref_len, query_len, num_queries)
            os.remove(filename)
            return matrices
        # match = re.match(r"\[(\d+),(\d+),(\d+)\]=([\d.]+)", line.strip())
        # if match:
        #     x, y, z, val = map(float, match.groups())
        #     cost_matrices[int(x)][int(y), int(z)] = val
        # match = re.match(
        #     r"\[(\d+),(\d+),(\d+)\]=\(([^,]+),([^,]+),([^,]+),([^,]+)\)", line.strip())
        # if match:
        #     x, y, z, val, left, top, diag = match.groups()
        #     x, y, z = int(x), int(y), int(z)
        #     cost_matrices[x][y,z] = val
        #     cost_left_matrices[x][y,z] = np.inf if left == "inf" else float(left)
        #     cost_top_matrices[x][y,z] = np.inf if top == "inf" else float(top)
        #     cost_diag_matrices[x][y,z] = np.inf if diag == "inf" else float(diag)

    # return list(zip(cost_matrices, cost_left_matrices, cost_top_matrices, cost_diag_matrices))
    return False


def write_reference_data(reference, file_path):
    with open(file_path, "w") as file:
        file.write(" ".join(f"{x:.5f}" for x in reference) + "\n")
    return file_path

def write_query_data(queries, file_path):
    with open(file_path, "w") as f:
        for query in queries:
            f.write(" ".join(f"{x:.5f}" for x in query) + "\n")
    return file_path

def write_data(reference, queries):
    ref_data_file = os.path.join(SCRIPT_DIR, "temp_ref.data")
    write_reference_data(reference, ref_data_file)
    print(f"Wrote reference data to {ref_data_file}")
    query_data_file = os.path.join(SCRIPT_DIR, "temp_query.data")
    write_query_data(queries, query_data_file)
    print(f"Wrote query data to {query_data_file}")
    return ref_data_file, query_data_file

def launch_DTWax(ref_data_file, queries_data_file, ref_len, query_len, num_queries, segment_size=1, query_batch_size=64, thorough=False):
    build_script = os.path.join(SCRIPT_DIR, "../build/build_DTWax_v2.sh")
    build_command = [build_script, 
                     "-ref_len", str(ref_len),
                     "-query_len", str(query_len), 
                     "-num_reads", str(num_queries),
                     "-segment_size", str(segment_size),
                     "-query_batch_size", str(query_batch_size)]
    if thorough:
        build_command.append("-debug")
        main_executable = "../src/main_debug"
    else:
        main_executable = "../src/main"
    main_executable = os.path.join(SCRIPT_DIR, main_executable)

    print("Building DTWax", end='\r')
    start_time = time.time()
    try:
        subprocess.run(
            build_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"Building DTWax failed.")
        print("stderr:\n", e.stderr.strip())
        return
    print(f"DTWax built in {(time.time() - start_time) * 1000:.2f} ms")
    
    print("Running DTWax  ")
    start_time = time.time()
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            [main_executable, ref_data_file, queries_data_file],
            capture_output=True,
            text=True,
            check=True
    )
    except subprocess.CalledProcessError as e:
        print(f"Running DTWax failed.")
        print("stderr:\n", e.stderr.strip())
        return
    
    print(f"DTWax ran in {(time.perf_counter() - start_time) * 1000:.2f} ms")

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
    # except subprocess.CalledProcessError as e:
    #     print("DTWax execution failed.")


def launch_python_dtw(ref_data_file, queries_data_file, thorough=False):
    print("Running dtw-python", end='\r')
    start_time = time.time()
    with open(ref_data_file, "r") as file:
        reference = list(map(float, file.readline().strip().split()))
    with open(queries_data_file, "r") as file:
        queries = []
        for line in file:
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

def compare_scores_and_matrices(dtwax_results, python_results):
    # Computes the costs_left, costs_top, and costs_diag matrices
    print("#dtwax_results##########")
    print(len(dtwax_results))
    print("#dtwax_results[0]##########")
    print(len(dtwax_results[0]))
    print("#dtwax_results[0][0]##########")
    print(dtwax_results[0][0])
    print("#dtwax_results[0][1]##########")
    print(dtwax_results[0][1].shape)
    print("############")
    def get_all_python_costs(python_costs):
        M, N = python_costs.shape
        inf_col = np.full((M, 1), np.inf)
        inf_row = np.full((1, N), np.inf)
        python_costs_left = np.hstack((inf_col, python_costs[:, :-1]))
        python_costs_top = np.vstack((inf_row, python_costs[:-1, :]))
        python_costs_diag = np.vstack((inf_row, python_costs[:-1, :]))
        python_costs_diag = np.hstack((inf_col, python_costs_diag[:, :-1]))
        python_costs_diag[0, 0] = 0
        return np.stack((python_costs, python_costs_left, python_costs_top, python_costs_diag), axis=-1)
    
    print("Analyzing results (scores and cost matrices)")
    python_results = [(score, get_all_python_costs(matrix))
                      for score, matrix in python_results]
    comparisons = [compare_score_and_matrix(d, p)
                   for d, p in zip(dtwax_results, python_results)]
    # if save_failures:
    #     for q, passes, d, p in zip(queries, comparisons, dtwax_results, python_results):
    #         if not passes:
    #             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #             save_test_case(ref, q, d, p, directory=timestamp)
    #             print(f"Saved failing test case to {timestamp}")

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
    mismatched_indices = [
        (i, j)
        for i, (row_a, row_b) in enumerate(zip(matrix_a, matrix_b))
        for j, (a, b) in enumerate(zip(row_a, row_b))
        if not compare_score(a, b, tolerance)
    ]

    if len(mismatched_indices) > 0:
        print("Mismatched indices:", mismatched_indices[:10])
    # else:
    #     print("All values match within tolerance.")
    x, y = len(matrix_a), len(matrix_a[0])
    print(f"Found {len(mismatched_indices)}/{x*y} ({len(mismatched_indices) / (x*y):.2%}) elements didn't match")
    return len(mismatched_indices) == 0

def compare_score_and_matrix(a, b):
    score_a, matrices_a = a
    score_b, matrices_b = b
    print("$$$")
    print(matrices_a.shape)
    print(len(matrices_b))
    print("$$$")
    scores_match = compare_score(score_a, score_b)
    print(f"score {score_a} and score {score_b} match: {scores_match}")
    costs_match = compare_matrix(matrices_a[:,:,0], matrices_b[:,:,0])
    costs_left_match = compare_matrix(matrices_a[:,:,1], matrices_b[:,:,1])
    costs_top_match = compare_matrix(matrices_a[:,:,2], matrices_b[:,:,2])
    costs_diag_match = compare_matrix(matrices_a[:,:,3], matrices_b[:,:,3])
    # matrices_match = np.all([compare_matrix(a, b) for a, b in zip(matrices_a, matrices_b)])
    matrices_match = costs_match and costs_left_match and costs_top_match and costs_diag_match
    print(f"matrices match: {matrices_match}")
    
    # for i in range(4):
    #     print(matrices_a[:16, :16, i])
    
    return scores_match and matrices_match


# def save_test_case(reference, query, dtwax_results, python_results, directory="."):
#     dtwax_score, dtwax_matrices = dtwax_results
#     python_score, python_matrices = python_results
#     # Save the reference and query into a data file
#     os.makedirs(os.path.join(SCRIPT_DIR, directory))
#     write_data(reference, [query], f"{directory}/data.txt")

#     # Save the cost matrices
#     file_path = os.path.join(SCRIPT_DIR, f"{directory}/matrices.p")
#     with open(file_path, "wb") as f:
#         pickle.dump((dtwax_matrices, python_matrices), f)
        
#     # Save the figure
#     title = f"python:{python_score}, dtwax={dtwax_score}"
#     save_diffs_plot(dtwax_matrices, python_matrices, f"{directory}/diffs.png", title)
    
# def save_diffs_plot(dtwax_matrices, python_matrices, file_path='cost_matrices_diff.png', plot_title="Matrix differences"):
#     diffs = [x == y for x, y in zip(dtwax_matrices, python_matrices)]
#     titles = ["Costs", "Costs left", "Costs top", "Costs diag"]
#     cmap = mcolors.ListedColormap(['#D1D1D1', '#444444'])
#     norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     for ax, diff, title in zip(axes.flat, diffs, titles):
#         im = ax.imshow(diff, cmap=cmap, norm=norm)
#         ax.xaxis.tick_top()
#         ax.set_title(title)
#     legend_patches = [
#         mpatches.Patch(color=cmap.colors[1], label='Correct value'),
#         mpatches.Patch(color=cmap.colors[0], label='Incorrect value')
#     ]

#     fig.suptitle(plot_title, y=0.95, fontsize=16)
#     fig.legend(handles=legend_patches, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.9))
#     fig.tight_layout()
#     plt.savefig(file_path, dpi=150, bbox_inches='tight')
#     plt.close()
