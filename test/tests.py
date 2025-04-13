import test_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from multiprocessing import Pool
import functools
import re
import pickle
import random

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _run_test(ref_len, query_len, num_queries=1, segment_size=1, thorough=False):
    seed = np.random.randint(10**9)  # Generate a random seed
    np.random.seed(seed)
    reference = np.random.rand(ref_len).astype(np.float32)
    queries = np.random.rand(num_queries, query_len).astype(np.float32)

    test_passes = test_utils.run_test(
        reference, queries, segment_size, thorough)
    print("Passing params:" if test_passes else "Failing params:")
    print(
        f"seed, ref_len, query_len, num_queries, segment_size = {seed}, {ref_len}, {query_len}, {num_queries}, {segment_size}")
    return test_passes

# def _run_test_thorough(ref_len, query_len, num_queries=1, segment_size=1):
#     print(f"ref length={ref_len}, query len={query_len}, num_queries={num_queries}, segment size={segment_size}")
#     reference = np.random.rand(ref_len).astype(np.float32)
#     queries = np.random.rand(num_queries, query_len).astype(np.float32)
#     return test_utils.run_test(reference, queries, segment_size)

# def _save_test_case(reference, query, dtwax_results, python_results, directory="."):
#     dtwax_score, dtwax_matrices = dtwax_results
#     python_score, python_matrices = python_results
#     # Save the reference and query into a data file
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     os.makedirs(os.path.join(script_dir, f"{directory}"), exist_ok=True)
#     file_path = os.path.join(script_dir, f"{directory}/data.txt")
#     with open(file_path, "w") as outFile:
#         outFile.write(" ".join(map(str, reference)) + "\n")
#         outFile.write(" ".join(map(str, query)) + "\n")

#     # Save the figure
#     file_path = os.path.join(script_dir, f"{directory}/diffs.png")
#     title = f"python:{python_score}, dtwax={dtwax_score}"
#     _save_diffs_plot(dtwax_matrices, python_matrices, file_path, title)

#     # Save the cost matrices
#     file_path = os.path.join(script_dir, f"{directory}/matrices.p")
#     with open(file_path, "wb") as f:
#         pickle.dump((dtwax_matrices, python_matrices), f)


def _sweep(ref_batch_counts, query_batch_counts, segment_sizes, depth, thorough=False):
    all_pass = True
    for ref_batch_count in ref_batch_counts:
        for query_batch_count in query_batch_counts:
            for segment_size in segment_sizes:
                ref_len = 64*segment_size*ref_batch_count
                query_len = 64*query_batch_count
                test_passes = _run_test(
                    ref_len, query_len, depth, segment_size, thorough)
                # ensure test is run before aggregating booleans, o/w it'll be skipped when all_pass is already false
                all_pass = all_pass and test_passes
    return all_pass


def rlarge_qlarge():
    # numbers chosen to be similar to protein squiggle/template lens, rounded up to a multiple of 64
    all_pass = True
    # for segment_size in [1,2,4,10,20]:
    for segment_size in [1]:
        # test_passes = _run_test(64*600, 64*27, num_queries=1, segment_size=segment_size, thorough=False)
        test_passes = _run_test(2560, 320, num_queries=20,
                                segment_size=8, thorough=True, save_failures=True)
        all_pass = all_pass and test_passes
    return all_pass


def fast_sweep():
    ref_batch_counts = [1, 2, 5]
    query_batch_counts = [1, 2, 5]
    segment_sizes = [1, 2, 3, 4, 8]
    depth = 100
    return _sweep(ref_batch_counts, query_batch_counts, segment_sizes, depth, thorough=False)


def thorough_sweep():
    ref_batch_counts = [1, 2, 5]
    query_batch_counts = [1, 2, 5]
    segment_sizes = [1, 2, 3, 4, 8]
    depth = 10
    # ref_batch_counts = [1]
    # query_batch_counts = [1]
    # segment_sizes = [2]
    # depth = 1
    return _sweep(ref_batch_counts, query_batch_counts, segment_sizes, depth, thorough=True)


def protein_lookup_test():
    from aminoscribe.aminoscribe import generate_squiggle
    import multiprocessing as mp
    import time
    from scipy.stats import spearmanr
    def generate_reference(protein_id):
        nterm = "SDDGDGGSSDSSGSGSSESGSESSGSGSSDSSGG"
        # the cterm sequence has been designed to aid with normalization
        cterm = "YYYYYSTSSDGDEEDGDDSTSYYYYYSTSSDGEDDEGDDSTSYYYYYSTSSDGEDEDGDDSTSYYYYYSTSSDGDEDEGDDSTSYYYYYSTSSDGDEEDGDDSTSYYYYYSTSSDGDEEDGDDSTS"
        # generate template once and then re-use
        squiggle = generate_squiggle(protein_id=protein_id, seed=42,
                                    cterm=cterm, nterm=nterm,
                                    filter_noise=True,
                                    normalize=True, norm_cutoff=4500,
                                    downsample=True, downsample_factor=10)

        # Pad up to nearest multiple of pad_multiple
        pad_multiple = 64*8
        padded_length = ((len(squiggle) + pad_multiple-1) //
                        pad_multiple) * pad_multiple

        return np.pad(squiggle, (0, padded_length - len(squiggle)), constant_values=5)


    squiggle = generate_reference("E2RYF6")
    for bucket in range(1):
        # Load the templates
        file_path = os.path.join(SCRIPT_DIR, "protein_lookup_test_data", f"bucket_{bucket}.pickle")
        with open(file_path, "rb") as inFile:
            _, templates = pickle.load(inFile)
            templates = [list(x[2]) for x in templates]
        print(f"Loaded {len(templates)} templates of length {len(templates[0])}")
        
        # Run a dtwax based lookup test
        ref_data_file = os.path.join(SCRIPT_DIR, "temp_ref.data")
        squiggle_path = test_utils.write_reference_data(squiggle, ref_data_file)
        templates_path = os.path.join(SCRIPT_DIR, "protein_lookup_test_data", f"queries_binary_{bucket}.txt")
        dtwax_scores, stdout = test_utils.launch_DTWax(squiggle_path, templates_path, 
                                                  len(squiggle), len(templates[0]), len(templates),
                                                  segment_size=8, query_batch_size=len(templates[0]))
        
        print(stdout)
        # Run a python based protein lookup test
        partial_func = functools.partial(test_utils.python_dtw_score, squiggle)
        # print(f"Found {mp.cpu_count()} cpus to use")
        # print("Running dtw-python", end='\r')
        start_time = time.time()
        with mp.Pool(processes=min(len(templates), mp.cpu_count())) as pool:
            python_scores = pool.map(partial_func, templates)
        # print(f"Collected {len(python_scores)} results")
        print(f"Parallel dtw-python ran in {(time.time() - start_time) * 1000 :.2f} ms")
        
        # Test the rankings
        correlation, _ = spearmanr(python_scores, dtwax_scores)
        print(f"Spearman correlation: {correlation:.2f}")
    


    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(script_dir, 'protein_id_test.txt')
    # dtwax_scores, _ = test_utils.launch_DTWax(file_path)
    # python_scores = test_utils.launch_python_dtw(file_path)
    # return test_utils.compare_scores(dtwax_scores, python_scores)

# This DOES NOT work if you have more queries than BLOCK_NUM
# The cost matrices will not be reconstructed from the print statements accurately
def debug_a_test():
    # params to check (make changes here)
    ref_batches = 5
    query_batches = 5
    segment_size = 1
    num_queries = 100
    query_batch_size = 64

    # derived values
    ref_len = 64 * segment_size * ref_batches
    query_len = query_batches * query_batch_size
    reference = np.random.randint(0, 2, ref_len, dtype=np.int32)
    queries = np.random.randint(0, 2, (num_queries, query_len), dtype=np.int32)
    file_path = test_utils.write_data(reference, queries)
    dtwax_scores, std_out = test_utils.launch_DTWax(
        file_path, segment_size, query_batch_size, thorough = True)
    dtwax_matrices = test_utils.process_stdout(
        std_out, len(reference), len(queries[0]), len(queries))
    python_dtw = functools.partial(
        test_utils.python_dtw_score_matrix, reference)
    with Pool() as pool:
        python_results = pool.map(python_dtw, queries)

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

    def compare_results(x, y):
        score_x, matrices_x = x
        score_y, matrices_y = y
        return (score_x == score_y) and np.all([np.array_equal(a, b) for a, b in zip(matrices_x, matrices_y)])

    dtwax_results = list(zip(dtwax_scores, dtwax_matrices))
    python_results = [(score, get_all_python_costs(matrix))
                      for score, matrix in python_results]
    for i in range(len(queries)):
        dtwax_score, dtwax_matrix = dtwax_results[i]
        python_score, python_matrix = python_results[i]

        diff_count = np.sum(dtwax_matrix[0] != python_matrix[0])
        threshold = dtwax_matrix[0].size / 1000
        if dtwax_score != python_score or diff_count > threshold:
            _save_test_case(
                reference, queries[i], dtwax_results[i], python_results[i], directory=f"query{i}")
        # for j in range(4):
        #     mismatches = np.where(dtwax_results[i][1][j] != python_results[i][1][j])
        #     mismatch_indices = [(int(x), int(y)) for x, y in zip(*mismatches)]
        #     print("Mismatched indices:", mismatch_indices[:10])
    comparisons= [compare_results(d, p)
                   for d, p in zip(dtwax_results, python_results)]
    print(f"{np.sum(comparisons)}/{len(comparisons)} queries were correctly processed")
    return np.all(comparisons)


def performance_test():
    print("This is the performance test")
    num_queries = 20000
    segment_size = 32
    query_batch_size = 64*9
    
    ref_len = 64*410 # median full resolution protein squiggle length
    ref_len = 64*512
    query_len = 64*9 # median template length
    reference = np.random.rand(ref_len).astype(np.float32)
    queries = np.random.rand(num_queries, query_len).astype(np.float32)
    file_path = test_utils.write_data(reference, queries)
    dtwax_scores, std_out = test_utils.launch_DTWax(file_path, segment_size, query_batch_size)
    for line in std_out:
        match = re.search(r"TIMING:\s([\d.]+)\sms\s\(concurrent_DTW_kernel_launch\)", line.strip())
        if match:
            kernel_runtime = float(match.group(1))
            break
    print(f"DTWax kernel ran in {kernel_runtime} ms")
    return True


def by_data_file_single_query():
    segment_size = 1
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(script_dir, '../data_files/64_64.txt')
    file_path = os.path.join(script_dir, './passing_ints.txt')

    with open(file_path, "r") as inFile:
        reference = list(map(float, inFile.readline().strip().split()))
        queries = []
        for line in inFile:
            queries.append(list(map(float, line.strip().split())))
    query = queries[0]
    dtwax_scores, dtwax_stdout = test_utils.launch_DTWax(
        file_path, segment_size, thorough=True)
    dtw_matrices = test_utils.process_stdout(
        dtwax_stdout, len(reference), len(queries[0]))
    python_score, python_costs = test_utils.python_dtw_score_matrix(
        reference, query)
    M, N = python_costs.shape
    inf_col = np.full((M, 1), np.inf)
    python_costs_left = np.hstack((inf_col, python_costs[:, :-1]))
    inf_row = np.full((1, N), np.inf)
    python_costs_top = np.vstack((inf_row, python_costs[:-1, :]))
    python_costs_diag = np.vstack((inf_row, python_costs[:-1, :]))
    python_costs_diag = np.hstack((inf_col, python_costs_diag[:, :-1]))
    python_costs_diag[0, 0] = 0

    print("dtwax:")
    print(dtw_matrices[3])
    print("python:")
    print(python_costs_diag)


    diffs = [x == y for x, y in zip(dtw_matrices, [
                                    python_costs, python_costs_left, python_costs_top, python_costs_diag])]
    titles = ["Costs", "Costs left", "Costs top", "Costs diag"]
    cmap = mcolors.ListedColormap(['#d8b365', '#5ab4ac'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, diff, title in zip(axes.flat, diffs, titles):
        im = ax.imshow(diff, cmap=cmap, norm=norm)
        ax.xaxis.tick_top()
        ax.set_title(title)
    legend_patches = [
        mpatches.Patch(color=cmap.colors[1], label='Correct Value'),
        mpatches.Patch(color=cmap.colors[0], label='Incorrect Value')
    ]

    # Add the legend to the figure (position it below the subplots)
    fig.legend(handles=legend_patches, loc='upper center', ncol=2, fontsize=12)
    plt.savefig(f'cost_matrix_diff.png', dpi=150, bbox_inches='tight')
    plt.close()
    return np.all([np.all(x) for x in diffs])


def by_data_file():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(script_dir, '../data_files/failing_ints.txt')
    file_path = os.path.join(script_dir, './failing_ints.txt')
    file_path = os.path.join(script_dir, './passing_ints.txt')
    with open(file_path, "r") as inFile:
        reference = list(map(float, inFile.readline().strip().split()))
        queries = []
        for line in inFile:
            queries.append(list(map(float, line.strip().split())))
    dtwax_scores, dtwax_stdout = test_utils.launch_DTWax(
        file_path, segment_size=2, thorough=True)
    dtwax_matrices = test_utils.process_stdout(
        dtwax_stdout, len(reference), len(queries[0]), len(queries))
    for i, query in enumerate(queries):
        python_score, python_matrix = test_utils.python_dtw_score_matrix(
            reference, query)
        diff = dtwax_matrices[i] != python_matrix
        cmap = mcolors.ListedColormap(['#5ab4ac', '#d8b365'])
        norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
        plt.imshow(diff, cmap=cmap, norm=norm)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.xlabel('reference')
        plt.ylabel('query')
        cbar = plt.colorbar(label='Difference', ticks=[0, 1])
        cbar.ax.set_yticklabels(['correct value', 'incorrect value'])
        plt.title('Matrix Differences')
        plt.savefig(f'cost_matrix_diff_{i}.png')
        plt.close()

    return python_score == dtwax_scores[0]

def random_ints_fast():
    # params to check (make changes here)
    num_queries = 32
    segment_size = 32
    query_batch_size = 64*9
    
    ref_len = 64*410 # median full resolution protein squiggle length
    ref_len = 64*512
    query_len = 64*9 # median template length
    
    reference = np.random.randint(0, 2, ref_len, dtype=np.int32)
    queries = np.random.randint(0, 2, (num_queries, query_len), dtype=np.int32)
    
    # prep data and run
    file_path = test_utils.write_data(reference, queries)
    dtwax_scores, _ = test_utils.launch_DTWax(
        file_path, segment_size, query_batch_size, thorough=False)
    python_scores = test_utils.launch_python_dtw(file_path)
    return test_utils.compare_scores(dtwax_scores, python_scores)

def random_ints_thorough():
    segment_size = 1
    num_queries = 10
    ref_len = 64*segment_size*10
    query_len = 64*10

    reference = np.random.randint(0, 2, ref_len, dtype=np.int32)
    queries = np.random.randint(0, 2, (num_queries, query_len), dtype=np.int32)
    file_path = test_utils.write_data(reference, queries)
    dtwax_scores, std_out = test_utils.launch_DTWax(
        file_path, segment_size, thorough=True)
    dtwax_matrices = test_utils.process_stdout(
        std_out, len(reference), len(queries[0]), len(queries))
    dtwax_results = zip(dtwax_scores, dtwax_matrices)

    python_dtw = functools.partial(
        test_utils.python_dtw_score_matrix, reference)
    with Pool() as pool:
        python_results = pool.map(python_dtw, queries)

    def compare_results(x, y):
        score_x, matrix_x = x
        score_y, matrix_y = y
        return (score_x == score_y) and np.array_equal(matrix_x, matrix_y)

    comparisons = [compare_results(d, p)
                   for d, p in zip(dtwax_results, python_results)]
    print(f"{np.sum(comparisons)}/{len(comparisons)} cost matrices matched")
    false_indices = [i for i, val in enumerate(comparisons) if not val][:10]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    for i in false_indices:
        diff = dtwax_matrices[i] != python_results[i][1]
        cmap = mcolors.ListedColormap(['#5ab4ac', '#d8b365'])
        norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
        plt.imshow(diff, cmap=cmap, norm=norm)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.xlabel('reference')
        plt.ylabel('query')
        cbar = plt.colorbar(label='Difference', ticks=[0, 1])
        cbar.ax.set_yticklabels(['correct value', 'incorrect value'])
        plt.title('Matrix Differences')
        file_path = os.path.join(
            script_dir, f'./matrix_diffs/cost_matrix_diff_{i}.png')
        plt.savefig(file_path)
        print(f"Saved matrix diff plot to {file_path}")
        plt.close()

    return np.all(comparisons)

    # passing, mismatch = test_utils.compare_scores_get_mismatch(dtwax_scores, python_scores)
    # if mismatch is not None:
    #     script_dir = os.path.dirname(os.path.realpath(__file__))
    #     failing_file_path = os.path.join(script_dir, "failing_ints.txt")
    #     with open(failing_file_path, "w") as outFile:
    #         outFile.write(" ".join(map(str, reference)) + "\n")
    #         outFile.write(" ".join(map(str, queries[mismatch])) + "\n")
    #     print(f"Failing case written to {failing_file_path}")

    # # This doesn't necessarily use a passing case, it assumes the first query passed...
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # passing_file_path = os.path.join(script_dir, "passing_ints.txt")
    # with open(passing_file_path, "w") as outFile:
    #     outFile.write(" ".join(map(str, reference)) + "\n")
    #     outFile.write(" ".join(map(str, queries[0])) + "\n")
    # print(f"Passing case written to {passing_file_path}")
    # return passing
