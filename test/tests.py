import test_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool
import functools

# Force reload of test_utils.py
import importlib
importlib.reload(test_utils)

# Notation:
# rX = reference of length X
# qX = queries of length X
# segX = using a segment size of X
# countX = using X number of queries

def _run_4_subtests(ref_len, query_len, segment_size=1):
    print("There are 4 sub-tests which check that DTWax is doing global alignment")
    print("Subtest 1: Reference is all 1s, Query is all 1s except query[0]=5")
    test_passes = True
    reference = [1]*ref_len
    query = [1]*query_len
    query[0] = 5
    queries = [query]
    subtest_passed = test_utils.run_test(reference, queries, segment_size)
    test_passes = test_passes and subtest_passed
    
    print("Subtest 2: Reference is all 1s, Query is all 1s except query[-1]=5")
    reference = [1]*ref_len
    query = [1]*query_len
    query[-1] = 5
    queries = [query]
    subtest_passed = test_utils.run_test(reference, queries, segment_size)
    test_passes = test_passes and subtest_passed
    
    print("Subtest 3: Reference is all 1s except reference[0]=5, Query is all 1s")
    reference = [1]*ref_len
    query = [1]*query_len
    reference[0] = 5
    queries = [query]
    subtest_passed = test_utils.run_test(reference, queries, segment_size)
    test_passes = test_passes and subtest_passed
    
    print("Subtest 4: Reference is all 1s except reference[-1]=5, Query is all 1s")
    reference = [1]*ref_len
    query = [1]*query_len
    reference[-1] = 5
    queries = [query]
    subtest_passed = test_utils.run_test(reference, queries, segment_size)
    test_passes = test_passes and subtest_passed
    
    return test_passes

def r64_q64():
    print("Testing a reference of length 64 and a single query of length 64")
    return _run_4_subtests(64, 64)

def r256_q64():
    print("Testing a reference of length 256 and a single query of length 64")
    print("This will test reference batching (dividing the reference into smaller sections)")
    print("A length of 256 will give 4 reference batches, each of length 64")
    return _run_4_subtests(256, 64)
    
def r64_q256():
    print("Testing a reference of length 64 with a single query of length 256")
    print("This will test query batching (dividing query into smaller sections)")
    print("A length of 256 will give 4 query batches, each of length 64")
    return _run_4_subtests(64, 256)
    
def r256_q256():
    print("Testing a reference of length 256 with a single query of length 256")
    print("This will test reference batching in conjunction with query batching")
    return _run_4_subtests(256, 256)

def r256_q256_seg4_count4():
    reference = [1]*256
    query = [1]*256
    query[0] = 5
    queries = [query]*4
    queries[0] = [1]*256
    return test_utils.run_test(reference, queries, 4)

def random_r64_q64():
    reference = np.random.rand(64).astype(np.float32)
    queries = [np.random.rand(64).astype(np.float32)]
    return test_utils.run_test(reference, queries)

def r38k_q1728_count20():
    reference = np.random.rand(38336).astype(np.float32)
    queries = np.random.rand(20, 1728).astype(np.float32)
    return test_utils.run_test(reference, queries, segment_size=1)

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
    dtwax_scores, dtwax_stdout = test_utils.launch_DTWax(file_path, segment_size=2, debug=True)
    dtwax_matrices = test_utils.process_stdout(dtwax_stdout, len(reference), len(queries[0]), len(queries))
    for i, query in enumerate(queries):
        python_score, python_matrix = test_utils.python_dtw_score_debug(reference, query)
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
    segment_size = 2
    reference = np.random.randint(0, 2, 64*segment_size*2, dtype=np.int32)
    queries = np.random.randint(0, 2, (10, 64*2), dtype=np.int32)
    file_path = test_utils.write_temp_data(reference, queries)
    dtwax_scores, _ = test_utils.launch_DTWax(file_path, segment_size, debug=False)
    python_scores = test_utils.launch_python_dtw(file_path)
    return test_utils.compare_scores(dtwax_scores, python_scores)

def random_ints_thorough():
    segment_size = 2
    reference = np.random.randint(0, 2, 64*segment_size*2, dtype=np.int32)
    queries = np.random.randint(0, 2, (10, 64*2), dtype=np.int32)
    file_path = test_utils.write_temp_data(reference, queries)
    dtwax_scores, std_out = test_utils.launch_DTWax(file_path, segment_size, debug=True)
    dtwax_matrices = test_utils.process_stdout(std_out, len(reference), len(queries[0]), len(queries))
    dtwax_results = zip(dtwax_scores, dtwax_matrices)

    python_dtw = functools.partial(test_utils.python_dtw_score_debug, reference)
    with Pool() as pool:
        python_results = pool.map(python_dtw, queries)

    def compare_results(x,y):
        score_x, matrix_x = x
        score_y, matrix_y = y
        return (score_x == score_y) and np.array_equal(matrix_x, matrix_y)
    
    comparisons = [compare_results(d, p) for d, p in zip(dtwax_results, python_results)]
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
        file_path = os.path.join(script_dir, f'./matrix_diffs/cost_matrix_diff_{i}.png')
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

def protein_id():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'protein_id_test.txt')
    dtwax_scores, _ = test_utils.launch_DTWax(file_path, segment_size=1)
    python_scores = test_utils.launch_python_dtw(file_path)
    return test_utils.compare_scores(dtwax_scores, python_scores)

def failing_seg2():
    reference = [1]*128
    query = [1]*64
    reference[0] = 5
    # query[0] = 5
    file_path = test_utils.write_temp_data(reference, [query])
    dtwax_scores, dtwax_stdout = test_utils.launch_DTWax(file_path, segment_size=2)
    dtwax_matrix = test_utils.process_stdout(dtwax_stdout, len(reference), len(query))
    python_score, python_matrix = test_utils.python_dtw_score_debug(reference, query)
    print("DTWax:")
    print(dtwax_matrix)
    print("Python:")
    print(python_matrix)
    return python_score == dtwax_scores[0]