import run_test
import numpy as np
import os

# Force reload of run_test.py
import importlib
importlib.reload(run_test)

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
    test_passes = test_passes and run_test.run_test(reference, queries, segment_size)
    
    print("Subtest 2: Reference is all 1s, Query is all 1s except query[-1]=5")
    reference = [1]*ref_len
    query = [1]*query_len
    query[-1] = 5
    queries = [query]
    test_passes = test_passes and run_test.run_test(reference, queries)
    
    print("Subtest 3: Reference is all 1s except reference[0]=5, Query is all 1s")
    reference = [1]*ref_len
    query = [1]*query_len
    reference[0] = 5
    queries = [query]
    test_passes = test_passes and run_test.run_test(reference, queries)
    
    print("Subtest 4: Reference is all 1s except reference[-1]=5, Query is all 1s")
    reference = [1]*ref_len
    query = [1]*query_len
    reference[-1] = 5
    queries = [query]
    test_passes = test_passes and run_test.run_test(reference, queries)
    
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
    
def r256_q256_seg4():
    print("Testing a reference of length 256 with a single query of length 256 and segment size of 4")
    return _run_4_subtests(256, 256, 4)

def r256_q256_seg4_count4():
    reference = [1]*256
    query = [1]*256
    query[0] = 5
    queries = [query]*4
    queries[0] = [1]*256
    return run_test.run_test(reference, queries, 4)

def random_r64_q64():
    reference = np.random.rand(64).astype(np.float32)
    queries = [np.random.rand(64).astype(np.float32)]
    return run_test.run_test(reference, queries)

def r10k_q5k_seg8_count8():
    reference = np.random.rand(64*5000).astype(np.float32)
    queries = np.random.rand(8, 64*10).astype(np.float32)
    return run_test.run_test(reference, queries, 1)

def random_ints():
    reference = np.random.randint(0, 2, 64*2, dtype=np.int32)
    queries = np.random.randint(0, 2, (10000, 64*2), dtype=np.int32)
    file_path = run_test.write_temp_data(reference, queries)
    dtwax_scores = run_test.launch_DTWax(file_path, segment_size)
    python_scores = run_test.launch_python_dtw(file_path)
    passing, mismatch = run_test.compare_scores_get_mismatch(dtwax_scores, python_scores)
    if mismatch is not None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        failing_file_path = os.path.join(script_dir, "failing_ints.txt")
        with open(failing_file_path, "w") as outFile:
            outFile.write(" ".join(map(str, reference)) + "\n")
            outFile.write(" ".join(map(str, queries[mismatch])) + "\n")
        print(f"Failing case written to {failing_file_path}")
    return passing

def protein_id():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'protein_id_test.txt')
    dtwax_scores = run_test.launch_DTWax(file_path, segment_size=8)
    python_scores = run_test.launch_python_dtw(file_path)
    return run_test.compare_scores(dtwax_scores, python_scores)

def failing_int():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, '../data_files/failing_ints.txt')
    dtwax_scores = run_test.launch_DTWax(file_path)
    python_scores = run_test.launch_python_dtw(file_path)
    return run_test.compare_scores(dtwax_scores, python_scores)