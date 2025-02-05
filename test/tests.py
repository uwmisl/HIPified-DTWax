import run_test

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
    # seg_size = 4
    # ref_len = 256
    # query_len = 256
    # reference = [1]*ref_len
    # query = [1]*query_len
    # query[0] = 5
    # queries = [query]
    # return run_test.run_test(reference, queries, seg_size)

def r256_q256_seg4_count4():
    seg_size = 4
    ref_len = 256
    query_len = 256
    reference = [1]*ref_len
    query = [1]*query_len
    query[0] = 5
    queries = [query]*4
    queries[0] = [1]*query_len
    return run_test.run_test(reference, queries, seg_size)