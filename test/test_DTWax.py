import tests
import inspect

# Force reload of run_test.py
import importlib
importlib.reload(tests)

def get_all_tests():
    all_functions = inspect.getmembers(tests, predicate=inspect.isfunction)
    # There are some helper functions that shouldn't be included as a test to run
    tests_to_run = [func for name, func in all_functions if not name.startswith('_')]
    return tests_to_run

def main():
    # Comment out the tests you want to skip here in this list
    tests_to_run = [
        # tests.r64_q64,
        # tests.r256_q64,
        # tests.r64_q256,
        # tests.r256_q256,
        # tests.r256_q256_seg4,
        # tests.r10k_q5k_seg8_count8,
        # tests.r256_q256_seg4_count4,
        # tests.random_r64_q64,
        # tests.protein_id,
        # tests.random_ints,
        tests.failing_int
    ]
    
    # Alternately, this will run all tests, even ones not specified above
    # (useful if you've added tests and don't want to copy paste test names)
    # tests_to_run = get_all_tests()
    
    passing_tests = 0
    for i, test in enumerate(tests_to_run):
        print("~~~~~~~~")
        print(f"Test {i}: {test.__name__}")
        print("~~~~~~~~")
        result = test()
        if result:
            passing_tests += 1
        print("\n\n")
    
    total_tests = len(tests_to_run)
    if passing_tests == total_tests:
        print(f"\033[92m{passing_tests}/{total_tests} tests passed.\033[0m")  # Green if all tests passed
    else:
        print(f"\033[91m{passing_tests}/{total_tests} tests passed.\033[0m")  # Red if any test failed
    
if __name__ == "__main__":
    main()
