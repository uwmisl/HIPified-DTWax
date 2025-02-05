import subprocess
import os
import re
from dtw import dtw

def python_dtw_score(reference, query):
    return dtw(query, reference, dist_method='sqeuclidean', distance_only=True).distance

def run_test(reference, queries, segment_size=1):
    # Get the location for the test datasets
    # (get the absolute path, so this works wherever this script is called from)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'data.temp')
    with open(file_path, "w") as outFile:
        outFile.write(" ".join(map(str, reference)) + "\n")
        for query in queries:
            outFile.write(" ".join(map(str, query)) + "\n")
    dtwax_scores = run_DTWax(file_path, segment_size)
    python_scores = [python_dtw_score(reference, query) for query in queries]
    
    print(f"\t DTWax scores: [{', '.join(map(str, dtwax_scores))}]")
    print(f"\t Python scores: [{', '.join(map(str, python_scores))}]")
    
    passed = all(dtwax_score == python_score for dtwax_score, python_score in zip(dtwax_scores, python_scores))
    if passed:
        print("\033[92mTest passed.\033[0m") # prints in green
    else:
        print("\033[91mTest failed.\033[0m")  # prints in red
        print(f"\t DTWax scores: [{', '.join(map(str, dtwax_scores))}]")
        print(f"\t Python scores: [{', '.join(map(str, python_scores))}]")

    return passed
        

def run_DTWax(data_file, segment_size=1):
    build_script = "../build/build_DTWax.sh"
    main_executable = "../src/main_debug"

    # Build command
    try:
        subprocess.run([build_script, data_file, str(segment_size)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed for {data_file}.")
        print("stderr:\n", e.stderr.strip())
        return

    # Run the main_debug executable
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
                # print(line.strip())
                continue
            if found_results and re.match(r"^\d+\s+\d+\s+\d+\s+\d+", line.strip()):
                # print(line.strip())
                scores.append(int(line.strip().split()[-1]))
        # print(scores)
        return scores
    except subprocess.CalledProcessError as e:
        print("Execution failed.")
        print("stderr:\n", e.stderr.strip())
