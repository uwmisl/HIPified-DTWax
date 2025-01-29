import run_test
import numpy as np
from dtw import dtw
import os

# Force reload of run_test.py
import importlib
importlib.reload(run_test)

def main():
    # Generating an even simple test dataset
    ref = [1]*64
    query = [1]*64
    query[0] = 5
    print(f'Generated a reference of length {len(ref)} and a query of length {len(query)}')
    alignment = dtw(query, ref, dist_method='sqeuclidean', distance_only=True)
    print(f'The Squared Euclidiean dtw distance is {alignment.distance}')

    # Get the location for the test datasets
    # (get the absolute path, so this works wherever this script is called from)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'data.txt')
    with open(file_path, "w") as outFile:
        outFile.write(" ".join(map(str, ref)) + "\n")
        outFile.write(" ".join(map(str, query)) + "\n")
    
    run_test.run_test(file_path)

if __name__ == "__main__":
    main()
