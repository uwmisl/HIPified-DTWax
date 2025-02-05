# HIPified DTWax

## Testing
There is a python test harness and a suite of functionality tests in `DTWax/test`.

### Setting up the test environment
A requirements.txt is provided to help set up the necessary environment (you may replace `my_env` with a different name of your choosing if you'd like):
```
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

The next time you want to activate the environment you just use:
```
source my_env/bin/activate
```

### Running tests
Then the test suite can be run using:
```
python3 test_DTWax.py
```

This will perform a number of builds and runs of DTWax and will verify the results by comparing the DTWax scores with the [dtw-python](https://pypi.org/project/dtw-python/) scores. Results are printed to the console, something like this:

```
~~~~~~~~
Test 5: r64_q64
~~~~~~~~
Testing a reference of length 64 and a single query of length 64
There are 4 sub-tests which check that DTWax is doing global alignment
Subtest 1: Reference is all 1s, Query is all 1s except query[0]=5
         DTWax scores: [16]
         Python scores: [16.0]
Test passed.
Subtest 2: Reference is all 1s, Query is all 1s except query[-1]=5
         DTWax scores: [16]
         Python scores: [16.0]
Test passed.
Subtest 3: Reference is all 1s except reference[0]=5, Query is all 1s
         DTWax scores: [16]
         Python scores: [16.0]
Test passed.
Subtest 4: Reference is all 1s except reference[-1]=5, Query is all 1s
         DTWax scores: [16]
         Python scores: [16.0]
Test passed.
```

### Adding tests
Tests can be toggled on/off in `test_DTWax.py` and tests can be added to `tests.py`. Helpful utilities live in `run_test.py`

# Acknowledgements

DTWax developed by Hari Sadasivan
- Original Repo: https://github.com/harisankarsadasivan/DTWax/tree/FAST5
- Publication: https://fortuneonline.org/articles/accelerated-dynamic-time-warping.pdf


HIPification primarily done using `hipify-perl`