import subprocess

def run_test(data_file):
    build_script = "../build/build_DTWax.sh"
    main_executable = "../src/main_debug"

    # Build command
    try:
        subprocess.run([build_script, data_file], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
        print(f"Build succeeded for {data_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed for {data_file}.")
        print("Error Output:", e.stderr.strip())
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
        for line in result_lines:
            if "Results:" in line:
                found_results = True
                continue
            if found_results and line.strip():
                print(line.strip())
    except subprocess.CalledProcessError as e:
        print("Execution failed.")
        print("Error Output:", e.stderr.strip())
