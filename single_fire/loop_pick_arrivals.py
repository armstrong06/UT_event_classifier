import subprocess

# path the pick_arrivals.py from where run_pick_arrivals.py is being called
code_path = "."
# List of values for --number argument
source_types = ["EX", "EQ"]

# Loop through the values and call the script with different arguments
for source in source_types:
    # Call the script with different argparse values.
    subprocess.run(['python', f'{code_path}/pick_arrivals.py',
                     '--source', source,
                     '-e', 'BASE',
                     '-o', "/uufs/chpc.utah.edu/common/home/koper-group3/alysha",
                     ])
