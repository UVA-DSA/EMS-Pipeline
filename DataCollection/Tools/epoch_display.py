import time
import os


os.system('cls' if os.name == 'nt' else 'clear')

while True:
    # Get current epoch time in nanoseconds
    epoch_ns = time.time_ns()

    # Move the cursor to the beginning of the line and overwrite the previous output
    print(f"\r{epoch_ns}", end="", flush=True)

    # Short delay to avoid overwhelming output
    time.sleep(0.01)  # 1 millisecond delay

