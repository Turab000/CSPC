import time
import decay

N0 = 200000
rate = 0.4

# Pure Python loop version
start = time.perf_counter()
decay.simulate_loop(N0, rate)
end = time.perf_counter()

loop_time = end - start


# NumPy version
start = time.perf_counter()
decay.simulate(N0, rate)
end = time.perf_counter()

numpy_time = end - start


# Speed-up calculation
speedup = loop_time / numpy_time


print(f"Pure Python loop: {loop_time:.6f} s")
print(f"NumPy version:     {numpy_time:.6f} s")
print(f"NumPy is {speedup:.2f}x faster")