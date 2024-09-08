import os

count = 0
tasks = os.listdir('./final_bench')
for t in tasks:
    samples = os.listdir(f'./final_bench/{t}/')
    count += len(samples)

print(count)
    