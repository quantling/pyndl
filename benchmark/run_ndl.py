import gc
import time
from pathlib import Path

import pandas as pd
from pyndl import ndl, count, io


def clock(func, args, **kwargs):
    gc.collect()
    start = time.time()
    result = func(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return result, duration


event_dir = Path('events')

LAMBDA_ = 1.0
ALPHA = 0.1
BETAS = (0.1, 0.1)

repeats = 10

df = pd.DataFrame()
for r in range(repeats):
    for file_path in event_dir.glob('*.tab.gz'):
        file_path = str(file_path)
        print(file_path)
    
        _, duration_thread1 = clock(ndl.ndl, (file_path, ALPHA, BETAS, LAMBDA_), n_jobs=1, method='threading')
        print("Threading (single)", duration_thread1)
    
        _, duration_thread4 = clock(ndl.ndl, (file_path, ALPHA, BETAS, LAMBDA_), n_jobs=2, method='threading')
        print("Threading", duration_thread4)
    
        _, duration_omp1 = clock(ndl.ndl, (file_path, ALPHA, BETAS, LAMBDA_), n_jobs=1, method='openmp') 
        print("OpenMP (single)", duration_omp1)
    
        _, duration_omp4 = clock(ndl.ndl, (file_path, ALPHA, BETAS, LAMBDA_), n_jobs=2, method='openmp') 
        print("OpenMP", duration_omp4)
    
        df = pd.concat([df, 
                        pd.DataFrame(
                        {'event_file': [file_path], 
                         'repeats': [r + 1],
                         'wctime-pyndl_thread1': [duration_thread1], 
                         'wctime-pyndl_thread4': [duration_thread4], 
                         'wctime-pyndl_openmp4': [duration_omp4], 
                         'wctime-pyndl_openmp1': [duration_omp1]})], ignore_index=True)
        df.to_csv('pyndl_result.csv')

print(df)

