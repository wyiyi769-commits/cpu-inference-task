import time
import numpy as np
import os
import sys

def main():
    # --- 1. 参数设置 ---
    TARGET_DURATION_HOURS = 10  # 运行时长：10小时 (36000秒)
    TARGET_MEMORY_GB = 4        # 目标内存：4GB
    TARGET_CPU_LOAD = 0.07      # 目标CPU负载：7%
    

    OUTPUT_DIR = "/data"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "result.txt")

    end_time = time.time() + (TARGET_DURATION_HOURS * 3600)
    
    print(f"[Init] Starting Inference Simulation Task")
    print(f"[Config] Duration: {TARGET_DURATION_HOURS}h | Memory: {TARGET_MEMORY_GB}GB | CPU Target: {TARGET_CPU_LOAD*100}%")

 
    matrix_size = 23170
    print(f"[Memory] Allocating {TARGET_MEMORY_GB}GB RAM (Matrix shape: {matrix_size}x{matrix_size})...")
    
    try:
        huge_matrix = np.ones((matrix_size, matrix_size), dtype=np.float64)
        print(f"[Memory] Allocation successful. Holding memory.")
    except MemoryError:
        print("[Error] Not enough memory to allocate 4GB.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        print(f"[Warn] {OUTPUT_DIR} not found. Switching to current directory for logs.")
        OUTPUT_DIR = "." 
        OUTPUT_FILE = "result.txt"
    
    # 写入开始记录
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"Task Started at: {time.ctime()}\n")
        f.write(f"Config: {TARGET_MEMORY_GB}GB RAM, {TARGET_CPU_LOAD*100}% CPU\n")

    # --- 4. CPU 负载控制 (Load Simulation) ---
    print(f"[Compute] Starting loop to maintain ~{TARGET_CPU_LOAD*100}% CPU usage...")
    
    # 计数器，用于定期写入日志（避免频繁IO）
    loop_count = 0
    
    while time.time() < end_time:
        start_work = time.time()
        
        # [模拟推理计算]
        slice_size = 1000 
        _ = np.dot(huge_matrix[:slice_size, :slice_size], huge_matrix[:slice_size, :slice_size])
        
        work_duration = time.time() - start_work
        
        # [动态休眠]
        if work_duration > 0:
            sleep_time = (work_duration / TARGET_CPU_LOAD) - work_duration
            time.sleep(max(0, sleep_time))
        else:
            time.sleep(0.1)

        # 每隔一段时间（约1小时）更新一次日志文件，证明任务还活着
        loop_count += 1
        if loop_count % 3600 == 0:
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"Running... Current Time: {time.ctime()}\n")
            print(f"[Status] Still running... {time.ctime()}")

    # --- 5. 任务结束 ---
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"Task Finished at: {time.ctime()}\n")
    print("[Finished] Task completed successfully.")

if __name__ == "__main__":
    main()
