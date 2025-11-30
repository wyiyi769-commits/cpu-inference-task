import time
import numpy as np
import os
import sys

def main():
    # --- 1. 参数设置 ---
    TARGET_DURATION_HOURS = 10  # 运行时长：10小时
    TARGET_MEMORY_GB = 4        # 目标内存：4GB
    TARGET_CPU_LOAD = 0.07      # 目标CPU负载：7% (在5-10%之间)
    
    end_time = time.time() + (TARGET_DURATION_HOURS * 3600)
    
    print(f"[Init] Starting Inference Simulation Task")
    print(f"[Config] Duration: {TARGET_DURATION_HOURS}h | Memory: {TARGET_MEMORY_GB}GB | CPU Target: {TARGET_CPU_LOAD*100}%")

    # --- 2. 内存锁定 (Memory Allocation) ---
    # float64 占用 8 bytes. 
    # 4GB = 4 * 1024^3 bytes
    # 元素数量 = 4 * 1024^3 / 8 = 536,870,912
    # 开方取整构建方阵: sqrt(536,870,912) ≈ 23170
    
    matrix_size = 23170
    print(f"[Memory] Allocating {TARGET_MEMORY_GB}GB RAM (Matrix shape: {matrix_size}x{matrix_size})...")
    
    try:
        # 使用 np.ones 强制分配并初始化内存（避免Lazy allocation）
        # 注意：这里会占用大量物理内存，请确保宿主机资源充足
        huge_matrix = np.ones((matrix_size, matrix_size), dtype=np.float64)
        print(f"[Memory] Allocation successful. Holding memory.")
    except MemoryError:
        print("[Error] Not enough memory to allocate 4GB.")
        sys.exit(1)

    # --- 3. CPU 负载控制 (Load Simulation) ---
    print(f"[Compute] Starting loop to maintain ~{TARGET_CPU_LOAD*100}% CPU usage...")
    
    while time.time() < end_time:
        start_work = time.time()
        
        # [模拟推理计算]
        # 做一个小规模的矩阵乘法作为“脉冲”计算
        # 使用切片计算避免单次计算时间过长导致CPU瞬时冲高
        slice_size = 1000 
        _ = np.dot(huge_matrix[:slice_size, :slice_size], huge_matrix[:slice_size, :slice_size])
        
        work_duration = time.time() - start_work
        
        # [动态休眠]
        # 公式：Load = Work / (Work + Sleep)
        # Sleep = (Work / Load) - Work
        if work_duration > 0:
            sleep_time = (work_duration / TARGET_CPU_LOAD) - work_duration
            time.sleep(sleep_time)
        else:
            time.sleep(0.1) # 防止计算太快出现除零等异常

    print("[Finished] Task completed successfully.")

if __name__ == "__main__":
    main()
