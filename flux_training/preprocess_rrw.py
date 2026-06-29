import os
import shutil
import glob

# 定义目标目录
transmission_dir = "../sirs/train/rrw/transmission_layer"
blended_dir = "../sirs/train/rrw/blended"

# 如果目标目录不存在，创建它们
os.makedirs(transmission_dir, exist_ok=True)
os.makedirs(blended_dir, exist_ok=True)

# 定义RRWDatasets目录路径
rrw_datasets_dir = "../rrw"  # 请替换为您的实际路径

# 所有可能的子文件夹
all_subdirs = [
    "ref_hf3", "ref_hf4", "ref_hz2", "ref_hz3", "ref_hz4", "ref_hz6", 
    "ref0_vivo1", "reflection_camera1", "reflection_camera2", 
    "reflection_hf0", "reflection_hf1", "reflection_hf2", 
    "reflection_vivo2", "reflection_vivo3"
]

# 用于跟踪全局计数
file_count = 0

# 处理每个主文件夹
for i, main_folder in enumerate(all_subdirs, 1):
    main_folder_path = os.path.join(rrw_datasets_dir, main_folder)
    
    if not os.path.exists(main_folder_path):
        print(f"警告：{main_folder_path} 不存在，跳过")
        continue
        
    print(f"处理主文件夹[{i}]: {main_folder}")
    
    # 查找所有GT文件夹（如GT_hf3）
    gt_dirs = glob.glob(os.path.join(main_folder_path, "GT_*"))
    
    for gt_dir in gt_dirs:
        gt_dir_name = os.path.basename(gt_dir)
        print(f"  处理GT目录: {gt_dir_name}")
        
        # 获取所有GT图像（如wild_out1_GT.png）
        gt_images = sorted(glob.glob(os.path.join(gt_dir, "*_GT.png")))
        
        for j, gt_path in enumerate(gt_images, 1):
            gt_filename = os.path.basename(gt_path)
            prefix = gt_filename.replace("_GT.png", "")  # 提取前缀（如 wild_out1）
            gt_ext = os.path.splitext(gt_filename)[1]    # 获取扩展名（.png）
            
            print(f"    处理GT图像[{j}]: {gt_filename}")
            
            # 查找对应的数据子文件夹 - 注意：它在主文件夹下，而不是GT文件夹下
            data_folder = os.path.join(main_folder_path, prefix)
            if not os.path.exists(data_folder):
                print(f"      警告：未找到对应的数据文件夹 {data_folder}")
                continue
            
            # 获取该数据子文件夹中的所有帧图像
            frame_images = sorted(glob.glob(os.path.join(data_folder, "frame_*.jpg")))
            
            if not frame_images:
                print(f"      警告：在 {data_folder} 中未找到帧图像")
                continue
            
            print(f"      找到 {len(frame_images)} 个帧图像")
            
            # 为每个帧图像创建一个对应的文件对
            for k, frame_path in enumerate(frame_images, 1):
                frame_ext = os.path.splitext(os.path.basename(frame_path))[1]  # 获取帧图像扩展名
                
                # 全局索引计数增加
                file_count += 1
                
                # 创建新的命名文件名
                new_filename = f"{i}_{j}_{k}"
                
                # 复制GT图像到transmission_dir
                new_gt_path = os.path.join(transmission_dir, f"{new_filename}{gt_ext}")
                if not os.path.exists(new_gt_path):
                    shutil.copy(gt_path, new_gt_path)
                    print(f"      复制GT[{file_count}]: {gt_filename} -> {os.path.basename(new_gt_path)}")
                
                # 复制帧图像到blended_dir
                new_frame_path = os.path.join(blended_dir, f"{new_filename}{frame_ext}")
                if not os.path.exists(new_frame_path):
                    shutil.copy(frame_path, new_frame_path)
                    print(f"      复制帧[{file_count}]: {os.path.basename(frame_path)} -> {os.path.basename(new_frame_path)}")

print(f"所有图片组织完成！共处理 {file_count} 对图像。")