import os
import json
import glob

def fix_labelme_json_files(directory):
    """
    修复 LabelMe JSON 文件中 `shape_type="rectangle"` 但 `points` 长度大于 2 的情况。
    LabelMe 标准中，rectangle 的 points 只能包含 2 个点：左上角和右下角。
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    fixed_count = 0
    total_issues = 0

    print(f"[*] 开始检查并修复目录中的 JSON 文件: {directory}")
    print(f"[*] 找到 {len(json_files)} 个 JSON 文件。")

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[!] 无法解析 JSON: {json_path}")
                continue

        needs_fix = False
        shapes = data.get('shapes', [])
        
        for shape in shapes:
            shape_type = shape.get('shape_type')
            points = shape.get('points', [])
            
            # 如果是 rectangle，且点的数量不是 2 (通常是 4)
            if shape_type == 'rectangle' and len(points) > 2:
                total_issues += 1
                needs_fix = True
                
                # 计算边界框 (xmin, ymin) 和 (xmax, ymax)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                # 修复为标准的 2 个点
                shape['points'] = [
                    [xmin, ymin],
                    [xmax, ymax]
                ]

        if needs_fix:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            fixed_count += 1
            print(f"  [+] 已修复: {os.path.basename(json_path)}")

    print("-" * 50)
    print(f"[*] 修复完成！")
    print(f"[*] 共处理了 {fixed_count} 个文件中的 {total_issues} 处格式错误。")
    print(f"[*] 现在你应该可以正常用 LabelMe 打开这个数据集了。")

if __name__ == "__main__":
    target_dir = "/home/fs-ai/llama-qwen/dataset_fences_only"
    fix_labelme_json_files(target_dir)
