#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import json
import os
from collections import Counter

def build_sample_to_scene(json_dir):
    """
    从原始 JSON 文件构建 sample_token -> scene_name 映射
    """
    sample_file = os.path.join(json_dir, 'sample.json')
    scene_file = os.path.join(json_dir, 'scene.json')

    with open(sample_file, 'r') as f:
        samples = json.load(f)
    with open(scene_file, 'r') as f:
        scenes = json.load(f)

    # scene_token -> scene_name
    scene_name_map = {scene['token']: scene['name'] for scene in scenes}

    # sample_token -> scene_name
    sample_scene_map = {}
    for sample in samples:
        scene_token = sample['scene_token']
        sample_scene_map[sample['token']] = scene_name_map.get(scene_token, 'unknown')

    return sample_scene_map

def analyze_pkl(pkl_path, sample_scene_map):
    """
    分析 pkl 文件，返回场景计数器
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 提取 infos 列表（兼容 dict 或 list 格式）
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    elif isinstance(data, list):
        infos = data
    else:
        raise ValueError("pkl 格式不符合预期，应为包含 'infos' 的字典或列表")

    counter = Counter()
    unmatched = 0
    for info in infos:
        token = info.get('token')  # 样本 token
        if token in sample_scene_map:
            counter[sample_scene_map[token]] += 1
        else:
            unmatched += 1

    if unmatched > 0:
        print(f"警告: {unmatched} 个样本无法匹配到场景（可能不是训练/验证集中的样本？）")
    return counter

def save_scene_list(counter, filename):
    with open(filename, 'w') as f:
        for scene, count in counter.most_common():
            f.write(f"{scene}: {count}\n")

def main():
    # ====== 请根据您的实际路径修改以下变量 ======
    base_dir = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt"
    train_pkl = os.path.join(base_dir, "tyjt_infos_train.pkl")
    val_pkl   = os.path.join(base_dir, "tyjt_infos_val.pkl")
    json_dir  = os.path.join(base_dir, "v1.0-tyjt-trainval")
    # ==========================================

    print("正在构建 sample → scene 映射...")
    sample_scene_map = build_sample_to_scene(json_dir)
    print(f"映射构建完成，共包含 {len(sample_scene_map)} 个样本\n")

    # 分析训练集并保存
    print("=" * 60)
    print("训练集 场景统计")
    train_counter = analyze_pkl(train_pkl, sample_scene_map)
    print(f"总场景数: {len(train_counter)}")
    print("场景列表（按样本数排序）:")
    for scene, count in train_counter.most_common():
        print(f"  {scene}: {count}")
    save_scene_list(train_counter, "train_scenes.txt")
    print("训练集场景列表已保存至 train_scenes.txt\n")

    # 分析验证集并保存
    print("=" * 60)
    print("验证集 场景统计")
    val_counter = analyze_pkl(val_pkl, sample_scene_map)
    print(f"总场景数: {len(val_counter)}")
    print("场景列表（按样本数排序）:")
    for scene, count in val_counter.most_common():
        print(f"  {scene}: {count}")
    save_scene_list(val_counter, "val_scenes.txt")
    print("验证集场景列表已保存至 val_scenes.txt")

if __name__ == "__main__":
    main()