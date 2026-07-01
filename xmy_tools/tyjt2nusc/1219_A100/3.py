#!/usr/bin/env python3
"""
精准诊断：检查log_token 'ed790a5d-6ffa-c98b-f443-3882a0f7d752' 的引用问题
"""

import json
from pathlib import Path

def diagnose_specific_token(data_dir):
    """诊断特定Token的引用问题"""
    version_dir = Path(data_dir) / "v1.0-tyjt"
    problem_token = "ed790a5d-6ffa-c98b-f443-3882a0f7d752"
    
    print(f"🔍 诊断Token引用问题: {problem_token}")
    print("=" * 60)
    
    # 1. 检查这个Token是否存在于log表中
    with open(version_dir / "log.json", 'r') as f:
        logs = json.load(f)
    
    log_exists = any(log['token'] == problem_token for log in logs)
    print(f"1. Token是否在log表中: {'✅ 存在' if log_exists else '❌ 不存在'}")
    
    if log_exists:
        for log in logs:
            if log['token'] == problem_token:
                print(f"   对应log记录: {log}")
                break
    
    # 2. 检查scene表是否引用了这个Token
    with open(version_dir / "scene.json", 'r') as f:
        scenes = json.load(f)
    
    problematic_scenes = []
    for i, scene in enumerate(scenes):
        if scene.get('log_token') == problem_token:
            problematic_scenes.append((i, scene))
    
    print(f"\n2. scene表引用检查:")
    if problematic_scenes:
        print(f"   ⚠️  有 {len(problematic_scenes)} 个scene引用了此Token")
        for idx, scene in problematic_scenes[:3]:  # 只显示前3个
            print(f"     Scene[{idx}]: token={scene['token'][:16]}..., name={scene['name']}")
    else:
        print("   ✅ 没有scene直接引用此Token")
    
    # 3. 检查map表的log_tokens数组
    with open(version_dir / "map.json", 'r') as f:
        maps = json.load(f)
    
    print(f"\n3. map.log_tokens检查:")
    if maps:
        map_record = maps[0]
        log_tokens = map_record.get('log_tokens', [])
        print(f"   map.log_tokens 数量: {len(log_tokens)}")
        
        if problem_token in log_tokens:
            print(f"   ✅ Token在map.log_tokens中")
        else:
            print(f"   ⚠️  Token不在map.log_tokens中")
            
        # 检查map.log_tokens中的所有token是否都存在于log表
        log_token_set = {log['token'] for log in logs}
        missing_in_log = [t for t in log_tokens if t not in log_token_set]
        
        if missing_in_log:
            print(f"   ❌ map.log_tokens中有 {len(missing_in_log)} 个token不在log表中")
            for token in missing_in_log[:3]:
                print(f"     缺失token: {token}")
        else:
            print(f"   ✅ map.log_tokens中的所有token都存在于log表")
    
    # 4. 检查其他可能的引用
    print(f"\n4. 其他表检查:")
    
    # 检查sample表是否有log_token字段（正常情况下不应该有）
    with open(version_dir / "sample.json", 'r') as f:
        samples = json.load(f)
    
    samples_with_log_token = [s for s in samples if 'log_token' in s]
    print(f"   sample表包含log_token字段的记录: {len(samples_with_log_token)}")
    
    # 检查sample_data表
    with open(version_dir / "sample_data.json", 'r') as f:
        sample_datas = json.load(f)
    
    # 查找任何包含此Token的字段
    found_in_other_tables = []
    for table_name in ['sample', 'sample_data', 'sample_annotation', 'instance', 'ego_pose', 'calibrated_sensor', 'sensor']:
        table_file = version_dir / f"{table_name}.json"
        if table_file.exists():
            with open(table_file, 'r') as f:
                table_data = json.load(f)
            
            for record in table_data:
                for key, value in record.items():
                    if value == problem_token:
                        found_in_other_tables.append((table_name, record.get('token', 'unknown'), key))
    
    if found_in_other_tables:
        print(f"   ⚠️  在其他表中找到此Token引用:")
        for table_name, record_token, field_name in found_in_other_tables[:5]:
            print(f"     表:{table_name}, 记录:{record_token[:16]}..., 字段:{field_name}")
    else:
        print(f"   ✅ 此Token没有出现在其他异常表中")

if __name__ == "__main__":
    data_dir = "./output-1226-v1.9/step1/nuscenes_tyjt"
    diagnose_specific_token(data_dir)