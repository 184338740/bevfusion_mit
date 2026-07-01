import yaml
import argparse
import pprint
from collections import OrderedDict
import re

Debug = False

class UniversalYAMLParser:
    """通用YAML解析器：支持非标准语法、结构化打印、PDB调试（修复变量注入）"""
    
    def __init__(self, yaml_path, resolve_variables=True):
        self.yaml_path = yaml_path
        self.resolve_variables = resolve_variables
        self.raw_config = None  
        self.config = None      
        self._load_and_parse()
        if Debug: import pdb; pdb.set_trace()

    def _preprocess_yaml_content(self, raw_content):
        """预处理非标准YAML语法"""
        # 处理Python切片 [1,2,3][:2] → [1,2,3]
        slice_pattern = re.compile(r'(\[.*?\])\[\:\d+\]')
        content = slice_pattern.sub(r'\1', raw_content)
        # 处理变量运算 ${var}*2 → ${var}
        math_pattern = re.compile(r'(\$\{[\w]+\})\*?\d+')
        content = math_pattern.sub(r'\1', content)
        return content

    def _load_raw_yaml(self):
        """加载并预处理YAML"""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            preprocessed_content = self._preprocess_yaml_content(raw_content)
            self.raw_config = yaml.safe_load(preprocessed_content)
            return raw_content, preprocessed_content
        except Exception as e:
            raise RuntimeError(f"加载YAML失败: {str(e)}\n提示：可能包含Python切片/运算等非标准YAML语法")

    def _resolve_yaml_variables(self, raw_content):
        """解析${变量}引用"""
        if not self.resolve_variables:
            return raw_content
        var_pattern = re.compile(r'^(\w+):\s*(.+?)$', re.MULTILINE)
        variables = OrderedDict()
        lines = raw_content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('#') or ':' not in line_stripped or '{' in line_stripped:
                continue
            match = var_pattern.match(line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                if not any(char in value for char in ['{', '}', '[', ']']):
                    variables[key] = value
        resolved_content = raw_content
        for var_name, var_value in variables.items():
            resolved_content = resolved_content.replace(f'${{{var_name}}}', var_value)
        return resolved_content

    def _load_and_parse(self):
        """核心解析逻辑"""
        raw_content, preprocessed_content = self._load_raw_yaml()
        if self.resolve_variables:
            resolved_content = self._resolve_yaml_variables(preprocessed_content)
        else:
            resolved_content = preprocessed_content
        try:
            self.config = yaml.safe_load(resolved_content)
            if isinstance(self.config, dict):
                self.config = OrderedDict(self.config)
        except yaml.YAMLError as e:
            raise RuntimeError(f"解析YAML失败（预处理后）: {str(e)}")

    def print_structured(self, indent=4, depth=None, sort_keys=False):
        """结构化打印"""
        print("="*80)
        print(f"YAML文件结构化内容（{self.yaml_path}）")
        print("="*80)
        pprint.pprint(
            self.config,
            indent=indent,
            depth=depth,
            sort_dicts=sort_keys,
            width=120
        )
        print("\n  注意：已自动处理Python切片等非标准YAML语法")

    def get_config(self):
        """获取配置字典"""
        return self.config

    # def debug_mode(self):
    #     """修复版调试模式：强制注入变量到PDB环境"""
    #     print("\n" + "="*80)
    #     print("进入PDB调试模式 - 可用变量：")
    #     print("  - cfg = parser.get_config()  # 完整配置（核心推荐）")
    #     print("  - raw_cfg = parser.raw_config  # 预处理后的配置")
    #     print("  - parser = 当前解析器实例")
    #     print("="*80 + "\n")
        
    #     # 强制定义变量（解决作用域问题）
    #     cfg = self.get_config()
    #     raw_cfg = self.raw_config
        
    #     # 启动PDB并保留变量
    #     import pdb
    #     pdb.set_trace()

def main():
    parser = argparse.ArgumentParser(description='通用YAML解析工具（修复PDB变量注入）')
    parser.add_argument('--yaml_path', required=True, help='YAML文件路径')
    parser.add_argument('--no_resolve', action='store_true', help='禁用${变量}解析')
    # parser.add_argument('--debug', action='store_true', help='启动PDB调试模式')
    parser.add_argument('--sort', action='store_true', help='打印时按字母排序key')
    
    args = parser.parse_args()
    
    yaml_parser = UniversalYAMLParser(
        yaml_path=args.yaml_path,
        resolve_variables=not args.no_resolve
    )
    
    yaml_parser.print_structured(
        indent=2,
        depth=3,
        sort_keys=args.sort
    )
    
    # xmy 
    cfg = yaml_parser.config
    # print(f'{"-"*30} [train_pipeline] {"-"*30}')
    print(f'cfg["train_pipeline"]: ')
    lens_train_pipeline = len(cfg["train_pipeline"])
    for i, train_i in enumerate(cfg["train_pipeline"]):
        # 用dict.get()容错：没有'type'键时返回"未知组件（无type键）"
        comp_type = train_i.get('type', '未知组件（无type键）')
        # 打印时标注异常项，方便定位
        print(f"  {i}/{lens_train_pipeline}: {comp_type}")
        # 可选：打印异常项的完整内容，排查问题
        if 'type' not in train_i:
            print(f"    ❌ 异常项{i}完整内容：{train_i}")

    # print(f'{"-"*30} [train_pipeline] {"-"*30}')
    # pprint.pprint(
    #     cfg["train_pipeline"],
    #     indent=2,
    #     depth=2,
    #     sort_dicts=False,
    #     width=120
    # )
    


if __name__ == '__main__':
    main()