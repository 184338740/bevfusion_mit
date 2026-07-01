""" ======================= 注册器追踪调试版 - 增强版 ======================= """
import os
import builtins
import sys
import inspect
import traceback

# 全局调试标志
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# 优先使用 pdbpp
try:
    import pdbpp as pdb
    PDB_MODULE = pdb
    PDB_TYPE = "pdbpp"
except ImportError:
    import pdb
    PDB_MODULE = pdb
    PDB_TYPE = "pdb"

# 注册器追踪字典
registry_tracker = {}

def get_project_relative_path(full_path):
    """获取相对于项目根目录的路径"""
    try:
        # 获取当前工作目录（项目根目录）
        cwd = os.getcwd()
        # 返回相对路径
        relative_path = os.path.relpath(full_path, cwd)
        return relative_path
    except:
        return full_path

def debug_print(*args, **kwargs):
    """全局调试打印 - 增强版"""
    if DEBUG:
        try:
            # 获取调用堆栈，跳过debug_print本身
            stack = inspect.stack()
            # 第一个是当前函数，第二个是调用者
            caller_frame = stack[1] if len(stack) > 1 else stack[0]
            
            # 使用完整路径但显示相对项目路径
            full_path = caller_frame.filename
            relative_path = get_project_relative_path(full_path)
            lineno = caller_frame.lineno
            func_name = caller_frame.function
            
            # 添加颜色编码基于文件类型
            file_ext = os.path.splitext(relative_path)[1]
            color_map = {
                '.py': '🟢',   # Python文件
                '.yaml': '🔵', # 配置文件
                '.cpp': '🟡',  # C++文件
                '.cu': '🟣',   # CUDA文件
            }
            color = color_map.get(file_ext, '⚪')
            
            print(f"{color}>>> [{relative_path}:{lineno} {func_name}()]", *args, **kwargs)
        except Exception as e:
            print(f"🔴>>> [获取位置失败: {e}]", *args, **kwargs)

def print_call_stack(limit=10, show_full_path=False):
    """增强版调用栈信息 - 显示完整层级"""
    if DEBUG:
        print("\n" + "="*80)
        print("📞 调用栈追踪 (完整层级):")
        print("="*80)
        
        stack = inspect.stack()
        for i, frame_info in enumerate(stack[2:limit+2]):  # 跳过前2个内部帧
            full_path = frame_info.filename
            if show_full_path:
                path_display = full_path
            else:
                path_display = get_project_relative_path(full_path)
            
            lineno = frame_info.lineno
            func_name = frame_info.function
            
            # 获取代码上下文（如果有）
            code_context = ""
            if frame_info.code_context and frame_info.code_context[0]:
                code_context = frame_info.code_context[0].strip()
                if len(code_context) > 50:
                    code_context = code_context[:47] + "..."
            
            # 层级缩进
            indent = "  " * i
            print(f"{indent}[{i}] {path_display}:{lineno}")
            print(f"{indent}    └── {func_name}()")
            if code_context:
                print(f"{indent}        📝 {code_context}")
        
        print("="*80)

def debug_break():
    """全局断点函数 - 显示完整调用栈"""
    if DEBUG:
        # 获取调用者的帧信息
        stack = inspect.stack()
        caller_frame = stack[1] if len(stack) > 1 else stack[0]
        
        full_path = caller_frame.filename
        relative_path = get_project_relative_path(full_path)
        lineno = caller_frame.lineno
        func_name = caller_frame.function
        
        print("\n" + "🔴"*80)
        print(f"🔴 断点位置: {relative_path}:{lineno} {func_name}()")
        print("🔴"*80)

        # 显示详细调用栈
        print_call_stack(limit=12, show_full_path=False)
        
        print("进入调试模式...")
        PDB_MODULE.set_trace()

def debug_break_at(line_offset=0):
    """在指定偏移位置设置断点"""
    if DEBUG:
        stack = inspect.stack()
        # 根据偏移量计算正确的帧索引
        frame_index = 1 + line_offset
        if frame_index >= len(stack):
            frame_index = len(stack) - 1
        
        caller_frame = stack[frame_index]
        
        full_path = caller_frame.filename
        relative_path = get_project_relative_path(full_path)
        lineno = caller_frame.lineno
        func_name = caller_frame.function
        
        print("\n" + "🟡"*80)
        print(f"🟡 断点位置: {relative_path}:{lineno} {func_name}()")
        print("🟡"*80)

        # 显示调用栈
        print_call_stack(limit=8, show_full_path=False)
        
        print("进入调试模式...")
        PDB_MODULE.set_trace()

def debug_break_if(condition, message=""):
    """条件断点"""
    if DEBUG and condition:
        stack = inspect.stack()
        caller_frame = stack[1] if len(stack) > 1 else stack[0]
        
        full_path = caller_frame.filename
        relative_path = get_project_relative_path(full_path)
        lineno = caller_frame.lineno
        func_name = caller_frame.function
        
        print("\n" + "🔵"*80)
        print(f"🔵 条件断点: {relative_path}:{lineno} {func_name}()")
        if message:
            print(f"🔵 条件: {message}")
        print("🔵"*80)
        
        print_call_stack(limit=8, show_full_path=False)
        
        print("进入调试模式...")
        PDB_MODULE.set_trace()

def track_registry(registry_obj, name="unknown"):
    """追踪注册器"""
    if DEBUG:
        registry_tracker[name] = {
            'obj': registry_obj,
            'modules': [],
            'registry_dict': {}
        }
        debug_print(f"开始追踪注册器: {name}")

def registry_hook(registry_name, module_name, module_class):
    """注册器钩子函数"""
    if DEBUG and registry_name in registry_tracker:
        tracker = registry_tracker[registry_name]
        tracker['modules'].append({
            'name': module_name,
            'class': module_class,
            'file': sys._getframe().f_back.f_code.co_filename
        })
        debug_print(f"注册器 '{registry_name}' 添加模块: {module_name}")

def debug_stack_trace(limit=8, message="调用栈追踪"):
    """增强版调用栈追踪 - 可直接调用"""
    if DEBUG:
        print(f"\n🎯 {message}:")
        print("-" * 60)
        
        stack = inspect.stack()
        for i, frame_info in enumerate(stack[2:limit+2]):
            full_path = frame_info.filename
            relative_path = get_project_relative_path(full_path)
            lineno = frame_info.lineno
            func_name = frame_info.function
            
            # 文件类型图标
            file_ext = os.path.splitext(relative_path)[1]
            icon_map = {
                '.py': '🐍',
                '.yaml': '⚙️',
                '.cpp': '⚡', 
                '.cu': '🔥',
                '': '📄'
            }
            icon = icon_map.get(file_ext, '📄')
            
            print(f"  {icon} [{i}] {relative_path}:{lineno}")
            print(f"     └── {func_name}()")
        
        print("-" * 60)

# ======================= 新增功能：MRO和属性方法分析 =======================

def debug_mro(obj, show_files=True):
    """显示对象的MRO继承链"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🔵 MRO继承链分析: {obj.__class__.__name__}")
    print("-" * 70)
    
    for i, cls in enumerate(obj.__class__.__mro__):
        class_name = cls.__name__
        
        try:
            if cls.__module__ == 'builtins':
                source = "[内置类]"
            else:
                file_path = inspect.getfile(cls)
                if show_files:
                    source = file_path
                else:
                    source = os.path.basename(file_path)
        except TypeError:
            source = "[内置类]"
        except Exception as e:
            source = f"[错误: {e}]"
        
        print(f">>>[xmy]🔵[{i}] {class_name:25} @ {source}")

def debug_methods(obj, show_inherited=True):
    """显示对象的所有方法及定义来源"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🔵 方法来源分析: {obj.__class__.__name__}")
    print("-" * 70)
    
    method_count = 0
    for attr in dir(obj):
        if not attr.startswith('_') and callable(getattr(obj, attr)):
            method_count += 1
            
            # 查找方法最初定义的类
            defining_class = None
            for cls in reversed(obj.__class__.__mro__):
                if (hasattr(cls, attr) and 
                    callable(getattr(cls, attr)) and 
                    attr in cls.__dict__):
                    defining_class = cls
                    break
            
            if defining_class:
                try:
                    if defining_class.__module__ == 'builtins':
                        source = "内置类"
                    else:
                        file_path = inspect.getfile(defining_class)
                        source = os.path.basename(file_path)
                except:
                    source = "未知"
                
                inheritance = "" if defining_class == obj.__class__ else " [继承]"
                print(f">>>[xmy]🔵 {attr:25} <- {defining_class.__name__:20} ({source}){inheritance}")
            else:
                print(f">>>[xmy]🔵 {attr:25} <- [无法确定来源]")
    
    print(f">>>[xmy]🔵 总计: {method_count} 个方法")

def debug_class_defined_methods(obj):
    """显示每个类实际定义的方法（非继承）"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🔵 各MRO类实际定义的方法: {obj.__class__.__name__}")
    print("=" * 70)
    
    for cls in obj.__class__.__mro__:
        if cls.__module__ == 'builtins':
            continue
            
        # 获取类字典中定义的方法（排除继承的）
        class_defined_methods = []
        for attr_name, attr_value in cls.__dict__.items():
            if not attr_name.startswith('_') and callable(attr_value):
                class_defined_methods.append(attr_name)
        
        if class_defined_methods:
            try:
                file_path = inspect.getfile(cls)
                print(f"\n>>>[xmy]🔵 {cls.__name__:20} 定义的方法 @ {os.path.basename(file_path)}:")
                for method in sorted(class_defined_methods):
                    print(f">>>[xmy]🔵   🎯 {method}")
            except Exception as e:
                print(f"\n>>>[xmy]🔵 {cls.__name__:20} 定义的方法 [文件无法获取]:")
                for method in sorted(class_defined_methods):
                    print(f">>>[xmy]🔵   🎯 {method}")

def debug_getattr_info(obj, attr_name):
    """显示getattr调用的详细信息"""
    if not DEBUG:
        return getattr(obj, attr_name)
    
    print(f"\n>>>[xmy]🔵 getattr追踪: {attr_name}")
    print("-" * 60)
    
    # 显示当前调用栈
    debug_stack_trace(limit=4, message="getattr调用位置")
    
    # 查找属性定义位置
    defining_class = None
    for cls in reversed(obj.__class__.__mro__):
        if hasattr(cls, attr_name):
            defining_class = cls
            break
    
    if defining_class:
        try:
            if defining_class.__module__ == 'builtins':
                source = "内置类"
            else:
                attr_value = getattr(defining_class, attr_name)
                file_path = inspect.getfile(attr_value)
                line_no = inspect.getsourcelines(attr_value)[1]
                source = f"{os.path.basename(file_path)}:{line_no}"
        except:
            source = "未知"
        
        print(f">>>[xmy]🔵 属性定义: {defining_class.__name__} @ {source}")
        print(f">>>[xmy]🔵 属性类型: {type(getattr(obj, attr_name)).__name__}")
        print(f">>>[xmy]🔵 是否可调用: {callable(getattr(obj, attr_name))}")
    else:
        print(f">>>[xmy]🔵 属性 {attr_name} 不存在")
    
    print("-" * 60)
    return getattr(obj, attr_name)

def debug_method_execution(method, method_name, *args, **kwargs):
    """调试方法执行过程"""
    if not DEBUG:
        return method(*args, **kwargs)
    
    print(f"\n>>>[xmy]🎯 开始执行: {method_name}")
    
    # 显示方法定义位置
    try:
        file_path = inspect.getfile(method)
        line_no = inspect.getsourcelines(method)[1]
        print(f">>>[xmy]🎯 方法定义: {file_path}:{line_no}")
    except:
        print(f">>>[xmy]🎯 方法定义: [无法获取]")
    
    # 显示调用栈
    debug_stack_trace(limit=4, message="方法执行入口")
    
    # 执行方法
    result = method(*args, **kwargs)
    
    print(f">>>[xmy]✅ 完成执行: {method_name}")
    return result

# ======================= 重写内置函数 =======================

if DEBUG:
    builtins.dprint = debug_print
    builtins.dbreak = debug_break
    builtins.dbreak_at = debug_break_at
    builtins.dbreak_if = debug_break_if
    builtins.dstack = print_call_stack
    builtins.dtrace = debug_stack_trace
    
    # 新增调试函数
    builtins.dmro = debug_mro                    # MRO分析
    builtins.dmethods = debug_methods            # 方法分析
    builtins.dclass_methods = debug_class_defined_methods  # 类定义方法
    builtins.dgetattr = debug_getattr_info       # getattr追踪
    builtins.drun = debug_method_execution       # 方法执行追踪
    
    builtins.track_registry = track_registry
    builtins.registry_hook = registry_hook
    builtins.registry_tracker = registry_tracker
    builtins.DEBUG = True
    
    print(f"=== 注册器追踪调试模式已启用 ({PDB_TYPE}) ===")
    print("使用说明:")
    print("  dbreak()           - 在当前行设置断点，显示调用栈")
    print("  dbreak_at(1)       - 在调用者位置设置断点") 
    print("  dbreak_if(cond)    - 条件断点")
    print("  dstack()           - 详细调用栈")
    print("  dtrace()           - 简洁调用栈追踪")
    print("  dprint()           - 带位置信息的调试打印")
    print("  dmro(obj)          - 显示MRO继承链")
    print("  dmethods(obj)      - 显示所有方法及来源")
    print("  dgetattr(obj, attr)- 追踪getattr调用")
    print("  drun(method, name) - 追踪方法执行")
    print("  track_registry()   - 注册器追踪")
else:
    builtins.dprint = lambda *args, **kwargs: None
    builtins.dbreak = lambda: None
    builtins.dbreak_at = lambda *args: None
    builtins.dbreak_if = lambda *args: None
    builtins.dstack = lambda *args: None
    builtins.dtrace = lambda *args: None
    
    # 新增调试函数的空实现
    builtins.dmro = lambda *args, **kwargs: None
    builtins.dmethods = lambda *args, **kwargs: None
    builtins.dclass_methods = lambda *args, **kwargs: None
    builtins.dgetattr = lambda obj, attr: getattr(obj, attr)
    builtins.drun = lambda method, name, *args, **kwargs: method(*args, **kwargs)
    
    builtins.track_registry = lambda *args, **kwargs: None
    builtins.registry_hook = lambda *args, **kwargs: None
    builtins.DEBUG = False

# ======================= 一行代码调试工具 =======================

def debug_getattr_one_liner(obj, attr_name, default=None):
    """一行代码版本的getattr调试"""
    if not DEBUG:
        return getattr(obj, attr_name, default)
    
    import inspect
    # 显示调用栈
    stack = inspect.stack()
    print(f">>>[xmy]🔵 getattr({obj.__class__.__name__}, '{attr_name}')")
    for i, frame in enumerate(stack[1:4]):
        file_path = os.path.abspath(frame.filename)
        print(f">>>[xmy]🔵[{i}] {file_path}:{frame.lineno} {frame.function}()")
    
    # 查找定义位置
    for cls in reversed(obj.__class__.__mro__):
        if hasattr(cls, attr_name):
            try:
                if cls.__module__ != 'builtins':
                    attr_value = getattr(cls, attr_name)
                    file_path = inspect.getfile(attr_value)
                    print(f">>>[xmy]🔵 定义在: {cls.__name__} @ {file_path}")
            except:
                pass
            break
    
    return getattr(obj, attr_name, default)

def debug_mro_one_liner(obj):
    """一行代码版本的MRO显示"""
    if not DEBUG:
        return
    
    [print(f">>>[xmy]🔵[{i}] {cls.__name__:20} @ {inspect.getfile(cls) if hasattr(cls, '__module__') and cls.__module__ != 'builtins' else '[内置类]'}") for i, cls in enumerate(obj.__class__.__mro__)]

def debug_methods_one_liner(obj):
    """一行代码版本的方法分析"""
    if not DEBUG:
        return
    
    [print(f">>>[xmy]🔵 {attr:20} @ {next((c.__name__ for c in reversed(obj.__class__.__mro__) if hasattr(c, attr) and callable(getattr(c, attr)) and attr in c.__dict__), '继承')}") for attr in dir(obj) if not attr.startswith('_') and callable(getattr(obj, attr))]

# ======================= 注册器调试增强 =======================

def debug_registry_build(registry_name, component_name, cfg):
    """调试注册器构建过程"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🟢 注册器构建: {registry_name} -> {component_name}")
    debug_stack_trace(limit=6, message="注册器构建调用栈")
    
    # 显示配置信息
    if cfg and isinstance(cfg, dict):
        print(f">>>[xmy]🟢 配置信息:")
        for key, value in list(cfg.items())[:3]:  # 只显示前3个配置项
            print(f">>>[xmy]🟢   {key}: {value}")
        if len(cfg) > 3:
            print(f">>>[xmy]🟢   ... 还有 {len(cfg)-3} 个配置项")

def debug_model_components(model):
    """调试模型组件结构"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🔵 模型组件结构: {model.__class__.__name__}")
    print("=" * 70)
    
    # 检查编码器
    if hasattr(model, 'encoders') and model.encoders:
        print(f">>>[xmy]🔵 编码器: {list(model.encoders.keys())}")
        for modality in model.encoders:
            if hasattr(model.encoders[modality], 'keys'):
                print(f">>>[xmy]🔵   {modality}: {list(model.encoders[modality].keys())}")
    
    # 检查融合器
    if hasattr(model, 'fuser'):
        print(f">>>[xmy]🔵 融合器: {type(model.fuser).__name__ if model.fuser else '无'}")
    
    # 检查解码器
    if hasattr(model, 'decoder') and model.decoder:
        print(f">>>[xmy]🔵 解码器: {list(model.decoder.keys())}")
    
    # 检查任务头
    if hasattr(model, 'heads') and model.heads:
        print(f">>>[xmy]🔵 任务头: {list(model.heads.keys())}")

# ======================= 训练过程调试 =======================

def debug_train_epoch(runner, epoch, data_loader):
    """调试训练epoch过程"""
    if not DEBUG:
        return
    
    print(f"\n>>>[xmy]🎯 开始训练 Epoch {epoch}")
    print(f">>>[xmy]🎯 数据加载器: {len(data_loader)} 个batch")
    print(f">>>[xmy]🎯 Runner类型: {runner.__class__.__name__}")
    
    # 显示模型信息
    if hasattr(runner, 'model'):
        model = runner.model
        print(f">>>[xmy]🎯 模型: {model.__class__.__name__}")
        print(f">>>[xmy]🎯 设备: {next(model.parameters()).device}")
        print(f">>>[xmy]🎯 训练模式: {model.training}")

def debug_batch_processing(batch_idx, data_batch):
    """调试batch处理过程"""
    if not DEBUG:
        return
    
    if batch_idx % 50 == 0:  # 每50个batch输出一次
        print(f"\n>>>[xmy]🔵 Batch {batch_idx} 数据信息:")
        
        # 显示batch中的关键信息
        if isinstance(data_batch, dict):
            for key, value in list(data_batch.items())[:5]:  # 只显示前5个key
                if hasattr(value, 'shape'):
                    print(f">>>[xmy]🔵   {key}: {value.shape} {value.dtype}")
                elif isinstance(value, list):
                    print(f">>>[xmy]🔵   {key}: list[{len(value)}]")
                else:
                    print(f">>>[xmy]🔵   {key}: {type(value).__name__}")

# 导出新增函数
if DEBUG:
    builtins.debug_getattr_one_liner = debug_getattr_one_liner
    builtins.debug_mro_one_liner = debug_mro_one_liner
    builtins.debug_methods_one_liner = debug_methods_one_liner
    builtins.debug_registry_build = debug_registry_build
    builtins.debug_model_components = debug_model_components
    builtins.debug_train_epoch = debug_train_epoch
    builtins.debug_batch_processing = debug_batch_processing

# ======================= 初始化完成 =======================

if __name__ == "__main__":
    print("=== BEVFusion调试框架加载完成 ===")
    print("设置环境变量: export DEBUG=true 启用调试模式")
    print("使用: from debug_init import * 导入所有调试功能")