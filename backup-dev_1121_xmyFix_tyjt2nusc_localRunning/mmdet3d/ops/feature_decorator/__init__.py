# FIX: 临时解决循环导入错误
# 错误: ImportError: cannot import name 'feature_decorator_ext' from partially initialized module 'mmdet3d.ops.feature_decorator' 
# 日期: 2025-10-09, xmy

# 方法1：直接注释
# from .feature_decorator import feature_decorator

# 方法2： 采用try方式， 则 bevfusion_mit_xmy/mmdet3d/ops/__init__.py 中的代码也不需要注释了
"""
try:
    from .feature_decorator_ext import feature_decorator_ext
    feature_decorator = feature_decorator_ext.feature_decorator
except ImportError:
    # Fallback to Python implementation
    from .feature_decorator import feature_decorator
    class MockExt:
        def __init__(self):
            self.feature_decorator = feature_decorator
    feature_decorator_ext = MockExt()

__all__ = ['feature_decorator_ext', 'feature_decorator']
"""