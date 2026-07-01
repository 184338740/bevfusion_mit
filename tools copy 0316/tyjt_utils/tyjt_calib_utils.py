import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set, Union, Any


class CalibrationProcessor:
    """标定数据处理"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    @staticmethod
    def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
        """旋转矩阵转四元数"""
        m = np.array(matrix, dtype=np.float64)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        
        return [float(w), float(x), float(y), float(z)]
    
    @staticmethod
    def parse_transform(calib_params: Union[Dict, List]) -> np.ndarray:
        """解析标定参数为4x4变换矩阵 - V1.9修复版"""
        try:
            transform = np.eye(4, dtype=np.float64)
            
            if isinstance(calib_params, Dict):
                # 情况1: hikvision格式 - 有嵌套的transform
                if 'transform' in calib_params:
                    # 处理: {"transform": {"tx": ..., "ty": ..., ...}}
                    actual_params = calib_params['transform']
                    return CalibrationProcessor.parse_transform(actual_params)
                
                # 情况2: all_in_one格式 - 直接的tx/ty/tz/rx/ry/rz/rw
                elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                    translation = np.array([
                        float(calib_params['tx']),
                        float(calib_params['ty']), 
                        float(calib_params['tz'])
                    ], dtype=np.float64)
                    
                    # TYJT格式：[rx, ry, rz, rw] = [x, y, z, w]
                    rx = float(calib_params['rx'])  # x
                    ry = float(calib_params['ry'])  # y
                    rz = float(calib_params['rz'])  # z
                    rw = float(calib_params['rw'])  # w
                    
                    # 传递给函数：[rw, rx, ry, rz] = [w, x, y, z]
                    rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([rw, rx, ry, rz])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                
                # 情况3: translation/rotation格式
                elif 'translation' in calib_params and 'rotation' in calib_params:
                    translation = np.array(calib_params['translation'], dtype=np.float64)
                    rotation_quat = calib_params['rotation']
                    
                    if len(rotation_quat) == 4:
                        # 这里需要确认格式！可能是[x, y, z, w]或[w, x, y, z]
                        # 根据TYJT惯例，假设是[x, y, z, w]
                        x, y, z, w = rotation_quat
                        # 传递给函数：[w, x, y, z]
                        rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([w, x, y, z])
                        transform[:3, :3] = rotation_matrix
                        transform[:3, 3] = translation
                else:
                    # 未知格式，返回单位矩阵
                    print(f"警告：未知的标定参数格式，可用键: {list(calib_params.keys())}")
                    return np.eye(4, dtype=np.float64)
            
            elif isinstance(calib_params, List):
                # 情况4: 列表格式 [tx, ty, tz, rx, ry, rz, rw]
                if len(calib_params) == 7:
                    tx, ty, tz, rx, ry, rz, rw = [float(x) for x in calib_params]
                    translation = np.array([tx, ty, tz], dtype=np.float64)
                    
                    # 传递给函数：[rw, rx, ry, rz] = [w, x, y, z]
                    rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([rw, rx, ry, rz])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                else:
                    print(f"警告：不支持的列表格式长度: {len(calib_params)}")
                    return np.eye(4, dtype=np.float64)
            
            else:
                print(f"警告：不支持的标定参数类型: {type(calib_params)}")
                return np.eye(4, dtype=np.float64)
            
            return transform
            
        except Exception as e:
            print(f"解析标定参数失败: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4, dtype=np.float64)
