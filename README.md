# FingerprintRecognition
基于OpenCV和tinker的指纹识别系统，使用的硬件为AS608

- `Fingerprint.py` `utils.py` 用于指纹图像的处理和匹配，指纹特征的提取和匹配采用了MCC算法 (Minutia Cylinder-Code)
- `getFingerprint.py` 用于从AS608得到指纹图像
- `savenpz.py` 用于保存npz形式的指纹特征
- `main.py` 主程序与演示窗口

详细报告见`readme.pdf`。
