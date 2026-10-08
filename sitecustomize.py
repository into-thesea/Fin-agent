"""sitecustomize — Python 启动时自动设置环境变量

解决 Windows 上 numpy/OpenBLAS 多线程内存分配失败问题。
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
