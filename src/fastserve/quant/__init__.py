"""Reference quantizers, written for reading: RTN, FP8/NF4 formats, GPTQ, AWQ, Hadamard rotation, W8A8.

Every function here is "fake" quantization: values are snapped to a low-bit grid and immediately turned back
into floats. That reproduces the exact error of low-bit storage, which is what quality experiments need.
Speed needs real low-bit kernels, which is M4's job.
"""
