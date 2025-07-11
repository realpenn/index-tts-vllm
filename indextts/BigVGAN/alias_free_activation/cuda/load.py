# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import subprocess
import hashlib
import time

from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


import re
import shutil
import tempfile

# 全局缓存变量
_compiled_module_cache = None

# 是否启用缓存机制（可通过环境变量控制）
ENABLE_CUDA_CACHE = os.environ.get("BIGVGAN_ENABLE_CACHE", "1").lower() in ("1", "true", "yes")

# 补丁修复：sources 路径含中文字符时，生成 build.ninja 乱码导致编译失败
# 使用临时目录来规避 ninja 编译失败（比如中文路径）
def chinese_path_compile_support(sources, buildpath):
    pattern = re.compile(r'[\u4e00-\u9fff]')  
    if not bool(pattern.search(str(sources[0].resolve()))):
        return buildpath # 检测非中文路径跳过
    # Create build directory
    resolves = [ item.name for item in sources]
    ninja_compile_dir = os.path.join(tempfile.gettempdir(), "BigVGAN", "cuda")
    os.makedirs(ninja_compile_dir, exist_ok=True)
    new_buildpath = os.path.join(ninja_compile_dir, "build")
    os.makedirs(new_buildpath, exist_ok=True)
    print(f"ninja_buildpath: {new_buildpath}")
    # Copy files to directory
    sources.clear()
    current_dir = os.path.dirname(__file__)
    ALLOWED_EXTENSIONS = {'.py', '.cu', '.cpp', '.h'}
    for filename in os.listdir(current_dir):
        item = pathlib.Path(current_dir).joinpath(filename)
        tar_path = pathlib.Path(ninja_compile_dir).joinpath(item.name)
        if not item.suffix.lower() in ALLOWED_EXTENSIONS:continue
        pathlib.Path(shutil.copy2(item, tar_path))
        if tar_path.name in resolves:sources.append(tar_path)
    return new_buildpath



def _get_source_hash(sources):
    """计算源文件的哈希值，用于检测是否需要重新编译"""
    hasher = hashlib.md5()
    for source_path in sources:
        if os.path.exists(source_path):
            with open(source_path, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()


def _get_cache_file_path(buildpath):
    """获取缓存文件路径"""
    return os.path.join(buildpath, "compilation_hash.txt")


def _should_recompile(sources, buildpath):
    """检查是否需要重新编译"""
    cache_file = _get_cache_file_path(buildpath)
    current_hash = _get_source_hash(sources)
    
    # 如果缓存文件不存在，需要编译
    if not os.path.exists(cache_file):
        return True, current_hash
    
    # 读取之前的哈希值
    try:
        with open(cache_file, 'r') as f:
            cached_hash = f.read().strip()
        # 如果哈希值不同，需要重新编译
        return cached_hash != current_hash, current_hash
    except:
        return True, current_hash


def _save_compilation_hash(buildpath, source_hash):
    """保存编译哈希值"""
    cache_file = _get_cache_file_path(buildpath)
    try:
        with open(cache_file, 'w') as f:
            f.write(source_hash)
    except:
        pass  # 忽略写入错误


def _compile_module():
    """编译 CUDA 扩展模块"""
    global _compiled_module_cache
    
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    buildpath = _create_build_dir(buildpath)

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    
    # 兼容方案：ninja 特殊字符路径编译支持处理（比如中文路径）
    buildpath = chinese_path_compile_support(sources, buildpath)
    
    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )
    
    # 保存编译哈希值（如果启用缓存）
    if ENABLE_CUDA_CACHE:
        source_hash = _get_source_hash(sources)
        _save_compilation_hash(buildpath, source_hash)
        _compiled_module_cache = anti_alias_activation_cuda
        
    print(">> CUDA extension module compiled successfully")
    return anti_alias_activation_cuda


def load():
    global _compiled_module_cache
    
    # 如果禁用缓存，直接编译
    if not ENABLE_CUDA_CACHE:
        print(">> CUDA cache disabled, compiling...")
        return _compile_module()
    
    # 如果已经编译过，直接返回缓存的模块
    if _compiled_module_cache is not None:
        return _compiled_module_cache
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    buildpath = _create_build_dir(buildpath)

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    
    # 检查是否需要重新编译
    should_recompile, source_hash = _should_recompile(sources, buildpath)
    
    # 检查编译后的库文件是否存在
    import glob
    compiled_files = glob.glob(str(buildpath / "*anti_alias_activation_cuda*"))
    library_exists = len(compiled_files) > 0
    
    if not should_recompile and library_exists:
        print(">> Using cached CUDA kernel for BigVGAN (skipping compilation)")
        try:
            # 尝试直接加载已编译的模块
            import importlib.util
            import sys
            
            # 查找编译后的 .so/.dll 文件
            for compiled_file in compiled_files:
                if compiled_file.endswith(('.so', '.dll', '.pyd')):
                    module_name = "anti_alias_activation_cuda"
                    spec = importlib.util.spec_from_file_location(module_name, compiled_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        _compiled_module_cache = module
                        return module
        except Exception as e:
            print(f">> Failed to load cached module, will recompile: {e}")
            should_recompile = True

    # 如果需要重新编译或库文件不存在，则编译
    if should_recompile or not library_exists:
        print(">> Building CUDA extension module for BigVGAN...")
        return _compile_module()
    
    return _compiled_module_cache


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    """创建构建目录，处理权限问题"""
    try:
        os.makedirs(buildpath, exist_ok=True)
    except OSError as e:
        print(f"创建构建目录 {buildpath} 失败: {e}")
        # 尝试使用临时目录
        import tempfile
        temp_build_dir = os.path.join(tempfile.gettempdir(), "BigVGAN_build")
        try:
            os.makedirs(temp_build_dir, exist_ok=True)
            print(f"使用临时构建目录: {temp_build_dir}")
            return pathlib.Path(temp_build_dir)
        except OSError:
            raise RuntimeError(f"无法创建构建目录，请检查权限: {buildpath}")
    return buildpath
