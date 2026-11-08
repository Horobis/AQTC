# AQTC

## 中文

AQTC 是一个用于压缩多模态大语言模型视频视觉 token 的方法。本仓库包含 AQTC 代码，以及一个基于本地 LLaVA-Video 模型和视频帧目录运行的演示脚本 `llava_demo.py`。

### 环境要求

- Linux
- NVIDIA GPU 和可用的 CUDA 驱动
- Conda
- 已下载到本机的 LLaVA-Video-7B-Qwen2 模型文件
- 视频抽帧目录，包含以数字命名的 `.jpg` 文件，例如 `0.jpg`、`1.jpg`

`environment.yml` 导出自 Linux/CUDA 环境，包含固定版本的 Python、PyTorch、Transformers 及相关依赖，不是 Windows 环境文件。分享或复现前，请先打开该文件，删除末尾机器专属的 `prefix:` 行；它指向导出环境所在服务器的绝对路径。然后在仓库根目录创建并启用环境：

```bash
conda env create -f environment.yml
conda activate aqtc
```

如果环境已经创建过，更新环境可运行：

```bash
conda env update -n aqtc -f environment.yml --prune
```

### 配置并运行演示

运行前，编辑 `llava_demo.py` 中的路径和设备设置：

```python
pretrained = "/path/to/LLaVA-Video-7B-Qwen2"
video_dir = Path("/path/to/video_frames")
device = torch.device("cuda:0")
device_map = "cuda:1"
```

- `pretrained`：本地 LLaVA-Video 模型目录，目录中应有模型配置、权重和 tokenizer 文件。
- `video_dir`：视频帧 `.jpg` 文件目录。脚本将按文件名中的数字排序，并最多读取 `max_frames` 帧。
- `device` 和 `device_map`：按服务器可见的 GPU 编号设置。示例使用两张 GPU；如果服务器只分配了一张 GPU，应根据显存情况将模型放置设置调整为该 GPU（通常为 `cuda:0`）。脚本当前设置的 `cuda:6` 和 `cuda:7` 要求相应编号的 GPU 对进程可见。

确认以上路径有效、GPU 可用后，在仓库根目录运行：

```bash
python llava_demo.py
```

脚本会打印模型加载位置、AQTC 视觉 token 压缩统计、峰值显存（若 CUDA 可用）和模型回答。脚本默认开启 Hugging Face 离线模式，因此不会自动下载模型；请提前准备好完整的本地模型文件。

演示问题、抽帧数量和 AQTC 参数可在 `llava_demo.py` 顶部修改，包括 `desc`、`max_frames`、`keep_ratio`、`memory_capacity` 及 novelty/dynamic budget 参数。

### 注意事项

- `environment.yml` 记录的是导出机器上的完整依赖快照。不同 Linux/CUDA 驱动或 GPU 环境可能需要调整 PyTorch/CUDA 相关依赖。
- 本演示依赖与导出环境相符的 PyTorch、Transformers、Accelerate、NumPy 和 Pillow 等包。若环境无法创建，请检查 Conda/Pip 安装错误及机器 CUDA 驱动兼容性。
- 模型权重和数据集通常不随代码仓库分发；请遵循各自许可，并自行准备文件。

## English

AQTC is a method for compressing visual tokens from videos for multimodal large language models. This repository includes AQTC code and `llava_demo.py`, a demo that uses a local LLaVA-Video model and a directory of extracted video frames.

### Requirements

- Linux
- An NVIDIA GPU with a working CUDA driver
- Conda
- A local copy of the LLaVA-Video-7B-Qwen2 model files
- A video-frame directory containing numerically named `.jpg` files, such as `0.jpg` and `1.jpg`

`environment.yml` was exported from a Linux/CUDA environment. It pins Python, PyTorch, Transformers, and related packages, and is not a Windows environment file. Before sharing or reproducing the environment, open the file and remove the machine-specific `prefix:` line at the end; it points to an absolute path on the exporting server. Then create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate aqtc
```

To update an existing environment:

```bash
conda env update -n aqtc -f environment.yml --prune
```

### Configure and run the demo

Before running the demo, edit the paths and device settings in `llava_demo.py`:

```python
pretrained = "/path/to/LLaVA-Video-7B-Qwen2"
video_dir = Path("/path/to/video_frames")
device = torch.device("cuda:0")
device_map = "cuda:1"
```

- `pretrained`: local LLaVA-Video model directory containing the model configuration, weights, and tokenizer files.
- `video_dir`: directory containing the video-frame `.jpg` files. The script sorts filenames numerically and reads up to `max_frames` frames.
- `device` and `device_map`: set these to GPU indices visible to the process. The example uses two GPUs. On a one-GPU server, adjust model placement to the available GPU (usually `cuda:0`) as appropriate for its memory. The script's current `cuda:6` and `cuda:7` settings require GPUs with those indices to be visible.

After verifying the paths and GPU availability, run this command from the repository root:

```bash
python llava_demo.py
```

The script prints the model-loading location, AQTC visual-token compression statistics, peak GPU memory (when CUDA is available), and the model response. Hugging Face offline mode is enabled by default, so the model is not downloaded automatically; prepare the complete local model files beforehand.

You can change the demo question, number of sampled frames, and AQTC parameters near the top of `llava_demo.py`, including `desc`, `max_frames`, `keep_ratio`, `memory_capacity`, and the novelty/dynamic-budget parameters.

### Notes

- `environment.yml` is a full dependency snapshot from the exporting machine. PyTorch/CUDA packages may need adjustment for a different Linux/CUDA driver or GPU setup.
- The demo relies on packages including PyTorch, Transformers, Accelerate, NumPy, and Pillow as pinned in the exported environment. If environment creation fails, inspect the Conda/Pip error and verify compatibility with the machine's CUDA driver.
- Model weights and datasets are generally not distributed with the code repository. Prepare them separately and follow their respective licenses.
