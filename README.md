
<div align="center">
    <br> QualityScaler-PySide6 image/video AI upscaler app (BSRGAN) <br><br>
    <a href="https://github.com/CrossDarkrix/QualityScaler-PySide6/releases">
	Download
    </a>
	<img width="316" height="333" alt="Preview" src="https://github.com/user-attachments/assets/afb89119-fc6f-4df4-a008-8452c5569632" />
</div>

## What do you think of this repository.
this repository is clone to [Djdefrag/QualityScaler](https://github.com/Djdefrag/QualityScaler). Rewritten PySide6.

## Credits.
BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

## Requirements.
- Windows 11 / Windows 10
- RAM >= 8Gb
- Directx12 compatible GPU


## How to install manually.
```sh
git clone https://github.com/CrossDarkrix/QualityScaler-PySide6.git
cd QualityScaler-PySide6
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python QualityScaler.py
```


## Features.
- [x] Easy to use GUI
- [x] Images and Videos upscale
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Multiple Gpu support
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## How is made.
QualityScaler is completely written in Python, from backend to frontend. 
External packages are:
- AI  -> torch / torch-directml
- GUI -> customtkinter / win32mica
- Image/video -> openCV / moviepy
- Packaging   -> Nuitka / upx
