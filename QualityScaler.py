import concurrent.futures
import functools
import io
import itertools
import os
import pathlib
import platform
import shutil
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from math import sqrt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from PIL import Image
from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QSize, Qt, Signal, Slot, QThread)
from PySide6.QtGui import (QFont, QIcon,
                           QImage, QStandardItemModel,
                           QPixmap, QStandardItem)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit, QListView, QPushButton, QAbstractItemView,
                               QMainWindow, QFileDialog, QMessageBox)
from moviepy import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from win32mica import MicaTheme, ApplyMica, MicaStyle

app_name  = "QualityScaler"
version   = "2.2"

githubme             = "https://github.com/Djdefrag/QualityScaler"
itchme               = "https://jangystudio.itch.io/qualityscaler"

half_precision       = True
AI_models_list       = [ 'BSRGANx4', 'BSRGANx2', 'BSRNetx4',
                         'RealSR_JPEGx4', 'RealSR_DPEDx4',
                         'RRDBx4', 'ESRGANx4',
                         'FSSR_JPEGx4', 'FSSR_DPEDx4' ]

file_extension_list  = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
supported_video_extensions  = ['.mp4', '.MP4',
                                '.webm', '.WEBM',
                                '.mkv', '.MKV',
                                '.flv', '.FLV',
                                '.gif', '.GIF',
                                '.m4v', ',M4V',
                                '.avi', '.AVI',
                                '.mov', '.MOV',
                                '.qt',
                                '.3gp', '.mpg', '.mpeg']
device_list_names    = []
vram_multiplier      = 1
multiplier_num_tiles = 4
windows_subversion   = int(platform.version().split('.')[2])
gpus_found           = torch_directml.device_count()
resize_algorithm     = cv2.INTER_AREA


class initAIModel(object):
    def __init__(self):
        self.current_path = os.getcwd()
        self.url = 'https://github.com/CrossDarkrix/QualityScaler-PySide6/releases/download/2.2/AIModel.zip'
        self.url2 = 'https://github.com/CrossDarkrix/QualityScaler-PySide6/releases/download/2.2/AIModel2.zip'
        self.urls = [self.url, self.url2]
        self.user_agent = 'Mozilla/5.0 (Linux; U; Android 8.0; en-la; Nexus Build/JPG991) AppleWebKit/511.2 (KHTML, like Gecko) Version/5.0 Mobile/11S444 YJApp-ANDROID jp.co.yahoo.android.yjtop/4.01.1.5'
        os.makedirs(os.path.join(os.getcwd(), 'AI'), exist_ok=True)

    def start_Download(self):
        def _unzip(download_file):
            with zipfile.ZipFile(io.BytesIO(download_file)) as z:
                z.extractall(os.path.join(self.current_path, 'AI'))
        os.chdir(os.path.join(self.current_path, 'AI'))
        [_unzip(download_file) for download_file in [urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': self.user_agent})).read() for url in self.urls]]
        os.makedirs(os.path.join(self.current_path, 'AI', 'done'), exist_ok=True)
        os.chdir(self.current_path)

def AI_enhance(model, img, backend, half_precision):
    img = img.astype(np.float32)

    if np.max(img) > 256:
        max_range = 65535  # 16 bit images
    else:
        max_range = 255

    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- process image (without the alpha channel) ------------------- #

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    if half_precision:
        img = img.unsqueeze(0).half().to(backend, non_blocking=True)
    else:
        img = img.unsqueeze(0).to(backend, non_blocking=True)

    output = model(img)

    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    if img_mode == 'L':  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #

    if img_mode == 'RGBA':
        alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
        if half_precision:
            alpha = alpha.unsqueeze(0).half().to(backend, non_blocking=True)
        else:
            alpha = alpha.unsqueeze(0).to(backend, non_blocking=True)

        output_alpha = model(alpha)  ## model

        output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535:
        output = (output_img * 65535.0).round().astype(np.uint16)  # 16-bit image
    else:
        output = (output_img * 255.0).round().astype(np.uint8)

    return output

def split_image(image_path,
                rows, cols,
                should_cleanup,
                output_dir=None):
    im = Image.open(image_path)
    im_width, im_height = im.size
    row_width = int(im_width / cols)
    row_height = int(im_height / rows)
    name, ext = os.path.splitext(image_path)
    name = os.path.basename(name)

    if output_dir != None:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
    else:
        output_dir = "./"

    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            box = (j * row_width, i * row_height, j * row_width +
                   row_width, i * row_height + row_height)
            outp = im.crop(box)
            outp_path = name + "_" + str(n) + ext
            outp_path = os.path.join(output_dir, outp_path)
            outp.save(outp_path)
            n += 1

    if should_cleanup: os.remove(image_path)

def get_tiles_paths_after_split(original_image, rows, cols):
    number_of_tiles = rows * cols

    tiles_paths = []
    for index in range(number_of_tiles):
        tile_path      = os.path.splitext(original_image)[0]
        tile_extension = os.path.splitext(original_image)[1]

        tile_path = tile_path + "_" + str(index) + tile_extension
        tiles_paths.append(tile_path)

    return tiles_paths

def split_frames_list_in_tiles(frame_list, n_tiles, cpu_number):
    list_of_tiles_list = []  # list of list of tiles, to rejoin
    tiles_to_upscale = []  # list of all tiles to upscale

    frame_directory_path = os.path.dirname(os.path.abspath(frame_list[0]))

    with ThreadPool(cpu_number) as pool:
        pool.starmap(split_image, zip(frame_list,
                                      itertools.repeat(n_tiles),
                                      itertools.repeat(n_tiles),
                                      itertools.repeat(False),
                                      itertools.repeat(frame_directory_path)))

    for frame in frame_list:
        tiles_list = get_tiles_paths_after_split(frame, n_tiles, n_tiles)
        list_of_tiles_list.append(tiles_list)
        for tile in tiles_list: tiles_to_upscale.append(tile)

    return tiles_to_upscale, list_of_tiles_list

def prepare_output_image_filename(image_path, selected_AI_model, resize_factor, selected_output_file_extension):
    # remove extension
    result_path = os.path.splitext(image_path)[0]

    resize_percentage = str(int(resize_factor * 100)) + "%"
    to_append = "_"  + selected_AI_model + "_" + resize_percentage + selected_output_file_extension

    if "_resized" in result_path:
        result_path = result_path.replace("_resized", "")
        result_path = result_path + to_append
    else:
        result_path = result_path + to_append

    return result_path


def check_if_file_is_video(file):
    for video_extension in supported_video_extensions:
        if video_extension in file:
            return True
        else:
            return False

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def image_read(image_to_prepare, flags = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(image_to_prepare, dtype=np.uint8), flags)

def image_write(path, image_data):
    _, file_extension = os.path.splitext(path)
    r, buff = cv2.imencode(file_extension, image_data)
    buff.tofile(path)

def extract_video_info(video_file):
    cap = cv2.VideoCapture(video_file)
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = num_frames / frame_rate
    minutes = int(duration / 60)
    seconds = duration % 60
    video_name = str(video_file.split("/")[-1])

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False: break
        image_write("temp.jpg", frame)
        break
    cap.release()
    video_label = ("VIDEO" + " | " + video_name + " | " + str(width) + "x"
                   + str(height) + " | " + str(minutes) + 'm:'
                   + str(round(seconds)) + "s | " + str(num_frames)
                   + "frames | " + str(round(frame_rate)) + "fps")

    img_icon = QIcon(QPixmap(QImage("temp.jpg").scaled(QSize(25, 25), aspectMode=Qt.AspectRatioMode.KeepAspectRatio, mode=Qt.TransformationMode.SmoothTransformation)))

    return video_label, img_icon

def image_info(image_file):
    image_name = '{}'.format(image_file.split('/')[-1])

    image = image_read(image_file, cv2.IMREAD_UNCHANGED)
    width = int(image.shape[1])
    height = int(image.shape[0])
    image_label = ("IMAGE" + " | " + image_name + " | " + str(width) + "x" + str(height))


    img_icon = QIcon(QPixmap(QImage(image_file).scaled(QSize(25, 25), aspectMode=Qt.AspectRatioMode.KeepAspectRatio, mode=Qt.TransformationMode.SmoothTransformation)))

    return image_label, img_icon

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir, mode=0o777)

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    img_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(img_sequence,
                                                    verbose=False,
                                                    logger=None,
                                                    fps=frame_rate)

    # extract audio
    try:
        video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3",
                                    verbose=False,
                                    logger=None)
    except Exception:
        pass

    return video_frames_list


def resize_frame(image_path, new_width, new_height, target_file_extension):
    new_image_path = image_path.replace('.jpg', "" + target_file_extension)

    old_image = cv2.imread(image_path.strip(), cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(old_image, (new_width, new_height),
                               interpolation=resize_algorithm)
    image_write(new_image_path, resized_image)

def resize_frame_list(image_list, resize_factor, target_file_extension, cpu_number):
    downscaled_images = []

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)

    with ThreadPool(cpu_number) as pool:
        pool.starmap(resize_frame, zip(image_list,
                                       itertools.repeat(new_width),
                                       itertools.repeat(new_height),
                                       itertools.repeat(target_file_extension)))

    for image in image_list:
        resized_image_path = image.replace('.jpg', "" + target_file_extension)
        downscaled_images.append(resized_image_path)

    return downscaled_images


class Gpu:
    def __init__(self, index, name):
        self.name   = name
        self.index  = index


def reverse_split(paths_to_merge,
                  rows,
                  cols,
                  image_path,
                  should_cleanup):
    images_to_merge = [Image.open(p) for p in paths_to_merge]
    image1 = images_to_merge[0]
    new_width = image1.size[0] * cols
    new_height = image1.size[1] * rows
    new_image = Image.new(image1.mode, (new_width, new_height))

    for i in range(0, rows):
        for j in range(0, cols):
            image = images_to_merge[i * cols + j]
            new_image.paste(image, (j * image.size[0], i * image.size[1]))
    new_image.save(image_path)

    if should_cleanup:
        for p in paths_to_merge:
            os.remove(p)

def reverse_split_multiple_frames(list_of_tiles_list,
                                  frames_upscaled_list,
                                  num_tiles,
                                  cpu_number):
    with ThreadPool(cpu_number) as pool:
        pool.starmap(reverse_split, zip(list_of_tiles_list,
                                        itertools.repeat(num_tiles),
                                        itertools.repeat(num_tiles),
                                        frames_upscaled_list,
                                        itertools.repeat(False)))


def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def prepare_model(selected_AI_model, backend, half_precision, upscale_factor):
    model_path = find_by_relative_path("AI" + os.sep + selected_AI_model + ".pth")

    model = BSRGAN_Net(in_nc=3,
                       out_nc=3,
                       nf=64,
                       nb=23,
                       gc=32,
                       sf=upscale_factor)
    _model_path = pathlib.Path('{}'.format(model_path))
    loadnet = torch.load(_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet, strict=True)
    model.eval()

    model.zero_grad(set_to_none=True)

    if half_precision: model = model.half()
    model = model.to(backend, non_blocking=True)

    return model

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class BSRGAN_Net(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(BSRGAN_Net, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4: self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4: fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class QualityScaler_ListView(QListView):
    add_itemd = Signal()
    def __init__(self, parent=None):
        super(QualityScaler_ListView, self).__init__(parent)
        self.files_path = []
        self.MODELs = QStandardItemModel()
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

    def _openFileSource(self):
        self.files_path, _ = QFileDialog.getOpenFileNames(None, caption="Open File", dir=os.path.expanduser('~'), filter='Files(*.gif *.png *.jpg *.jpeg *.tif *.bmp *.webp *.mp4 *.webm *.mkv *.flv *.avi *.mov *.mpg *.qt *.3gp)')
        if len(self.files_path) != 0:
            for file in self.files_path:
                threading.Thread(target=self._setMODEL, daemon=True, args=(file, )).start()
            self.setModel(self.MODELs)

    def _setMODEL(self, file):
        if check_if_file_is_video(file):
            # Video
            video_label, icon = extract_video_info(file)
            text = QStandardItem(video_label)
            text.setData(icon, Qt.ItemDataRole.DecorationRole)
            remove_file("temp.jpg")
        else:
            # Image
            image_label, icon = image_info(file)
            text = QStandardItem(image_label)
            text.setData(icon, Qt.ItemDataRole.DecorationRole)
        self.MODELs.appendRow(text)

def prepare_output_video_filename(video_path, selected_AI_model, resize_factor):
    result_video_path = os.path.splitext(video_path)[0] # remove extension

    resize_percentage = str(int(resize_factor * 100)) + "%"
    to_append = "_"  + selected_AI_model + "_" + resize_percentage + ".mp4"
    result_video_path = result_video_path + to_append

    return result_video_path

def video_need_tiles(frame, tiles_resolution):
    img_tmp             = image_read(frame)
    image_pixels        = (img_tmp.shape[1] * img_tmp.shape[0])
    tile_pixels         = (tiles_resolution * tiles_resolution)

    n_tiles = image_pixels/tile_pixels

    if n_tiles <= 1:
        return False, 0
    else:
        if (n_tiles % 2) != 0: n_tiles += 1
        n_tiles = round(sqrt(n_tiles * multiplier_num_tiles))

        return True, n_tiles

def video_reconstruction_by_frames(input_video_path, frames_upscaled_list,
                                   selected_AI_model, resize_factor, cpu_number):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    upscaled_video_path = prepare_output_video_filename(input_video_path, selected_AI_model, resize_factor)
    audio_file = app_name + "_temp" + os.sep + "audio.mp3"

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os.path.exists(audio_file):
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            audio   = audio_file,
                            verbose = False,
                            logger  = None,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                             fps     = frame_rate,
                             verbose = False,
                             logger  = None,
                             threads = cpu_number)

def show_error(exception):
    def _alert(exception):
        QMessageBox.critical(None, 'Error', 'Upscale failed caused by:\n\n {}\n\nPlease report the error on Github.com or Itch.io.\n\nThank you :)'.format(exception))
    threading.Thread(target=_alert, daemon=True, args=(exception, )).start()

def resize_image(image_path, resize_factor, selected_output_file_extension):
    new_image_path = (os.path.splitext(image_path)[0]
                      + "_resized"
                      + selected_output_file_extension)

    old_image  = image_read(image_path, cv2.IMREAD_UNCHANGED)
    new_width  = int(old_image.shape[1] * resize_factor)
    new_height = int(old_image.shape[0] * resize_factor)

    resized_image = cv2.resize(old_image, (new_width, new_height), interpolation = resize_algorithm)
    image_write(new_image_path, resized_image)
    return new_image_path

class Upscale(QThread):
    text = Signal(str)
    work_end_text = Signal(str)
    def __init__(self, cpu_of_number_count,
                 selected_AI_Model,
                 files_path,
                 selected_output_Format,
                 vram_number,
                 target_resize,
                 parent=None):
        super().__init__(parent)
        self.cpu_of_number_count = cpu_of_number_count
        self.selected_AI_Model = selected_AI_Model
        self.upscale_factor = None
        self.files_path = files_path
        self.selected_output_Format = selected_output_Format
        self.vram_number = vram_number
        self.target_resize = target_resize
        self.backend = None
        self.task = None
        self.task_signal = False

    def upscale_orchestrator(self):
        start = timer()
        torch.set_num_threads(self.cpu_of_number_count)
        if self.backend is not None and self.upscale_factor is not None:
            try:
                AI_model = prepare_model(self.selected_AI_Model, self.backend, half_precision, self.upscale_factor)
                for index in range(len(self.files_path)):
                    if self.task_signal:
                        break
                    self.text.emit("Upscaling " + str(index + 1) + "/" + str(len(self.files_path)))
                    file_path = self.files_path[index]
                    file_path = file_path.replace(os.path.dirname(self.files_path[index]), '{}_upscaled'.format(os.path.dirname(self.files_path[index])))
                    os.makedirs('{}_upscaled'.format(os.path.dirname(self.files_path[index])), exist_ok=True)
                    shutil.copy(self.files_path[index], '{}_upscaled'.format(os.path.dirname(self.files_path[index])))
                    if check_if_file_is_video(file_path):
                        self.upscale_video(file_path,
                                      AI_model,
                                      self.selected_AI_Model,
                                      self.upscale_factor,
                                      self.backend,
                                      self.selected_output_Format,
                                      self.vram_number,
                                      self.target_resize,
                                      self.cpu_of_number_count,
                                      half_precision)
                    else:
                        self.upscale_image(file_path,
                                      AI_model,
                                      self.selected_AI_Model,
                                      self.upscale_factor,
                                      self.backend,
                                      self.selected_output_Format,
                                      self.vram_number,
                                      self.target_resize,
                                      self.cpu_of_number_count,
                                      half_precision)
                    try:
                        os.remove(file_path)
                    except:
                        pass

                self.text.emit("All files completed (" + str(round(timer() - start)) + " sec.)")
                self.work_end_text.emit("UPSCALE")
                time.sleep(4)
                self.text.emit('Hi :)')
            except Exception:
                self.text.emit("All files completed (" + str(round(timer() - start)) + " sec.)")
                self.work_end_text.emit("UPSCALE")
                time.sleep(4)
                self.text.emit('Hi :)')

    def upscale_image(self, image_path,
                      AI_model,
                      selected_AI_model,
                      upscale_factor,
                      backend,
                      selected_output_file_extension,
                      tiles_resolution,
                      resize_factor,
                      cpu_number,
                      half_precision):

        # if image need resize before AI work
        if resize_factor != 1:
            image_path = resize_image(image_path,
                                      resize_factor,
                                      selected_output_file_extension)

        result_path = prepare_output_image_filename(image_path, selected_AI_model, resize_factor,
                                                    selected_output_file_extension)
        self.upscale_image_and_save(image_path,
                               AI_model,
                               result_path,
                               tiles_resolution,
                               upscale_factor,
                               backend,
                               half_precision)

        # if the image was sized before the AI work
        if resize_factor != 1: remove_file(image_path)

    def upscale_image_and_save(self, image,
                               AI_model,
                               result_path,
                               tiles_resolution,
                               upscale_factor,
                               backend,
                               half_precision):

        need_tiles, n_tiles = self.image_need_tiles(image, tiles_resolution)
        if need_tiles:
            split_image(image_path=image,
                        rows=n_tiles,
                        cols=n_tiles,
                        should_cleanup=False,
                        output_dir=os.path.dirname(os.path.abspath(image)))

            tiles_list = get_tiles_paths_after_split(image, n_tiles, n_tiles)

            with torch.no_grad():
                for tile in tiles_list:
                    tile_adapted = image_read(tile, cv2.IMREAD_UNCHANGED)
                    tile_upscaled = AI_enhance(AI_model, tile_adapted, backend, half_precision)
                    image_write(tile, tile_upscaled)

            reverse_split(paths_to_merge=tiles_list,
                          rows=n_tiles,
                          cols=n_tiles,
                          image_path=result_path,
                          should_cleanup=False)

            self.delete_list_of_files(tiles_list)
        else:
            with torch.no_grad():
                img_adapted = image_read(image, cv2.IMREAD_UNCHANGED)
                img_upscaled = AI_enhance(AI_model, img_adapted, backend, half_precision)
                image_write(result_path, img_upscaled)

    def image_need_tiles(self, image, tiles_resolution):
        img_tmp = image_read(image)
        image_pixels = (img_tmp.shape[1] * img_tmp.shape[0])
        tile_pixels = (tiles_resolution * tiles_resolution)

        n_tiles = image_pixels / tile_pixels

        if n_tiles <= 1:
            return False, 0
        else:
            if (n_tiles % 2) != 0: n_tiles += 1
            n_tiles = round(sqrt(n_tiles * multiplier_num_tiles))

            return True, n_tiles

    def delete_list_of_files(self, list_to_delete):
        if len(list_to_delete) > 0:
            for to_delete in list_to_delete:
                if os.path.exists(to_delete):
                    os.remove(to_delete)

    def upscale_video(self, video_path,
                      AI_model,
                      selected_AI_model,
                      upscale_factor,
                      backend,
                      selected_output_file_extension,
                      tiles_resolution,
                      resize_factor,
                      cpu_number,
                      half_precision):

        create_temp_dir(app_name + "_temp")

        self.text.emit("Extracting video frames")
        frame_list = extract_frames_from_video(video_path)

        if resize_factor != 1:
            self.text.emit("Resizing video frames")
            frame_list = resize_frame_list(frame_list,
                                           resize_factor,
                                           selected_output_file_extension,
                                           cpu_number)

        self.upscale_video_and_save(video_path,
                               frame_list,
                               AI_model,
                               tiles_resolution,
                               selected_AI_model,
                               backend,
                               resize_factor,
                               selected_output_file_extension,
                               half_precision,
                               cpu_number)

    def upscale_video_and_save(self, video_path,
                               frame_list,
                               AI_model,
                               tiles_resolution,
                               selected_AI_model,
                               backend,
                               resize_factor,
                               selected_output_file_extension,
                               half_precision,
                               cpu_number):

        self.text.emit("Upscaling video")
        frames_upscaled_list = []
        need_tiles, n_tiles = video_need_tiles(frame_list[0], tiles_resolution)

        # Prepare upscaled frames file paths
        for frame in frame_list:
            result_path = prepare_output_image_filename(frame, selected_AI_model, resize_factor,
                                                        selected_output_file_extension)
            frames_upscaled_list.append(result_path)

        if need_tiles:
            self.text.emit("Tiling frames...")
            tiles_to_upscale, list_of_tiles_list = split_frames_list_in_tiles(frame_list, n_tiles, cpu_number)
            how_many_tiles = len(tiles_to_upscale)

            for index in range(how_many_tiles):
                self.upscale_tiles(tiles_to_upscale[index],
                              AI_model,
                              backend,
                              half_precision)
                if (index % 8) == 0: self.text.emit(
                    "Upscaled tiles " + str(index + 1) + "/" + str(how_many_tiles))

            self.text.emit("Reconstructing frames by tiles...")
            reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list, n_tiles, cpu_number)

        else:
            how_many_frames = len(frame_list)

            for index in range(how_many_frames):
                self.upscale_single_frame(frame_list[index],
                                     AI_model,
                                     frames_upscaled_list[index],
                                     backend,
                                     half_precision)
                if (index % 8) == 0: self.text.emit(
                    "Upscaled frames " + str(index + 1) + "/" + str(how_many_frames))

        # Reconstruct the video with upscaled frames
        self.text.emit("Processing upscaled video")
        video_reconstruction_by_frames(video_path, frames_upscaled_list,
                                       selected_AI_model,
                                       resize_factor, cpu_number)

    def upscale_tiles(self, tile, AI_model, backend, half_precision):
        with torch.no_grad():
            tile_adapted = image_read(tile, cv2.IMREAD_UNCHANGED)
            tile_upscaled = AI_enhance(AI_model, tile_adapted, backend, half_precision)
            image_write(tile, tile_upscaled)

    def upscale_single_frame(self, frame, AI_model, result_path, backend, half_precision):
        with torch.no_grad():
            img_adapted = image_read(frame, cv2.IMREAD_UNCHANGED)
            img_upscaled = AI_enhance(AI_model, img_adapted, backend, half_precision)
            image_write(result_path, img_upscaled)

    def _init(self, backend, upscale_factor):
        self.backend = backend
        self.upscale_factor = upscale_factor

    def run(self):
        p = concurrent.futures.ThreadPoolExecutor(os.cpu_count() * 999999999999999)
        p.submit(self.upscale_orchestrator).result()


class QualityScaler(QMainWindow):
    def __init__(self):
        super(QualityScaler, self).__init__()
        icon_path = os.path.join(os.getcwd(), 'Assets', 'icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(QPixmap(QSize(96, 96)).fromImage(QImage(icon_path))))
        self.resize(589, 593)
        if not os.path.exists(os.path.join(os.getcwd(), 'AI')):
            concurrent.futures.ThreadPoolExecutor().submit(initAIModel().start_Download)
            QMessageBox.information(self, "Model Downloading....", "downloading Model data. Please Wait.....")
            while not os.path.exists(os.path.join(os.getcwd(), 'AI', 'done')):
                if os.path.exists(os.path.join(os.getcwd(), 'AI', 'done')):
                    shutil.rmtree(os.path.join(os.getcwd(), 'AI', 'done'))
                    break
                else:
                    time.sleep(1)
        self.item_list = QualityScaler_ListView(self)
        self.item_list.setGeometry(QRect(0, 0, 591, 301))
        self.clean = QPushButton(self)
        if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'clear_icon.png')):
            self.clean.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'clear_icon.png')))))
        self.clean.setObjectName(u"clean")
        self.clean.setGeometry(QRect(500, 0, 91, 31))
        self.github = QPushButton(self)
        if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'github_logo.png')):
            self.github.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'github_logo.png')))))
        self.github.clicked.connect(self.open_github)
        self.github.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.github.setObjectName(u"github")
        self.github.setGeometry(QRect(10, 310, 24, 24))
        self.itch_io = QPushButton(self)
        if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'itch_logo.png')):
            self.itch_io.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'itch_logo.png')))))
        self.itch_io.clicked.connect(self.open_itch)
        self.itch_io.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.itch_io.setObjectName(u"itch_io")
        self.itch_io.setGeometry(QRect(10, 340, 24, 24))
        self.ai_model = QPushButton(self)
        self.ai_model.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.ai_model.clicked.connect(self._ai_model_info)
        self.ai_model.setObjectName(u"ai_model")
        self.ai_model.setGeometry(QRect(60, 360, 101, 24))
        self.ai_model.setFlat(False)
        self.AI_Model_pull = QComboBox(self)
        self.AI_Model_pull.setStyleSheet("QComboBox{Background: Black;color: White;} QAbstractItemView{Background: Black;color: White;}")
        self.AI_Model_pull.setObjectName(u"AI_Model_pull")
        self.AI_Model_pull.setGeometry(QRect(60, 390, 101, 31))
        self.GPU_pull = QComboBox(self)
        self.GPU_pull.setStyleSheet("QComboBox{Background: Black;color: White;} QAbstractItemView{Background: Black;color: White;}")
        self.GPU_pull.setObjectName(u"GPU_pull")
        self.GPU_pull.setGeometry(QRect(60, 470, 101, 31))
        self.GPU_device = QPushButton(self)
        self.GPU_device.clicked.connect(self._gpu_info)
        self.GPU_device.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.GPU_device.setObjectName(u"GPU_device")
        self.GPU_device.setGeometry(QRect(60, 440, 101, 24))
        self.GPU_device.setFlat(False)
        self.output_formats = QPushButton(self)
        self.output_formats.clicked.connect(self._output_info)
        self.output_formats.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.output_formats.setObjectName(u"output_formats")
        self.output_formats.setGeometry(QRect(60, 520, 101, 24))
        self.output_formats.setFlat(False)
        self.output_pull = QComboBox(self)
        self.output_pull.setStyleSheet("QComboBox{Background: Black;color: White;} QAbstractItemView{Background: Black;color: White;}")
        self.output_pull.setObjectName(u"output_pull")
        self.output_pull.setGeometry(QRect(60, 550, 101, 31))
        self.resolution = QPushButton(self)
        self.resolution.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.resolution.setObjectName(u"resolution")
        self.resolution.clicked.connect(self._resolution_info)
        self.resolution.setGeometry(QRect(180, 360, 121, 24))
        self.resolution.setFlat(False)
        self.resolution_edit = QLineEdit(self)
        self.resolution_edit.setText('20')
        self.resolution_edit.setObjectName(u"resolution_edit")
        self.resolution_edit.setGeometry(QRect(180, 390, 121, 31))
        self.Vram_edit = QLineEdit(self)
        self.Vram_edit.setObjectName(u"Vram_edit")
        self.Vram_edit.setGeometry(QRect(180, 470, 121, 31))
        self.Vram_edit.setText('2')
        self.Vram = QPushButton(self)
        self.Vram.clicked.connect(self._VRAM_Info)
        self.Vram.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.Vram.setObjectName(u"Vram")
        self.Vram.setGeometry(QRect(180, 440, 121, 24))
        self.Vram.setFlat(False)
        self.num_CPU = QPushButton(self)
        self.num_CPU.clicked.connect(self._CPU_number_info)
        self.num_CPU.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.num_CPU.setObjectName(u"num_CPU")
        self.num_CPU.setGeometry(QRect(180, 520, 121, 24))
        self.num_CPU.setFlat(False)
        self.CPU_count = QLineEdit(self)
        self.CPU_count.setText('4')
        self.CPU_count.setObjectName(u"CPU_count")
        self.CPU_count.setGeometry(QRect(180, 550, 121, 31))
        self.work = QPushButton(self)
        if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')):
            self.work.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')))))
        self.work.clicked.connect(self._check_selected_files)
        self.work.setStyleSheet("QPushButton{Background: Black;color: Yellow;}")
        self.work.setObjectName(u"work")
        self.work.setGeometry(QRect(400, 540, 121, 31))
        self.title = QLabel(self)
        self.title.setObjectName(u"title")
        self.title.setGeometry(QRect(170, 315, 241, 31))
        self.title.setStyleSheet("QLabel{color: #e4007f}")
        font = QFont()
        font.setPointSize(20)
        self.title.setFont(font)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.work_number = QLabel(self)
        self.work_number.setStyleSheet("QLabel{background: Orange; color: Black;}")
        self.work_number.setGeometry(QRect(390, 360, 151, 40))
        self.work_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.select_files = QPushButton(self)
        self.select_files.setObjectName(u"select_files")
        self.select_files.setGeometry(QRect(210, 220, 171, 31))
        self.select_files.setStyleSheet("QPushButton{background: lightblue; color: Black;}")
        self.select_files.setText('SELECT FILES')
        self.select_files.clicked.connect(self._check_files)
        self.formats_text = QLabel(self)
        self.formats_text.setObjectName(u"formats_text")
        self.formats_text.setGeometry(QRect(137, 80, 331, 111))
        self.formats_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clean.setDisabled(True)
        self.clean.setVisible(False)
        self.clean.clicked.connect(self._clean_files)
        self.device_list = []
        self.device_list_names = []
        self.selected_GPU = ''
        self.selected_AI_Model = ''
        self.selected_output_Format = ''
        self.vram_number = 0
        self.cpu_of_number_count = 0
        self.GPU_pull.currentIndexChanged.connect(self._setGPUDev)
        for index in range(gpus_found):
            self.set_GPU(index)
        self.GPU_pull.addItems(self.device_list_names)
        self.AI_Model_pull.currentIndexChanged.connect(self._setAI)
        self.AI_Model_pull.addItems(AI_models_list)
        self.output_pull.currentIndexChanged.connect(self._setFormat)
        self.output_pull.addItems(file_extension_list)
        self.target_resize = 0
        self.process = None
        self.retranslateUi(self)

        QMetaObject.connectSlotsByName(self)
    # setupUi

    def retranslateUi(self, QualityScaler):
        QualityScaler.setWindowTitle(QCoreApplication.translate("QualityScaler", u"QualityScaler v2.2", None))
        self.clean.setText(QCoreApplication.translate("QualityScaler", "CLEAN", None))
        self.ai_model.setText(QCoreApplication.translate("QualityScaler", u"AI Model", None))
        self.GPU_device.setText(QCoreApplication.translate("QualityScaler", u"GPU", None))
        self.output_formats.setText(QCoreApplication.translate("QualityScaler", u"Output Format", None))
        self.resolution.setText(QCoreApplication.translate("QualityScaler", u"set Resolution(\uff05)", None))
        self.Vram.setText(QCoreApplication.translate("QualityScaler", u"GPU VRAM", None))
        self.num_CPU.setText(QCoreApplication.translate("QualityScaler", u"numbar of CPU", None))
        self.work.setText(QCoreApplication.translate("QualityScaler", u"UPSCALE", None))
        self.title.setText(QCoreApplication.translate("QualityScaler", u"QualityScaler v2.2", None))
        self.work_number.setText(QCoreApplication.translate("QualityScaler", u"Hi :)", None))
        self.formats_text.setText(' - SUPPORTED FILES -\n\nIMAGES - jpg png tif bmp webp\nVIDEOS - mp4 webm mkv flv gif avi mov mpg qt 3gp')
    # retranslateUi

    @Slot(str)
    def setWorkText(self, text):
        self.work_number.setText(text)

    @Slot(str)
    def setWorkEndText(self, text):
        self.work.setText(text)
        if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')):
            self.work.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')))))

    def _check_files(self, _):
        self.item_list._openFileSource()
        if len(self.item_list.files_path) != 0:
            self.select_files.setDisabled(True)
            self.select_files.setVisible(False)
            self.formats_text.setDisabled(True)
            self.formats_text.setVisible(False)
            self.clean.setDisabled(False)
            self.clean.setVisible(True)

    def _clean_files(self, _):
        self.item_list.files_path.clear()
        self.item_list.model().removeRows(0, self.item_list.model().rowCount())
        self.select_files.setDisabled(False)
        self.select_files.setVisible(True)
        self.formats_text.setDisabled(False)
        self.formats_text.setVisible(True)
        self.clean.setDisabled(True)
        self.clean.setVisible(False)

    def set_GPU(self, index):
        gpu = Gpu(index=index, name=torch_directml.device_name(index))
        self.device_list.append(gpu)
        self.device_list_names.append(gpu.name)

    def _setGPUDev(self, index):
        device_name = self.GPU_pull.itemText(index)
        for device in self.device_list:
            if device.name == device_name:
                self.selected_GPU = device.index

    def _setAI(self, index):
        self.selected_AI_Model = self.AI_Model_pull.itemText(index)

    def _setFormat(self, index):
        self.selected_output_Format = self.output_pull.itemText(index)

    def _CPU_Number(self):
        self.cpu_of_number_count = self.CPU_count.text()

    def _ai_model_info(self, _):
        QMessageBox.information(self, 'Ai Model', "This widget allows to choose between different AI: \n\n- BSRGANx2 | high upscale quality | upscale by 2.\n- BSRGANx4 | high upscale quality | upscale by 4.\n- BSRNetx4 | high upscale quality | upscale by 4.\n- RealSR_JPEGx4 | good upscale quality | upscale by 4.\n- RealSR_DPEDx4 | good upscale quality | upscale by 4.\n- RRDBx4 | good upscale quality | upscale by 4.\n- ESRGANx4 | good upscale quality | upscale by 4.\n- FSSR_JPEGx4 | good upscale quality | upscale by 4.\n- FSSR_DPEDx4 | good upscale quality | upscale by 4.\n\nTry all AI and choose the one that gives the best results")

    def _gpu_info(self, _):
        QMessageBox.information(self, 'GPU', "This widget allows to choose the gpu to run AI with. \n\nKeep in mind that the more powerful your gpu is, \nthe faster the upscale will be. \n\nIf the list is empty it means the app couldn't find \na compatible gpu, try updating your video card driver :)")

    def _output_info(self, _):
        QMessageBox.information(self, 'Output Formats', "This widget allows to choose the extension of upscaled image/frame.\n\n- png | very good quality | supports transparent images\n- jpg | good quality | very fast\n- jp2 (jpg2000) | very good quality | not very popular\n- bmp | highest quality | slow\n- tiff | highest quality | very slow")

    def _resolution_info(self, _):
        QMessageBox.information(self, 'Resolution', "This widget allows to choose the resolution input to the AI.\n\nFor example for a 100x100px image:\n- Input resolution 50% => input to AI 50x50px\n- Input resolution 100% => input to AI 100x100px\n- Input resolution 200% => input to AI 200x200px")

    def _VRAM_Info(self, _):
        QMessageBox.information(self, 'VRAM limiter GB', "This widget allows to set a limit on the gpu's VRAM memory usage. \n\n- For a gpu with 4 GB of Vram you must select 4\n- For a gpu with 6 GB of Vram you must select 6\n- For a gpu with 8 GB of Vram you must select 8\n- For integrated gpus (Intel-HD series | Vega 3,5,7) \n  that do not have dedicated memory, you must select 2 \n\nSelecting a value greater than the actual amount of gpu VRAM may result in upscale failure. ")

    def _CPU_number_info(self, _):
        QMessageBox.information(self, 'CPU number', "This widget allows you to choose how many cpus to devote to the app.\n\nWhere possible the app will use the number of processors you select, for example:\n- Extracting frames from videos\n- Resizing frames from videos\n- Recostructing final video\n- AI processing")

    def _check_selected_files(self, _):
        def _backText():
            time.sleep(3)
            self.work_number.setText("Hi :)")
        if self.work.text() == 'STOP':
            self.process.task_signal = True
            self.work.setText("UPSCALE")
            if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')):
                self.work.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'upscale_icon.png')))))
        else:
            if self.process is not None:
                if self.process.isFinished():
                    self.process = None
            if len(self.item_list.files_path) != 0 and self.process is None:
                if self._user_check_data():
                    self.work_number.setText('Loading...')
                    if "x2" in self.selected_AI_Model :
                        upscale_factor = 2
                    elif "x4" in self.selected_AI_Model :
                        upscale_factor = 4
                    backend = torch.device(torch_directml.device(int(self.selected_GPU)))
                    self.work.setText('STOP')
                    if os.path.exists(os.path.join(os.getcwd(), 'Assets', 'stop_icon.png')):
                        self.work.setIcon(QIcon(QPixmap(QSize(24, 24)).fromImage(QImage(os.path.join(os.getcwd(), 'Assets', 'stop_icon.png')))))
                    self.process = Upscale(self.cpu_of_number_count, self.selected_AI_Model, self.item_list.files_path, self.selected_output_Format, self.vram_number, self.target_resize)
                    self.process.setPriority(QThread.Priority.HighestPriority)
                    self.process.text.connect(self.setWorkText)
                    self.process.work_end_text.connect(self.setWorkEndText)
                    self.process._init(backend, upscale_factor)
                    self.process.start()
            else:
                self.work_number.setText("No File Selected!")
                threading.Thread(target=_backText, daemon=True).start()

    def _user_check_data(self):
        try:
            self.target_resize = int(float(self.resolution_edit.text()))
        except:
            return False
        if self.target_resize > 0:
            self.target_resize = self.target_resize / 100
        else:
            return False
        try:
            tiles_vram = 100 * int(float(self.Vram_edit.text()))
            if tiles_vram > 0:
                vram = int(float(self.Vram_edit.text()))
                self.vram_number = vram * 100
            else:
                return False
        except:
            return False
        if self.selected_GPU == '':
            self._setGPUDev(0)
        if self.selected_output_Format == '':
            self.selected_output_Format = self.output_pull.itemText(0)
        if self.selected_AI_Model == '':
            self.selected_AI_Model = self.AI_Model_pull.itemText(0)
        try:
            _cpu = int(float(self.CPU_count.text()))
            if _cpu <= 0:
                return False
            else:
                self.cpu_of_number_count = _cpu
        except:
            return False

        return True

    def open_github(self, _):
        webbrowser.open(githubme, new=1)

    def open_itch(self, _):
        webbrowser.open(itchme, new=1)


def main():
    app = QApplication(sys.argv)
    wind = QualityScaler()
    wind.setFixedSize(wind.size())
    ApplyMica(wind.winId(), MicaTheme.DARK, MicaStyle.DEFAULT)
    wind.show()
    app.exec()

if __name__ == '__main__':
    main()