from inspect import unwrap
import sys
from configparser import ConfigParser
import os
import glob
import time
import PIL.Image
import numpy as np
from scipy import signal, optimize
from scipy.signal import windows  # 福原が追加
from PIL import Image
import matplotlib.pyplot as plt
import math
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import cv2
    __HAS_OPENCV__ = True
except ImportError:
    __HAS_OPENCV__ = False


def get_phase(image, config, parameters=None):
    """
    retrieve phase from image.
    image: 2d array or 3d array for batch conversion
    """
    image = image - np.mean(image, axis=(-2, -1), keepdims=True)
    x = np.arange(image.shape[-2])[:, np.newaxis]
    y = np.arange(image.shape[-1])

    if parameters is None:
        if image.ndim == 3:
            raise ValueError('For the parameter estimation, use 2d array')
        fft = np.fft.rfft2(image)
        kx = np.fft.fftfreq(image.shape[0])
        ky = np.fft.rfftfreq(image.shape[1])

        # ------
        # find the modulation frequency 
        # ------
        # make some low frequency component zero
        fft = np.where(
            (np.abs(kx)[:, np.newaxis] < 0.1) * (np.abs(ky) < 0.1),
            0, np.abs(fft))
        # maximum of the fourier transform
        idx = np.unravel_index(np.argmax(fft), shape=fft.shape)
        kx_max = kx[idx[0]]
        ky_max = ky[idx[1]]

        roi_fraction = float(config['parameters']['roi_fraction'])
        sl_x = slice(
            int(image.shape[0] * (0.5 - 0.5 * roi_fraction)),
            int(image.shape[0] * (0.5 + 0.5 * roi_fraction))
        )
        sl_y = slice(
            int(image.shape[1] * (0.5 - 0.5 * roi_fraction)),
            int(image.shape[1] * (0.5 + 0.5 * roi_fraction))
        )
        
        parameters = {}

        # maximize the modulation frequency
        def func(p):
            kx, ky = p
            wave = np.exp(2j * np.pi * (kx * x + ky * y))
            return -np.mean(np.abs(image * wave)[sl_x, sl_y])
        
        optimize_method = config['parameters']['optimize_method'].strip()
        if optimize_method != 'none':
            result = optimize.minimize(
                func, x0=(kx_max, ky_max),
                method=optimize_method)
            kx_max, ky_max = result.x
        parameters['kx'] = kx_max
        parameters['ky'] = ky_max
    
    kx_max, ky_max = parameters['kx'], parameters['ky']
    wave = np.exp(2j * np.pi * (kx_max * x + ky_max * y))
    
    # convolute the window function
    n_waves = float(config['parameters']['n_waves'])
    n = int(n_waves / np.sqrt(kx_max**2 + ky_max**2))

    # defining weight
    window = getattr(windows, config['parameters']['window'])(n)
    window2d = window[:, np.newaxis] * window
    weight = np.sum(window2d) / n**2
    if image.ndim == 3:
        window = window[np.newaxis]

    convolved = image * wave
    # along x-direction
    convolved = signal.convolve(
        convolved, window[..., np.newaxis], mode='same', method='auto')
    # along y-direction
    convolved = signal.convolve(
        convolved, window[..., np.newaxis, :], mode='same', method='auto')
    
    return parameters, convolved / weight


def get_phase_from_reference(image, source):
    """
    Retrieve phase from an image with the aid of the source image.
    """
    return np.angle(image * np.exp(-1j * np.angle(source)))


def unwrap_phase(image):
    from skimage.restoration import unwrap_phase
    return unwrap_phase(image)


def save_array(array, src_path, suffix, image_format='bmp'):
    if isinstance(image_format, (list, tuple)):
        for img_format in image_format:
            save_array(array, src_path, suffix, image_format=img_format)
        return
        
    outpath = src_path[:src_path.rfind('.')] + suffix + '.' + image_format
    if image_format in ['bmp', 'png']:
        # squish array into 0-255 range
        vmin = np.min(array)
        vmax = np.max(array)
        image = PIL.Image.fromarray(
            ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        )
        image.save(outpath, format=image_format)
    elif image_format == 'csv':
        np.savetxt(outpath, array, fmt='%10.5f', delimiter=',')

def show_scrollable_phases(phase_list, cols=3, image_size=3):
    num_images = len(phase_list)
    rows = math.ceil(num_images / cols)

    # Tkinter ウィンドウ作成
    root = tk.Tk()
    root.title("Scrollable Phase Viewer")

    # スクロール可能キャンバスを作成
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # フレームをキャンバス上に作成
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Figure 作成
    fig, axes = plt.subplots(rows, cols, figsize=(cols * image_size, rows * image_size))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (crop_px, phase_img) in enumerate(phase_list):
        ax = axes[i]
        vmin = np.min(phase_img)
        vmax = np.max(phase_img)
        image = PIL.Image.fromarray(
            ((phase_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        )
        ax.imshow(image, cmap="gray")
        ax.set_title(f"crop={crop_px}")
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # Figure を Tkinter 上に埋め込む
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

    # スクロール範囲を設定
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError(
            'Usage: get_phase.py [path/to/reference_image] [path/to/target_image] [crop_min] [crop_max]'
        )

    reference_path = sys.argv[1]
    target_path = sys.argv[2]
    crop_min = int(sys.argv[3])
    crop_max = int(sys.argv[4])

    # 設定ファイル読み込み
    config = ConfigParser()
    config.read('settings.ini')
    image_format = config['output']['format'].strip()
    if ',' in image_format:
        image_format = [img_format.strip() for img_format in image_format.split(',')]

    ref_img = np.array(PIL.Image.open(reference_path))
    tgt_img = np.array(PIL.Image.open(target_path))

    # サイズ取得（ここを修正）
    height = ref_img.shape[0]

    # 保存用の phase 画像リスト
    phase_list = []

    # 一連のループ
    for crop_pixels in range(crop_min, crop_max + 1):
        new_height = height - crop_pixels

        # NumPy配列のみでクロップ（下からcrop_pixelsだけ削る）
        ref_cropped_np = ref_img[:new_height, :]
        tgt_cropped_np = tgt_img[:new_height, :]

        parameters, ref_conv = get_phase(ref_cropped_np, config, parameters=None)
        _, tgt_conv = get_phase(tgt_cropped_np, config, parameters=parameters)

        tgt_amplitude = np.abs(tgt_conv)
        tgt_phase = get_phase_from_reference(tgt_conv, ref_conv)

        if config['postprocess']['unwrap'] == 'True':
            tgt_phase = np.clip(
                unwrap_phase(tgt_phase),
                float(config['postprocess']['unwrap_phase_min']),
                float(config['postprocess']['unwrap_phase_max'])
            )

        # phase画像を蓄積
        phase_list.append((crop_pixels, tgt_phase.copy()))

    show_scrollable_phases(phase_list, cols=3, image_size=4)