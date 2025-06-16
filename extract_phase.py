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


def save_video(arrays, src_path, suffix, video_format='avi'):
    outpath = src_path[:src_path.rfind('.')] + suffix + '.' + video_format
    height, width = arrays.shape[1:]
    # video = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'DIVX'), 15.0, (width, height))
    #! 非圧縮(IYUV)に変更、出力のaviファイルをImageJで開くため
    #! ただし、処理に時間かかる
    video = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'IYUV'), 15.0, (width, height)) 
    for array in arrays:
        vmin = np.min(array)
        vmax = np.max(array)
        array = ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        array = np.stack([array, array, array], axis=-1)
        video.write(array)
    video.release()

#* [1]のpathが動画ファイルの場合、
#* 動画を画像に変換し1フレーム目の画像と画像すべての配列を返す関数
def _video2images(video):
    if not __HAS_OPENCV__:
        raise ImportError(
            "Open cv must be installed to read video.\n"
        )

    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read() #? 読み込み成功のブール値、読み込んだ画像データ（Numpy配列）
    count = 0
    images = []
    
    while success:
        success, image = vidcap.read()
        image = np.array(image)
        # here we just use the r channel. Maybe we need something here
        #? コメントではrチャンネル（RGB配列の0番目を取り出しているから）を使っていると書いているが、BGR配列の可能性もある。
        #? これはカラー画像をから一つのチャンネルのみを取り出すことで疑似的なグレースケールを創り出す操作をしている。
        if image.ndim == 3:
            image = image[..., 0]
        if success:
            images.append(image)

    return images[0], images


__VIDEO_EXTENSIONS__ = ['.avi', '.mp4']

if __name__ == '__main__':
    #* コマンドプロンプトからの実行時に引数がない場合は、エラーを出力
    if len(sys.argv) < 2:
        raise ValueError(
            'Usage: get_phase.py [path/to/image] '
            'or \nget_phase.py [path/to/reference_image] [path/to/target_image1] ... '
            '\n  [NOTE] target_image1 can include the wild cards.'
            '\n         e.g. step[0-9][0-9].bmp will cover step01.bmp, step02.bmp, ..., step99.bmp'
            'or \nget_phase.py [path/to/video] reference_frames(e.g. 1,2 or 1-10)'
        )
    
    config = ConfigParser()
    config.read('settings.ini')
    image_format = config['output']['format'].strip()
    video_format = config['output'].get('video_format').strip()
    if ',' in image_format:
        image_format = [img_format.strip() for img_format in image_format.split(',')]

    path = sys.argv[1]
    video_images = None
    started = time.time()
    #* sys.argv[1]が動画ファイルの場合、
    #* 動画ファイルを分解して1フレーム目の画像(image)と画像すべての配列(video_images)を取得
    if any(ex in path for ex in __VIDEO_EXTENSIONS__):  # video 
        image, video_images = _video2images(path)
    #* sys.argv[1]が動画ファイルでない場合、
    #* 入力が画像という前提すすむ
    else:
        image = PIL.Image.open(path)
    parameters, convolved = get_phase(np.array(image), config, parameters=None)
    amplitude, phase = np.abs(convolved), np.angle(convolved)

    will_unwrap = config['postprocess']['unwrap'] == 'True'

    #* 入力が画像ファイル一つだけの場合、
    #* 参照画像無しで画像の振幅と位相を計算し保存
    if len(sys.argv) == 2 and video_images is None:
        save_array(amplitude, path, '_amp', image_format=image_format)
        if will_unwrap:
            phase = unwrap_phase(phase)
        save_array(phase, path, '_phase', image_format=image_format)

    #* 入力の数は関係なく、
    #* 動画ファイルが入力された場合、
    elif video_images is not None:  # video
        image_format = config['output']['video_format'].strip()
        target_amplitudes = []
        target_phases = []

        num_batch = int(config['computation']['num_batch'])
        if num_batch < 0:  # compute full batch
            _, target_conv = get_phase(np.array(video_images), config, parameters=parameters)
        else:
            target_conv = []
            for i in range(0, len(video_images), num_batch):
                target_conv.append(get_phase(
                    np.stack(video_images[i: i + num_batch], axis=0), 
                    config, parameters=parameters)[1]
                )
            target_conv = np.concatenate(target_conv, axis=0)

        target_phases = get_phase_from_reference(target_conv, convolved)
        if will_unwrap:
            target_phases = np.clip(
                unwrap_phase(target_phases),
                float(config['postprocess']['unwrap_phase_min']),
                float(config['postprocess']['unwrap_phase_max'])
            )
        save_video(target_phases, path, '_phase', video_format='avi')

        target_amplitudes = np.abs(target_conv)
        save_video(target_amplitudes, path, '_amp', video_format='avi')

    #* 入力の数は関係なく、
    #* 画像ファイルが複数入力された場合、一枚目の画像を参照画像として計算
    else:
        paths = []
        for path in sys.argv[2:]:
            paths += glob.glob(path)
        print("converting ", paths)
            
        images = [np.array(PIL.Image.open(path)) for path in paths]
        _, target_conv = get_phase(np.array(images), config, parameters=parameters)

        target_amplitudes = np.abs(target_conv)
        target_phases = get_phase_from_reference(target_conv, convolved)
        if will_unwrap:
            target_phases = np.clip(
                unwrap_phase(target_phases),
                float(config['postprocess']['unwrap_phase_min']),
                float(config['postprocess']['unwrap_phase_max'])
            )

        for path, target_amplitude, target_phase in zip(
            paths, target_amplitudes, target_phases
        ):
            save_array(target_amplitude, path, '_amp', image_format=image_format)
            save_array(target_phase, path, '_phase', image_format=image_format)

    print(f"おわり！！！！") #?福原が追加
