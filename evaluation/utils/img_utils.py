
import cv2
import glob
import os 
from PIL import Image
import PIL


import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
 
import glob 
from os.path import join, split,exists,isfile
import argparse
from PIL import Image
import torch

def concat_images(image_paths, output_path = None, direction='horizontal'):
    """
    Concatenates multiple images and saves the result.

    Parameters:
        image_paths (list): List of image file paths.
        output_path (str): Path to save the concatenated image.
        direction (str): Direction to concatenate ('horizontal' or 'vertical').
    """
    if not image_paths or len(image_paths) < 2:
        raise ValueError("At least two images are required for concatenation.")
    
    # Open images
    
    
    if not isinstance(image_paths[0],PIL.Image.Image):
        images = [Image.open(img) for img in image_paths]
    else:
        images = image_paths



    # alpha = 0.5  
    # images.append(Image.blend(images[-2].convert("RGBA"), images[-1].convert("RGBA"), alpha))
    
    if direction == 'horizontal':
        # Calculate dimensions for the concatenated image
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Create a new blank image with the calculated dimensions
        new_image = Image.new('RGB', (total_width, max_height))
        
        # Paste images side by side
        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width
    
    elif direction == 'vertical':
        # Calculate dimensions for the concatenated image
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        # Create a new blank image with the calculated dimensions
        new_image = Image.new('RGB', (max_width, total_height))
        
        # Paste images one on top of the other
        y_offset = 0
        for img in images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height
    
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")
    
    # Save the concatenated image
    if output_path is not None:
        new_image.save(output_path)
        print(f"Concatenated image saved to {output_path}")
        
    return new_image








def video_to_frames(video_path):
    """
    Parses a video file into individual image frames.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        prefix (str): Prefix for the saved frame filenames.
        image_format (str): Image format for saved frames (e.g., 'jpg', 'png').

    Returns:
        int: Number of frames extracted.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    frame_count = 0
    imgs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        imgs.append(frame)
        frame_count += 1
    
    print(f"{frame_count} images in total")
    cap.release()
    return imgs






def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)





"""
give images in path format

"""
def imgs2video(imgs, fps = 10,out_path = 'img2video.mp4'):

    print('ready to compress %d imgs in to a video')
    h,w,c = cv2.imread(imgs[0]).shape
    frame_size = (w,h)
    fourcc=  cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

    for img in imgs:
        cur_frame = cv2.imread(img)
        if cur_frame is not None :
            video_writer.write(cur_frame)
    video_writer.release()

    print('the results video is saved at %s'%(out_path))
    

"""
give images in numpy format
"""
def imgs2video2(imgs, fps = 10,out_path = 'img2video.mp4'):

    print('ready to compress %d imgs in to a video')
    h,w,c = imgs[0].shape
    frame_size = (w,h)
    fourcc=  cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

    for img in imgs:
        if img is not None :
            video_writer.write(img)
    video_writer.release()

    print('the results video is saved at %s'%(out_path)) 



"""
given images in path format
"""
def images_to_gif(images, output_gif, duration=100, loop=0):
    """
    Compress a set of images into an animated GIF.

    :param image_folder: Path to the folder containing images.
    :param output_gif: Path for the output GIF file (e.g., 'output.gif').
    :param duration: Time per frame in milliseconds (default is 100ms per frame).
    :param loop: Number of loops (0 for infinite loop).
    """
    
    if isinstance(images[0], Image.Image):
        frames = images
    else:
        frames = [Image.open(img).convert("RGB") for img in images]
        


    # Save as GIF
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop)

    print(f"GIF saved at: {output_gif}")
    

"""
given images in PIL Image format
"""
def images_to_gif2(images, output_gif, duration=100, loop=0):
    """
    Compress a set of images into an animated GIF.

    :param image_folder: Path to the folder containing images.
    :param output_gif: Path for the output GIF file (e.g., 'output.gif').
    :param duration: Time per frame in milliseconds (default is 100ms per frame).
    :param loop: Number of loops (0 for infinite loop).
    """

    frames = [img.convert("RGB") for img in images]

    # Save as GIF
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop)

    print(f"GIF saved at: {output_gif}")





def read_exr_as_normals(filename):
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    # 获取 EXR 图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 读取 RGB 作为 Normal XYZ
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)  # 你的 EXR 采用 HALF (16-bit)
    
    normal_x = np.frombuffer(exr_file.channel('R', pixel_type), dtype=np.float16).reshape((height, width))
    normal_y = np.frombuffer(exr_file.channel('G', pixel_type), dtype=np.float16).reshape((height, width))
    normal_z = np.frombuffer(exr_file.channel('B', pixel_type), dtype=np.float16).reshape((height, width))

    # 组合成 (H, W, 3) 法线贴图
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

    return normal_map




def read_exr_as_normals(filename):
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    # 获取 EXR 图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 读取 RGB 作为 Normal XYZ
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)  # 你的 EXR 采用 HALF (16-bit)
    
    normal_x = np.frombuffer(exr_file.channel('R', pixel_type), dtype=np.float16).reshape((height, width))
    normal_y = np.frombuffer(exr_file.channel('G', pixel_type), dtype=np.float16).reshape((height, width))
    normal_z = np.frombuffer(exr_file.channel('B', pixel_type), dtype=np.float16).reshape((height, width))

    # 组合成 (H, W, 3) 法线贴图
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

    return normal_map



def normal_in_exrs_to_gif(exrs, save_path='logs/exr_normal2.gif'):
    img_bank= []
    for x in tqdm(exrs):
        normal_map = read_exr_as_normals(x)
        normalized_normal_map = (normal_map + 1) / 2
        img_bank.append(Image.fromarray((normalized_normal_map*255).astype(np.uint8)))    
    images_to_gif2(img_bank,save_path)


    

def read_exr_as_depth(filename, channel_name = 'R'):
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    # 获取 EXR 图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 读取 RGB 作为 Normal XYZ
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)  # 这里改为 FLOAT (32-bit)

    #* R, G, B, A channels are all the same 
    depth = np.frombuffer(exr_file.channel(channel_name, pixel_type), dtype=np.float32).reshape((height, width))

    return depth



def depth_in_exrs_to_gif(exrs, save_name = 'logs/exr_depth2.gif'):
    img_bank= []
    for x in tqdm(exrs):
        depth_map = read_exr_as_depth(x)
        depth_map = depth_map.copy()
        depth_map[depth_map== depth_map.max()] = 0
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        colormap = plt.cm.jet 
        rgb_image = colormap(normalized_depth)  # 将深度图映射到 RGBA 图像

        # 去除 alpha 通道，只保留 RGB 通道
        rgb_image = (rgb_image[:, :, :3] * 255).astype(np.uint8) 
        img_bank.append(Image.fromarray(rgb_image))
    
    images_to_gif2(img_bank,save_name)




import os 





from PIL import Image   

from utils.img_utils import * 

import numpy as np

import matplotlib.pyplot as plt 
import matplotlib


def colorize_depth_map(depth, mask=None, reverse_color=False):
    from decord import VideoReader,cpu
    
    
    cm = matplotlib.colormaps["Spectral"]


    # normalize
    norm_min = depth.min()
    norm_max = depth.max()
    
    depth = ((depth - norm_min) / (norm_max - norm_min))
    # colorize
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]  # Invert the depth values before applying colormap
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3] # (h,w,3)

    depth_colored = (img_colored_np * 255).astype(np.uint8) 
    # if mask is not None:
    #     masked_image = np.zeros_like(depth_colored)
    #     masked_image[mask.numpy()] = depth_colored[mask.numpy()]
    #     depth_colored_img = Image.fromarray(masked_image)
    # else:
    depth_colored_img = Image.fromarray(depth_colored)
    return depth_colored_img



def colorize_depth_map_quantile4norm(depth, mask=None, reverse_color=False):
    from decord import VideoReader,cpu
    
    cm = matplotlib.colormaps["Spectral"]


    # normalize
    norm_min = depth.min()
    # norm_max = depth.max()
    norm_max = np.quantile(depth, 0.99)
    
    depth = ((depth - norm_min) / (norm_max - norm_min))
    # print(f'after normalize, depth.max():{depth.max()},depth.min():{depth.min()}')

    depth = depth.clip(0,1)

    # colorize
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]  # Invert the depth values before applying colormap
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3] # (h,w,3)

    depth_colored = (img_colored_np * 255).astype(np.uint8) 
    
    # Apply mask: set masked regions to pink color
    if mask is not None:
        # Pink color: RGB(255, 192, 203)
        pink_color = np.array([255, 192, 203], dtype=np.uint8)
        depth_colored[~mask] = pink_color
        depth_colored_img = Image.fromarray(depth_colored)
    else:
        depth_colored_img = Image.fromarray(depth_colored)
    return depth_colored_img





if __name__== '__main__':

    

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/baai-cwm-1/baai_cwm_ml/cwm/shaocong.xu/exp/BlenderProc/examples/datasets/bop_challenge/output_debug_front8/bop_data/lm/train_pbr/000000/rgb")
    parser.add_argument('--output_path', type=str, default="logs/instance_segmaps.gif")
    parser.add_argument('--duration', type=int, default=20)
    args = parser.parse_args()



    out_name = args.output_path
    imgs = glob.glob(args.root_path + '/*')
    imgs = sorted([x for x in imgs if  x.endswith('jpg') or   x.endswith('png') or x.endswith('jpeg') ])

    
    print('there are %d imgs'%(len(imgs)))

    images_to_gif(imgs,output_gif = out_name, duration = args.duration)
