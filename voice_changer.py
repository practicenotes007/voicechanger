import argparse
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import tempfile
from scipy.signal import wiener

def remove_noise(audio_data, sr):
    """
    去除音频中的白噪声
    :param audio_data: 音频数据
    :param sr: 采样率
    :return: 处理后的音频数据
    """
    # 使用Wiener滤波器去除噪声
    return wiener(audio_data)

def change_pitch(audio_data, sr, pitch_factor):
    """
    改变音频的音调
    :param audio_data: 音频数据
    :param sr: 采样率
    :param pitch_factor: 音调改变因子
    :return: 处理后的音频数据
    """
    return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=12 * np.log2(pitch_factor))

def process_video(input_path, output_path, pitch_factor):
    """
    处理视频文件，改变其音频的音调
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param pitch_factor: 音调改变因子
    """
    print("开始处理视频...")
    
    temp_dir = tempfile.mkdtemp()
    temp_audio = os.path.join(temp_dir, "temp_audio.wav")
    
    try:
        # 加载视频
        video = VideoFileClip(input_path)
        
        # 提取音频
        print("提取音频...")
        video.audio.write_audiofile(temp_audio)
        
        # 处理音频
        print("处理音频...")
        audio_data, sr = librosa.load(temp_audio, sr=None)
        
        # 去除噪声
        print("去除噪声...")
        cleaned_audio = remove_noise(audio_data, sr)
        
        # 改变音调
        modified_audio = change_pitch(cleaned_audio, sr, pitch_factor)
        
        # 保存修改后的音频
        temp_modified_audio = os.path.join(temp_dir, "temp_modified_audio.wav")
        sf.write(temp_modified_audio, modified_audio, sr)
        
        # 合并音频和视频
        print("合成视频...")
        video = video.set_audio(AudioFileClip(temp_modified_audio))
        
        # 保存结果（显式指定fps）
        print("保存视频...")
        video.write_videofile(output_path, fps=video.fps)
        
        print(f"处理完成！输出文件保存在: {output_path}")
        
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            modified_audio_path = os.path.join(temp_dir, "temp_modified_audio.wav")
            if os.path.exists(modified_audio_path):
                os.remove(modified_audio_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='视频声音变频工具')
    parser.add_argument('--input', required=True, help='输入视频文件路径')
    parser.add_argument('--output', required=True, help='输出视频文件路径')
    parser.add_argument('--pitch', type=float, default=1.5, help='音调调整因子 (>1 提高音调, <1 降低音调)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入文件 '{args.input}' 不存在")
        return
    
    try:
        process_video(args.input, args.output, args.pitch)
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()