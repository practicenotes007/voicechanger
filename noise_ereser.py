import librosa
import numpy as np
import soundfile as sf
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

def process_audio(input_path, output_path):
    """
    处理音频文件，去除白噪声
    :param input_path: 输入音频路径
    :param output_path: 输出音频路径
    """
    print("开始处理音频...")
    
    try:
        # 加载音频
        audio_data, sr = librosa.load(input_path, sr=None)
        
        # 去除噪声
        print("去除噪声...")
        cleaned_audio = remove_noise(audio_data, sr)
        
        # 保存处理后的音频
        print("保存音频...")
        sf.write(output_path, cleaned_audio, sr)
        
        print(f"处理完成！输出文件保存在: {output_path}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='音频去噪工具')
    parser.add_argument('--input', required=True, help='输入音频文件路径')
    parser.add_argument('--output', required=True, help='输出音频文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入文件 '{args.input}' 不存在")
        return
    
    try:
        process_audio(args.input, args.output)
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()