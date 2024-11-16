import whisper
import ffmpeg
import os

def extract_audio(video_path, audio_path="audio.mp3"):
    """
    从视频文件中提取音频
    """
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
        print(f"音频已保存到: {audio_path}")
    except ffmpeg.Error as e:
        print("音频提取失败:", e)

def transcribe_audio(audio_path):
    """
    使用 Whisper 模型提取音频中的文字
    """
    model = whisper.load_model("base")  # 使用基础模型，可选 base、small、medium、large
    result = model.transcribe(audio_path)
    return result["text"]

if __name__ == "__main__":
    # 设置视频路径
    video_path = "your_video.mp4"
    audio_path = "extracted_audio.mp3"

    # 提取音频
    extract_audio(video_path, audio_path)

    # 转录文字
    text = transcribe_audio(audio_path)

    # 输出结果
    print("提取的文字内容：")
    print(text)

    # 保存到文本文件
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("文字已保存到 transcription.txt")
