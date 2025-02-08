from TTS.api import TTS
import os


def convert_text_to_audio_and_store(text, output_file_path, gender='female', emotion='NEU'):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # XTTS v2
    if gender == 'female':
        speaker_id = 1028
    elif gender == 'male':
        speaker_id = 1080
    else:
        return

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR.replace('\\Project\\cycleGAN_model', '') + '\\audio_files\\AudioWAV'
    speaker_wav_dir = AUDIO_DIR + f'\\{speaker_id}_DFA_{emotion}_XX.wav'

    all_files = [AUDIO_DIR + '\\' + s for s in os.listdir(AUDIO_DIR)]

    if speaker_wav_dir in all_files:
        tts.tts_to_file(text=text, file_path=output_file_path, speaker_wav=speaker_wav_dir, language='en')


def benchmark_emotion_conversion():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR + '\\audio_files\\AudioWAV'
    BENCHMARK_COMPLICATED_DIR = ROOT_DIR + '\\Benchmark_complicated.txt'
    BENCHMARK_EASY_DIR = ROOT_DIR + '\\Benchmark_easy.txt'
    BENCHMARK_COMPLICATED_OUTPUT_DIR = ROOT_DIR + '\\Benchmark_wav\\complicated'
    BENCHMARK_EASY_OUTPUT_DIR = ROOT_DIR + '\\Benchmark_wav\\easy'

    with open(BENCHMARK_EASY_DIR, 'r') as file:
        sentences_easy = file.readlines()
    sentences_easy = [s.replace('\n', '') for s in sentences_easy]

    with open(BENCHMARK_COMPLICATED_DIR, 'r') as file:
        sentences_complicated = file.readlines()
    sentences_complicated = [s.replace('\n', '') for s in sentences_complicated]

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")   # XTTS v2

    emotions = {s.split('_')[2] for s in os.listdir(AUDIO_DIR)}
    all_files = [AUDIO_DIR + '\\' + s for s in os.listdir(AUDIO_DIR)]

    for i, s in enumerate(sentences_easy):
        sentence_num = '0' + str(i) if i < 10 else str(i)
        female_speaker_id = 1028
        for emotion in emotions:
            female_speaker_wav = AUDIO_DIR + f'\\{female_speaker_id}_DFA_{emotion}_XX.wav'
            if female_speaker_wav in all_files:
                tts.tts_to_file(text=s,
                                file_path=BENCHMARK_EASY_OUTPUT_DIR + f'\\{emotion}' + f"\\FEMALE_{sentence_num}_{emotion}.wav",
                                speaker_wav=female_speaker_wav,
                                language='en')

    for i, s in enumerate(sentences_complicated):
        sentence_num = '0' + str(i) if i < 10 else str(i)
        female_speaker_id = 1028
        for emotion in emotions:
            female_speaker_wav = AUDIO_DIR + f'\\{female_speaker_id}_DFA_{emotion}_XX.wav'
            if female_speaker_wav in all_files:
                tts.tts_to_file(text=s,
                                file_path=BENCHMARK_COMPLICATED_OUTPUT_DIR + f'\\{emotion}' + f"\\FEMALE_{sentence_num}_{emotion}.wav",
                                speaker_wav=female_speaker_wav,
                                language='en')

    pass


if __name__ == '__main__':
    benchmark_emotion_conversion()
    convert_text_to_audio_and_store(text="After all that happened, I can't imagine my life any other way.",
                                    output_file_path='output_voice.wav', gender='female', emotion='ANG')

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # XTTS v2

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR.replace('\\Project\\cycleGAN_model', '') + '\\audio_files\\AudioWAV'
    HAPPY_DIR = ROOT_DIR.replace('\\Project\\cycleGAN_model', '') + '\\happy_audios_generated'
    GENERATED_DIR = ROOT_DIR.replace('\\Project\\cycleGAN_model', '') + '\\generated_audio_samples'

    speaker_ids = {s.split('_')[0] for s in os.listdir(AUDIO_DIR)}
    emotions = {s.split('_')[2] for s in os.listdir(AUDIO_DIR)}
    sentences = {s.split('_')[1] for s in os.listdir(AUDIO_DIR)}
    all_files = [AUDIO_DIR + '\\' + s for s in os.listdir(AUDIO_DIR)]

    # test different speaks
    # for speaker in speaker_ids:
    #     sp_audio = AUDIO_DIR + f'\\{speaker}_DFA_HAP_XX.wav'
    #     if sp_audio in all_files:
    #         tts.tts_to_file(text="I am excited!", file_path=HAPPY_DIR + f"\\{speaker}_HAP.wav", speaker_wav=sp_audio,
    #                         language='en')

    # generate samples for selected speakers
    female_speaker_id = 1028
    male_speaker_id = 1080
    for emotion in emotions:
        female_speaker_wav = AUDIO_DIR + f'\\{female_speaker_id}_DFA_{emotion}_XX.wav'
        if female_speaker_wav in all_files:
            tts.tts_to_file(text="After all that happened, I can't imagine my life any other way.",
                            file_path=GENERATED_DIR + f"\\FEMALE_{emotion}.wav", speaker_wav=female_speaker_wav,
                            language='en')

        male_speaker_wav = AUDIO_DIR + f'\\{male_speaker_id}_DFA_{emotion}_XX.wav'
        if female_speaker_wav in all_files:
            tts.tts_to_file(text="After all that happened, I can't imagine my life any other way.",
                            file_path=GENERATED_DIR + f"\\MALE_{emotion}.wav", speaker_wav=male_speaker_wav,
                            language='en')

    pass
