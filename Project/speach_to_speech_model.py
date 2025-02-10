from librosa import feature
import librosa.display
import torch
from torch import nn
import torch.optim as optim
import os
import librosa
import librosa.display

from Audio_to_Values import mp3_wav, audio_to_pitch_over_time, audio_to_volume_over_time


import soundfile as sf

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


class SpectrogramTransformer(nn.Module):
    """
    Sequence-to-Sequence model using transformer encoder and transformer decoder.
    Goal is to convert a emotional mel spectrogram from a neutral spectrogram
    """
    def __init__(self, input_dim=128, num_heads=8, hidden_dim=256, num_layers=6):
        super(SpectrogramTransformer, self).__init__()

        # encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # linear Projection
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, src, tgt):
        """
        :param src: Neutral spectrogram [seq_len, batch, features]
        :param tgt: Emotional spectrogram (shifted) [seq_len, batch, features]
        """
        memory = self.encoder(src)  # encode the neutral spectrogram
        output = self.decoder(tgt, memory)  # decode to the emotional spectrogram
        output = self.output_layer(output)  # project back to Mel scale
        return output


def create_model():
    # model configuration
    input_dim = 128
    num_heads = 8
    hidden_dim = 256
    num_layers = 6

    # initialize model
    model = SpectrogramTransformer(input_dim, num_heads, hidden_dim, num_layers)
    model = model.to(DEVICE)

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR + '\\audio_files\\AudioWAV'

    neutral_spec = load_audio_to_spectrogram(file_path=AUDIO_DIR + '\\1001_DFA_NEU_XX.wav')
    emotional_spec = load_audio_to_spectrogram(file_path=AUDIO_DIR + '\\1001_DFA_ANG_XX.wav')

    train_model(model, optimizer, criterion, AUDIO_DIR)

    # convert to audio and save
    generated_spec = generate_emotional_spectrogram(model, neutral_spec)
    audio_output = spectrogram_to_audio(generated_spec)
    sf.write("generated_emotional.wav", audio_output, 16000)

    pass


def load_audio_to_spectrogram(file_path, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    """Load audio and convert it to a Mel spectrogram.
    """
    waveform, sample_rate = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = librosa.power_to_db(mel_spec)  # Convert to decibels
    return torch.tensor(mel_spec, dtype=torch.float32, device=DEVICE)


def spectrogram_to_audio(mel_spec, sr=16000, n_fft=400, hop_length=160):
    """Convert a Mel spectrogram back to audio using Griffin-Lim.
    """
    mel_spec = librosa.db_to_power(mel_spec.detach().cpu().numpy())  # Convert from dB
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return audio


def train_step(model, optimizer, criterion, neutral_spectrogram, emotional_spectrogram):
    # single training step, called inside training loop
    model.train()

    # reshape for Transformer [seq_len, batch, features]
    neutral_spectrogram = neutral_spectrogram.T.unsqueeze(1)
    emotional_spectrogram = emotional_spectrogram.T.unsqueeze(1)

    optimizer.zero_grad()
    output = model(neutral_spectrogram, emotional_spectrogram)
    loss = criterion(output, emotional_spectrogram)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(model, optimizer, criterion, AUDIO_DIR, n_epochs=100):
    neutral_audio_file_names = [f for f in os.listdir(AUDIO_DIR) if 'NEU' in f]
    emotional_audio_file_names = [f for f in os.listdir(AUDIO_DIR) if 'ANG' in f]

    use_emotion = 'ANG'

    for i in range(n_epochs):
        epoch_loss = 0
        total_loss_count = 0
        for f_neutral in neutral_audio_file_names:
            neutral_spec = load_audio_to_spectrogram(file_path=AUDIO_DIR + '\\' + f_neutral)
            neutral_sentence, _ = filter_tags(f_neutral)
            loss_per_neutral = 0
            loss_count = 0
            for f_emotional in emotional_audio_file_names:
                emotional_sentence, emotion = filter_tags(f_emotional)
                if f_neutral.replace('NEU', '') != f_emotional.replace(emotion, '') or emotion != use_emotion:
                    continue
                emotional_spec = load_audio_to_spectrogram(file_path=AUDIO_DIR + '\\' + f_emotional)
                loss = train_step(model, optimizer, criterion, neutral_spec, emotional_spec)
                loss_per_neutral += loss
                loss_count += 1
            total_loss = loss_per_neutral / loss_count if loss_count > 0 else None
            if total_loss is not None:
                epoch_loss += total_loss
                total_loss_count += 1
            # print(f"Training Loss: {total_loss}")
        print(f'Epoch Loss: {epoch_loss / total_loss_count}')
    pass


def generate_emotional_spectrogram(model, neutral_spectrogram):
    # predict spectrogram using model
    model.eval()
    neutral_spectrogram = neutral_spectrogram.T.unsqueeze(1)  # [seq_len, batch, features]
    generated_spectrogram = model(neutral_spectrogram, neutral_spectrogram).squeeze(1).T
    return generated_spectrogram


def filter_tags(s):
    sp = s.split('_')
    return sp[1], sp[2]


if __name__ == '__main__':
    create_model()
