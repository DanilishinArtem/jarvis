import torch
import torchaudio
from Modules.tokenizer import VoiceBpeTokenizer
from Modules.gpt import GPT
from Modules.hifigan_decoder import HifiDecoder
from Modules.tokenizer import split_sentence
import sounddevice as sd


class Pipeline:
    def __init__(self, path_to_vocab, path_to_hifigan_decoder, path_to_gpt_model, path_to_sample_wav, path_to_mel_stats, device):
        self.device = device
        self.speed = 1.0
        self.path_to_vocab = path_to_vocab
        self.path_to_hifigan_decoder = path_to_hifigan_decoder
        self.path_to_gpt_model = path_to_gpt_model
        self.path_to_sample_wav = path_to_sample_wav
        self.mel_stats = torch.load(path_to_mel_stats)
        self.tokenizer = None
        self.hifigan_decoder = None
        self.gpt = None
        self.gpt_number_text_tokens = None
        self.gpt_start_text_token = None
        self.gpt_stop_text_token = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        
        self.init_models()

    def init_models(self):
        # initialization of the tokenizer
        self.tokenizer = VoiceBpeTokenizer(self.path_to_vocab)
        self.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
        self.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
        self.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")
        print(f'Tokenizer loaded.')

        # initialization of the hifigan model
        input_sample_rate: int = 22050
        output_sample_rate: int = 24000
        output_hop_length: int = 256
        gpt_code_stride_len: int = 1024
        decoder_input_dim: int = 1024
        d_vector_dim: int = 512
        cond_d_vector_in_each_upsampling_layer: bool = True
        self.get_hifigan_decoder(input_sample_rate, output_sample_rate, output_hop_length, gpt_code_stride_len, decoder_input_dim, d_vector_dim, cond_d_vector_in_each_upsampling_layer)
        state_dict_hifigan_decoder = torch.load(self.path_to_hifigan_decoder)
        self.hifigan_decoder.load_state_dict(state_dict_hifigan_decoder)
        print(f'Hifigan_decoder loaded.')

        # initialization of the gpt model
        gpt_max_audio_tokens: int = 605
        gpt_max_text_tokens: int = 402
        gpt_max_prompt_tokens: int = 70
        gpt_layers: int = 30
        gpt_n_model_channels: int = 1024
        gpt_n_heads: int = 16
        gpt_num_audio_tokens: int = 1026
        gpt_start_audio_token: int = 1024
        gpt_stop_audio_token: int = 1025
        gpt_code_stride_len: int = 1024
        gpt_use_masking_gt_prompt_approach: bool = True
        gpt_use_perceiver_resampler: bool = True
        self.get_gpt_model(gpt_layers,gpt_n_model_channels,self.gpt_start_text_token,self.gpt_stop_text_token,gpt_n_heads,gpt_max_text_tokens,gpt_max_audio_tokens,gpt_max_prompt_tokens,self.gpt_number_text_tokens,gpt_num_audio_tokens,gpt_start_audio_token,gpt_stop_audio_token,gpt_use_perceiver_resampler,gpt_code_stride_len)
        self.gpt.init_gpt_for_inference()
        state_dict_gpt = torch.load(self.path_to_gpt_model)
        self.gpt.load_state_dict(state_dict_gpt)
        print(f'Gpt model loaded.')

        # initialization of the gpt_cond_latent and speaker_embedding
        self.get_conditioning_latents()

    def load_audio(self, audiopath, sampling_rate):
        audio, lsr = torchaudio.load(audiopath)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        if torch.any(audio > 10) or not torch.any(audio < 0):
            print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)
        return audio

    def wav_to_mel_cloning(self,wav,mel_norms_file="",mel_norms=None,n_fft=4096,hop_length=1024,win_length=4096,power=2,normalized=False,sample_rate=22050,f_min=0,f_max=8000,n_mels=80):
        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            normalized=normalized,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).to(self.device)
        wav = wav.to(self.device)
        mel = mel_stft(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if mel_norms is None:
            mel_norms = torch.load(mel_norms_file, map_location=self.device)
        mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel

    def get_hifigan_decoder(self,input_sample_rate,output_sample_rate,output_hop_length,ar_mel_length_compression,decoder_input_dim,d_vector_dim,cond_d_vector_in_each_upsampling_layer):
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_hop_length=output_hop_length,
            ar_mel_length_compression=ar_mel_length_compression,
            decoder_input_dim=decoder_input_dim,
            d_vector_dim=d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=cond_d_vector_in_each_upsampling_layer,
        )

    def get_gpt_model(
        self,
        layers,
        model_dim,
        start_text_token,
        stop_text_token,
        heads,
        max_text_tokens,
        max_mel_tokens,
        max_prompt_tokens,
        number_text_tokens,
        num_audio_tokens,
        start_audio_token,
        stop_audio_token,
        use_perceiver_resampler,
        code_stride_len
        ):
        self.gpt = GPT(
            layers=layers,
            model_dim=model_dim,
            start_text_token=start_text_token,
            stop_text_token=stop_text_token,
            heads=heads,
            max_text_tokens=max_text_tokens,
            max_mel_tokens=max_mel_tokens,
            max_prompt_tokens=max_prompt_tokens,
            number_text_tokens=number_text_tokens,
            num_audio_tokens=num_audio_tokens,
            start_audio_token=start_audio_token,
            stop_audio_token=stop_audio_token,
            use_perceiver_resampler=use_perceiver_resampler,
            code_stride_len=code_stride_len,
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        max_ref_length=30,
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050
        ):

        if not isinstance(self.path_to_sample_wav, list):
            audio_paths = [self.path_to_sample_wav]
        else:
            audio_paths = self.path_to_sample_wav
        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = self.load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]
            self.speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(self.speaker_embedding)
            audios.append(audio)
        full_audio = torch.cat(audios, dim=-1)
        self.gpt_cond_latent = self.get_gpt_cond_latents(full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len)  # [1, 1024, T]

        if speaker_embeddings:
            self.speaker_embedding = torch.stack(speaker_embeddings)
            self.speaker_embedding = self.speaker_embedding.mean(dim=0)

        self.gpt_cond_latent = self.gpt_cond_latent.to(self.device)
        self.speaker_embedding = self.speaker_embedding.to(self.device)

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]

        style_embs = []
        for i in range(0, audio.shape[1], 22050 * chunk_length):
            audio_chunk = audio[:, i : i + 22050 * chunk_length]
            if audio_chunk.size(-1) < 22050 * 0.33:
                continue
            mel_chunk = self.wav_to_mel_cloning(
                audio_chunk,
                mel_norms=self.mel_stats.cpu(),
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
            style_embs.append(style_emb)
        cond_latent = torch.stack(style_embs).mean(dim=0)
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True).unsqueeze(-1).to(self.device))

    def generate(self, text):
        language = "ru"
        length_scale = 1.0 / max(self.speed, 0.05)
        text = [text]
        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang="ru")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=self.gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=True,
                    top_p=0.85,
                    top_k=50,
                    temperature=0.75,
                    num_return_sequences=1,
                    num_beams=1,
                    length_penalty=1.0,
                    repetition_penalty=5.0,
                    output_attentions=False,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=self.gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )
                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=self.speaker_embedding).cpu().squeeze())
        result = torch.cat(wavs, dim=0).numpy()
        sd.play(result, samplerate=24000)