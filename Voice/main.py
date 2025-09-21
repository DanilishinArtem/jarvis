from Modules.pipeline_synthesize import Pipeline


if __name__ == "__main__":
    # path_to_vocab, path_to_hifigan_decoder, path_to_gpt_model, path_to_sample_wav, path_to_mel_stats, device
    pipeline = Pipeline(
        path_to_vocab = "/Users/adanilishin/jarvis/Voice/Models/vocab.json",
        path_to_hifigan_decoder = "/Users/adanilishin/jarvis/Voice/Models/hifigan_decoder.pt",
        path_to_gpt_model = "/Users/adanilishin/jarvis/Voice/Models/gpt.pt",
        path_to_sample_wav = "/Users/adanilishin/jarvis/Voice/Samples/adanilishin_input.wav",
        path_to_mel_stats = "/Users/adanilishin/jarvis/Voice/Models/mel_stats.pth",
        device = "cpu"
    )
    print(f'[DEBUG] Pipeline initialized')
    
    text = "iPhone 19 получил ультратонкий корпус из титана с практически безрамочным дисплеем, который плавно переходит в боковые грани."
    pipeline.generate(text)
    print(f'[DEBUG] Voice generated')