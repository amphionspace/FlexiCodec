import whisper
import numpy as np
import random
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")

def text2idx(text, language="en", g2p_prob=0.0):

    language_token = tokenizer.to_language_token(language)
    g2p = np.random.random() < g2p_prob
    if g2p:
        from .g2p_phonemizer import g2p_phonemizer_en
        # Convert text to phonemes using G2P phonemizer
        text = g2p_phonemizer_en(text)
    return [language_token] + tokenizer.encode(text)

def text2idx_v2(
    text,
    language="en",
    g2p_prob=0.0,
    mode="train",
    mean=0.0,
    std=0.5,
    full_g2p_prob=0.2,
    shorten_g2p_sequence=True,
):
    import string
    import copy
    language_token = tokenizer.to_language_token(language)
    g2p = np.random.random() < g2p_prob
    if g2p:
        from .g2p_phonemizer import g2p_phonemizer_en as processor
        words = text.split()
        transformed_words = words
        if language in ["en", "fr"]:
            if mode == "train":
                norm_phone_ratio = np.abs(np.random.normal(mean, std))
                if np.random.random() < full_g2p_prob:
                    norm_phone_ratio = 1.0
                if norm_phone_ratio == 1.0:
                    transformed_words = processor(text, shorten_g2p_sequence=shorten_g2p_sequence)
                else:
                    norm_phone_ratio = min(max(norm_phone_ratio, 0), 1)
                    num_to_transform = round(len(words) * norm_phone_ratio)
                    num_to_transform = min(num_to_transform, len(words))
                    words_to_transform = random.sample(words, num_to_transform)
                    try:
                        transformed_words = [
                            processor(word, shorten_g2p_sequence=shorten_g2p_sequence) if word in words_to_transform and word not in string.punctuation else word
                            for word in words
                        ]
                        transformed_words = ' '.join(transformed_words)
                    except Exception as e:
                        print('exception', e)
                        transformed_words = text
                text = transformed_words
            elif mode == "eval":
                try:
                    transformed_words = [
                        processor(word, shorten_g2p_sequence=shorten_g2p_sequence) if word not in string.punctuation else word
                        for word in words
                    ]
                    text = ' '.join(transformed_words)
                except Exception as e:
                    print('exception', e)
                    text = ' '.join(words)
    return [language_token] + tokenizer.encode(text)

if __name__ == "__main__":
    print(text2idx("Hello, world!"))