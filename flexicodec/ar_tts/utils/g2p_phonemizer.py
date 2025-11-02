from phonemizer import phonemize
from phonemizer.separator import Separator
import whisper
import logging

from phonemizer.backend import EspeakBackend
phonemizer_en = EspeakBackend('en-us', preserve_punctuation=True, with_stress=False, language_switch="remove-flags")


def shorten_sequence(g2p_text):
    en_phn = g2p_text
    en_phn = en_phn.replace('ː', '~') # in order not to make the BPE sequence too long
    en_phn = en_phn.replace('ʌ', 'α')
    en_phn = en_phn.replace('ɑ', 'α')
    en_phn = en_phn.replace('ɪ', 'ι')
    en_phn = en_phn.replace('ʊ', 'υ')
    en_phn = en_phn.replace('ɹ', 'r')
    en_phn = en_phn.replace('ɐ', 'a')
    en_phn = en_phn.replace('ʃ', 'ρ')
    en_phn = en_phn.replace('ɛ', 'ε')
    en_phn = en_phn.replace('ŋ', 'ng')
    en_phn = en_phn.replace('ɡ', 'g')
    en_phn = en_phn.replace('ɔ', 'ο')
    en_phn = en_phn.replace('ʒ', 'ψ')
    en_phn = en_phn.replace('ɾ', 'τ')
    en_phn = en_phn.replace('ɚ', 'ər')
    en_phn = en_phn.replace('ᵻ', 'i')
    en_phn = en_phn.replace('ɜ', 'ə')
    return en_phn

# G2P function to convert text to IPA phones using the phonemizer library
def g2p_phonemizer_en(text, language='en-us', shorten_g2p_sequence=True):
    """
    Convert text to IPA phones using the phonemizer library.
    Returns: the g2p string.
    """
    # en_phn = phonemize(
    #     text,
    #     language=language,
    #     backend='espeak',
    #     strip=True,         # Strip any unnecessary text (like symbols)
    #     preserve_punctuation=True,  # Preserve punctuation for easier testing
    #     with_stress=False,  # Don't include stress markers in IPA
    #     njobs=4             # Number of parallel jobs (optional for performance)
    # )
    text = text.replace('^', "'")
    en_phn = phonemizer_en.phonemize([text], njobs=1)[0].strip()
    if shorten_g2p_sequence:
        en_phn = shorten_sequence(en_phn)
    return en_phn
class WarningFilter(logging.Filter):
    def filter(self, record):
        # 只过滤 phonemizer 中的 WARNING 级别日志
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        if record.name == "qcloud_cos.cos_client" and record.levelno == logging.INFO:
            return False
        if record.name == "jieba" and record.levelno == logging.DEBUG:
            return False
        return True


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)

if __name__ == '__main__':
    # Example input text
    input_text = "First, we propose a novel Mamba-based encoder-decoder architecture that overcomes the limitations of previous."

    # Step 1: Convert the text to IPA using G2P phonemizer
    ipa_input = g2p_phonemizer_en(input_text)
    print(f"Generated IPA: {ipa_input}")

    # # Step 2: Initialize Whisper's tokenizer
    # tokenizer = whisper.tokenizer.get_tokenizer(
    #     multilingual=True, 
    #     num_languages=100, 
    #     language='en', 
    #     task='transcribe'
    # )

    # # Step 3: Tokenize the IPA string
    # tokens = tokenizer.encode(ipa_input)

    # # Step 4: Decode the tokens back into text
    # decoded_text = tokenizer.decode(tokens)

    # # Output the results
    # print("Original IPA input: ", ipa_input)
    # print("Original IPA input len: ", len(ipa_input))
    # print("Tokens:", tokens)
    # print("Tokens len:", len(tokens))
    # print("Decoded Text:", decoded_text)
    # print(ipa_input == decoded_text)
