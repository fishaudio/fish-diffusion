from pathlib import Path
import yaml
from ipatok import tokenise

# from https://www.internationalphoneticassociation.org/sites/default/files/phonsymbol.pdf
phonemes_dict = yaml.safe_load(Path(__file__).with_name("symbols.yaml").open())

def text_to_sequence(text):
    seq = []
    tokens = tokenise(text, tones=True,strict=True)
    for token in tokens:
        for char in token:
            if char in phonemes_dict:
                seq.append(phonemes_dict[char])
            else:
                print("Unknown phoneme: {}".format(char))
    return seq