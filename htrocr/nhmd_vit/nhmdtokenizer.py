import torch

class NHMDTokenizer:
    def __init__(self):
        super().__init__()
        self.alphabet = "Øäíéûc1TČIüąp°CNšëć{,æDöãSà|0Uj—m5²ïek³f%řèZ'Q8çŢaXsJÆåqvẞ>ý[ŠÁ!±EVzA/3L\"B2.RhOÉ);φ79Wźr4ê ly~ūdwŚ]Åčð_K=ńH<„âiśĕ÷ìG§ó-:M#6užú^tFÖň&}«bgõYnáñ+(ø*?⁒`oxÓP"
        self.token2id = {token: i for i, token in enumerate(self.alphabet, 2)}
        self.id2token = {i: token for i, token in enumerate(self.alphabet, 2)}
        self.token2id.update({'<PAD>':0, '<EOS>':1})
        self.id2token.update({0:'<PAD>', 1:'<EOS>'})
        # Ignoring start of sentence token - no need to have it - start decoding with the first element
        self.bos_token_id = None
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def batch_encode_plus(self, char_sequences, padding="max_length", max_length=None, return_tensors='pt', truncation=True):       
        encoded_sequences = []
        for sequence in char_sequences:
            # Encode text and skip unknown chars
            encoded_sequence = [self.token2id[char] for char in sequence if char in self.token2id]
            encoded_sequence.append(self.eos_token_id)
            encoded_sequences.append(encoded_sequence)

        # Pad or truncate the sequences to max_length
        if max_length is not None:
            encoded_sequences = [seq[:max_length] + [self.pad_token_id] * (max_length - len(seq[:max_length])) for seq in encoded_sequences]

        input_ids = torch.tensor(encoded_sequences)
        attention_mask = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id2token[i] for i in list(token_ids.detach().cpu().numpy())]
        decoded = ''.join(tokens)
        if skip_special_tokens:
            decoded = decoded.replace('<PAD>', '').replace('<EOS>', '')
        return decoded