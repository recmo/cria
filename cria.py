#!/usr/bin/env python

from sentencepiece import SentencePieceProcessor
from zipfile import ZipFile
import json
import pickle
import numpy as np    

def main(
    prompt: str = "I believe the meaning of life is",
    n_tokens_to_generate: int = 5,
    model_size: str = "7B",
    models_dir: str = "models"
    ):

    # Load Tokenizer
    tokenizer = SentencePieceProcessor(model_file=f"{models_dir}/tokenizer.model")
    print(f"vocab_size: {tokenizer.vocab_size()}")
    print(f"bos_id: {tokenizer.bos_id()}")
    print(f"eos_id: {tokenizer.eos_id()}")
    print(f"pad_id: {tokenizer.pad_id()}")

    # Load model
    with open(f'{models_dir}/{model_size}/params.json') as f:
        params = json.load(f)
    with ZipFile(f'{models_dir}/{model_size}/consolidated.00.pth', 'r') as zip:
        class Unpickler(pickle.Unpickler):
            def find_class(self, mod_name, name):
                if mod_name == 'torch._utils' and name == '_rebuild_tensor_v2':
                    def rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        assert storage_offset == 0
                        assert requires_grad == False
                        assert len(backward_hooks) == 0
                        stride = np.array(stride) * storage.strides
                        return np.lib.stride_tricks.as_strided(storage, size, stride, writeable=False)
                    return rebuild_tensor
                if mod_name == 'torch' and name == 'HalfStorage':
                    return np.half
                return super().find_class(mod_name, name)
            def persistent_load(self, saved_id):
                (typename, dtype, key, location, count) = saved_id
                name = f'consolidated/data/{key}'
                assert zip.getinfo(name).file_size == count * dtype(0).itemsize
                data = np.frombuffer(zip.read(name), dtype)
                assert data.shape == (count,)
                return data
        with zip.open('consolidated/data.pkl') as data_pkl:
            data = Unpickler(data_pkl).load()

    print(params)
    for (key, value) in data.items():
        print(key, value.shape)

    # Tokenize prompt
    tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    tokens += [tokenizer.pad_id()] * n_tokens_to_generate
    print(f"Encoded \"{prompt}\" to {tokens}")

    # Embed tokens
    print("tok_embeddings.weight = ", data['tok_embeddings.weight'].shape, data['tok_embeddings.weight'])
    tokens = data['tok_embeddings.weight'][1,:]
    print(tokens.shape, tokens)

    # Untokenize result
    result = tokenizer.decode(tokens)
    print(result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

# torch.HalfStorage = 1 sign, 5 exponent, and 10 significand bits. 