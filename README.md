# Mispronunciation Detection project

## What we have so far 

- The APL model has been reimplemented. It is as of now of this shape : 

    `Acoustic (batch, time_length, acoustic_size), Phonetic (batch, time_length, phonetic_size) , Phoneme (batch, sequence_length) -> Output (batch, time_length, vocab_size)`

- Are implemented : 
    - The beam-search decoding used by the [APL article](https://arxiv.org/pdf/2110.07274)
    - The tokenization from a string of phoneme sequence to list of corresponding ids
    - The use of Wav2Vec vectors inputs (the hidden layers of the transformer network of a [version fine-tuned for phoneme recognition](https://huggingface.co/mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme)) and a pre-processing allowing learnable use of weighted hidden layers.

## What to do next 

- Reflexion on the training part : 
    - What loss and what task (META-learning, Entropy minimisation)
    - According to the choice, what modification to the model : a single prediction at each time of the whole sequence 