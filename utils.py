class CustomTokenizer:
    # Custom tokenizer class for converting text into sequences
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.next_index = 1

    def fit_on_texts(self, texts):
        # Fit the tokenizer on the provided texts
        for text in texts:
            if text not in self.word_index:
                self.word_index[text] = self.next_index
                self.index_word[self.next_index] = text
                self.next_index += 1

    def texts_to_sequences(self, texts):
        # Convert texts to sequences of indices based on the fitted tokenizer
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split(', '):
                word = word.strip()
                if word in self.word_index:
                    sequence.append(self.word_index[word])
            sequences.append(sequence)
        return sequences
