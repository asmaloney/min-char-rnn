"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Changes by Andy Maloney <asmaloney@gmail.com>:
    - fixed for Python 3
    - reorganized code using classes
    - added type hints
    - renamed some variables for clarity
    - added STARTING_TEXT string instead of just using one character
    - added constants at the top for easy modification
    - added option USE_WORDS to use words instead of characters

References:
    https://gist.github.com/karpathy/d4dee566867f8291f086
    https://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import numpy as np
import numpy.typing as npt

# The input file (text) to use as a source for training
INPUT_FILE: str = "./data/Shakespeare-large.txt"

# Text to start our inferences with
# Note: the words in this text must appear in the input file
INFERENCE_STARTING_TEXT: str = "I was anointed king at nine months old.\n"

# The number of tokens to output in our inference sample
INFERENCE_SAMPLE_SIZE: int = 300

# How frequently to run inference (e.g. every N iterations)
INFERENCE_FREQUENCY: int = 1000

# Use words instead of characters
TRAINING_USE_WORDS: bool = False

# If using words instead of chars, adjust parameters so we get output in a reasonable time.
# Might also consider using smaller data sets for experimenting since the number of tokens affects the run time dramatically.
if TRAINING_USE_WORDS:
    INFERENCE_SAMPLE_SIZE = 100
    INFERENCE_FREQUENCY = 10


# hyperparameters
TRAINING_HIDDEN_SIZE: int = 128  # size of hidden layer of neurons
TRAINING_SEQUENCE_LENGTH: int = 50  # number of steps to unroll the RNN for
TRAINING_LEARNING_RATE: float = 1e-1


# type aliases for readability
IntList = list[int]
StringList = list[str]
FloatArray = npt.NDArray[np.float64]


class InputData:
    def __init__(
        self,
        input_file_path: str,
        training_sequence_length: int,
        training_use_words: bool,
        inference_starting_text: str,
    ):
        """
        Handles reading text data from a file and then accessing it in sequence.
        """

        # initialize sequences
        self.sequence_length = training_sequence_length
        self.current_pos: int = 0

        # read data from file
        data: str = open(input_file_path, "r").read()

        # store our token information
        self.tokens: StringList = self.tokenize(data, training_use_words)
        self.data_size: int = len(self.tokens)

        unique_tokens: StringList = list(set(self.tokens))
        self.vocab_size: int = len(unique_tokens)

        print(f"input data has {self.data_size} tokens ({self.vocab_size} unique)")

        # create token <-> index mappings
        self.token_to_index: dict[str, int] = {
            ch: i for i, ch in enumerate(unique_tokens)
        }
        self.index_to_token: dict[int, str] = {
            i: ch for i, ch in enumerate(unique_tokens)
        }

        # lookup and store our starting text indices
        self.inference_start_indices: IntList = []

        start_tokens: StringList = self.tokenize(
            inference_starting_text, training_use_words
        )

        for i in range(len(start_tokens)):
            index: int = self.token_to_index[start_tokens[i]]
            self.inference_start_indices.append(index)

    def nextInputsAndTargets(self) -> tuple[IntList, IntList]:
        """
        Prepare inputs (we're sweeping from left to right in steps seq_length long)
        """
        inputs: IntList = [
            self.token_to_index[ch]
            for ch in self.tokens[
                self.current_pos : self.current_pos + self.sequence_length
            ]
        ]

        targets: IntList = [
            self.token_to_index[ch]
            for ch in self.tokens[
                self.current_pos + 1 : self.current_pos + self.sequence_length + 1
            ]
        ]

        self.current_pos += self.sequence_length

        # if we don't have enough in our data for another chunk, wrap around to the beginning
        if self.current_pos + self.sequence_length + 1 > self.data_size:
            self.current_pos = 0

        return inputs, targets

    def atStart(self) -> bool:
        return self.current_pos == 0

    def tokenize(self, text: str, use_words: bool) -> StringList:
        if use_words:
            return text.split() + [" ", "\n"]
        return list(text)


class RNN:
    def __init__(self, vocab_size: int, hidden_size: int, learning_rate: float):
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        # model parameters
        # W = weight
        # b = bias
        # x = input
        # h = hidden
        # y = output

        # input to hidden
        self.Wxh: FloatArray = np.random.randn(hidden_size, self.vocab_size) * 0.01
        # hidden to hidden
        self.Whh: FloatArray = np.random.randn(hidden_size, hidden_size) * 0.01
        # hidden to output
        self.Why: FloatArray = np.random.randn(self.vocab_size, hidden_size) * 0.01
        # hidden bias
        self.bh: FloatArray = np.zeros((hidden_size, 1))
        # output bias
        self.by: FloatArray = np.zeros((self.vocab_size, 1))

        # memory variables for Adaptive Gradient Algorithm (AdaGrad)
        self.mWxh: FloatArray = np.zeros_like(self.Wxh)
        self.mWhh: FloatArray = np.zeros_like(self.Whh)
        self.mWhy: FloatArray = np.zeros_like(self.Why)
        self.mbh: FloatArray = np.zeros_like(self.bh)
        self.mby: FloatArray = np.zeros_like(self.by)

    def train(
        self,
        inputs: IntList,
        targets: IntList,
        vocab_size: int,
        hidden_initial: FloatArray,
    ) -> tuple[
        float, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
    ]:
        """
        hidden_initial is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """

        num_inputs: int = len(inputs)

        # encode in 1-of-k representation
        xs: list[FloatArray] = [np.zeros((vocab_size, 1)) for _ in range(num_inputs)]

        ps: list[FloatArray] = [np.empty([]) for _ in range(num_inputs)]
        ys: list[FloatArray] = [np.empty([]) for _ in range(num_inputs)]
        hs: list[FloatArray] = [np.empty([]) for _ in range(num_inputs)]
        hs[-1] = np.copy(hidden_initial)

        loss: float = 0.0

        # forward pass
        for t in range(num_inputs):
            xs[t][inputs[t]] = 1.0

            # hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            )

            # unnormalized log probabilities for next chars
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # probabilities for next chars
            exp_yst: FloatArray = np.exp(ys[t])
            ps[t] = exp_yst / np.sum(exp_yst)

            # softmax (cross-entropy loss)
            loss += -np.log(ps[t][targets[t], 0])

        # backward pass: compute gradients going backwards
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(num_inputs)):
            dy = np.copy(ps[t])

            # backprop into y
            # see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dy[targets[t]] -= 1

            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw

            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[num_inputs - 1]

    def update(
        self,
        dWxh: FloatArray,
        dWhh: FloatArray,
        dWhy: FloatArray,
        dbh: FloatArray,
        dby: FloatArray,
    ):
        """
        Update parameters using Adaptive Gradient Algorithm (AdaGrad)
        """
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def infer(
        self,
        memory: FloatArray,
        vocab_size: int,
        inference_start_text_indices: IntList,
        inference_sample_size: int,
    ) -> IntList:
        """
        sample a sequence of integers from the model
        memory is memory state, inference_start_text_indices is list of token indices for the starting text
        """

        x: FloatArray = np.zeros((self.vocab_size, 1))

        # initialize with our starting text's indices
        indices: IntList = inference_start_text_indices.copy()
        for i in indices:
            x[i] = 1.0

        for _ in range(inference_sample_size):
            memory = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, memory) + self.bh)
            y: FloatArray = np.dot(self.Why, memory) + self.by
            expy: FloatArray = np.exp(y)
            p: FloatArray = expy / np.sum(expy)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1.0
            indices.append(ix)

        return indices


if __name__ == "__main__":
    data = InputData(
        INPUT_FILE,
        TRAINING_SEQUENCE_LENGTH,
        TRAINING_USE_WORDS,
        INFERENCE_STARTING_TEXT,
    )
    rnn = RNN(data.vocab_size, TRAINING_HIDDEN_SIZE, TRAINING_LEARNING_RATE)

    # init RNN memory
    memory: FloatArray = np.zeros((TRAINING_HIDDEN_SIZE, 1))

    # loss at iteration 0
    smooth_loss: float = -np.log(1.0 / data.vocab_size) * TRAINING_SEQUENCE_LENGTH

    iteration_number: int = 0

    while True:
        if data.atStart():
            memory.fill(0.0)

        inputs, targets = data.nextInputsAndTargets()

        # infer a sample from the model now and then
        if iteration_number % INFERENCE_FREQUENCY == 0:
            indices = rnn.infer(
                memory,
                data.vocab_size,
                data.inference_start_indices,
                INFERENCE_SAMPLE_SIZE,
            )

            tokens: StringList = [data.index_to_token[ix] for ix in indices]
            txt: str = ""
            if TRAINING_USE_WORDS:
                txt = " ".join(tokens)
            else:
                txt = "".join(tokens)

            print(f"---- iteration {iteration_number}\n{txt}")

        # forward TRAINING_SEQUENCE_LENGTH tokens through the net and fetch gradient
        loss, dWxh, dWhh, dWhy, dbh, dby, memory = rnn.train(
            inputs, targets, data.vocab_size, memory
        )
        smooth_loss = smooth_loss * 0.999 + (loss * 0.001)

        if iteration_number % INFERENCE_FREQUENCY == 0:
            print(f"---- iteration {iteration_number}; loss: {smooth_loss}\n")

        # update the model
        rnn.update(dWxh, dWhh, dWhy, dbh, dby)

        iteration_number += 1
