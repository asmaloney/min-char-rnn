# min-char-rnn

This is a fork of [Andrej Karpathy's short gist](https://gist.github.com/karpathy/d4dee566867f8291f086) which implements a minimal character-level recurrent neural network (RNN).

To try to understand the code better, I've made the following changes:

- fixed for Python 3
- reorganized the code using classes
- added type hints
- renamed some variables for clarity
- added `STARTING_TEXT` string instead of just starting with one character
- added constants at the top for easy modification (see below)
- added a way to use words instead of characters

Since it isn't a complete command-line utility (not the goal here), there are several constants you can modify at the top of the code:

- `INPUT_FILE`: the text it's using as its source
- `STARTING_TEXT`: the text to start off with (**Note:** the words in this text need to be in the input file)
- `SAMPLE_SIZE`: how many characters to output
- `SAMPLE_OUTPUT_FREQ`: output some sample text after this many iterations
- `USE_WORDS`: use words instead of tokens (see below)
- `HIDDEN_SIZE`: size of hidden layer of neurons
- `SEQUENCE_LENGTH`: number of steps to unroll the RNN for
- `LEARNING_RATE`: the learning rate

The data directory contains some public domain text to try including Hans Christian Andersen, Shakespeare (large and small), and Sherlock Holmes. Change `INPUT_FILE` to point at one of these (and modify the `STARTING_TEXT` to make more sense contextually)

## Words vs. Characters

I made some changes to allow the use of words instead of just characters. For the version that is strictly characters, see the [min-char-rnn](https://github.com/asmaloney/min-char-rnn/tree/min-char-rnn) tag.

To use words, set `USE_WORDS` to `True`. It will automatically adjust some settings, but you may need to play with them a bit more to get results.

Right now, the tokenization is very naive which affects the output. Eventually I will try a smarter tokenizer.

## Running
To run it:
```sh
python3 min-char-rnn.py
```

This will continue to output text until you stop it with <kbd>Control</kbd>-<kbd>C</kbd>.

## Example Output
```
% python3 min-char-rnn.py
input data has 4573338 characters (67 unique)
---- iteration 0
I was anointed king at nine months old.
,!OT[:hIP,Fv$EOlzTQWf-r:S'mXWc:;&''qKTOL-MB-ZFp'nkAYgba-koQW?OPQfQSZU.Ls!le];EhjsK&Eh3 Q$ZLizZ]ishpBAafSzX w$Ib IMtJSiDrzadcc[ldOj:l:pPLGu?-YuTv.I$JKZmcUTeqo$Grwd-LaTc
KjY&.-bOc?m]AEu3ioS;khgsMLit.pZGIatdvdWU
RDGQT:HReL'UzBEAe;$sWb-[lLIU?DeESp& TkMuTUibJ:X'CZ.Au3EuufbY]zN$kQgWqtjzA?!iO!QCIP
COBY;NBK
---- iteration 0; loss: 210.23463013700214

---- iteration 1000
I was anointed king at nine months old.
 honUMp ther Wornof hethof, cheflin

ofy:

bid the
INIMthae harRlmolh hasy pe hatouk hom kort
Ahend hed ow bh: phe
Safre ar
mande
t Chhhenke
chend oINler far th
FIUau thyhethe Lf hirat thi
Murt be WhlnTh:

dthe torrthomce thhh, The the het amatdeatmaaLUiod bankd othealat me her Thevlltherghau hot:
T
---- iteration 1000; loss: 178.5383480403174

...

---- iteration 6544000
I was anointed king at nine months old.
e's my wours.
Well not the seem, euce
Satch not from your that to'gle the will,
The him and tragoute wheter suttsle be?

PARISA:
Sick, shall wold,
But of our shign what his of the bearst he bavest freeg cord eaginester seppemion did:
Ay to but the may, bigst neather
Wetter demenazeds,
But you
Besies
---- iteration 6544000; loss: 85.80065445375027
```

So far it hsan't produced anything entirely in English for me... 😀