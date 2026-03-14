# The Architecture

Since we have now understood the underlying concepts behind what we're doing with this system and its intent, let's deep dive into the architecture itself so we can understand individual stages in detail.

## Data pre-processing

This is the initial and most crucial stage of this system which involves pulling a massive corpus of data sitting on [Hugging Face](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) into locally cached, tokenizable, batchable training data. Everything necessary for this data setup can be found in the `prepare.py` file, which contains 2 major personalities: the initial setup itself and the runtime utility module.

The setup script is primarily focused on downloading the dataset shards from Hugging Face and training the tokenizer. There is also one shard of data among the dataset that is pinned for evaluation and is never used for training. This shard is currently being referenced by `VAL_SHARD` in the codebase. Once the data is pulled, it is stored locally on disk and is then referenced later in the tokenizer training phase.

Karpathy here uses a BPE (Byte Pair Encoding) tokenizer which, in simple terms, repeatedly merges common adjacent byte or token pairs into larger reusable units. You can read more about BPE tokenizers [here](https://huggingface.co/learn/llm-course/en/chapter6/5). One might ask why a tokenizer is trained here instead of using a pre-trained one. The answer is that, in this scenario, the repo requires a self-contained setup and this tokenizer training loop also matches well with the bits-per-byte style evaluation which we will come to later. This stage creates a `tokenizer.pkl` and a `token_bytes.pt`. The first stores the tokenizer itself, while the second stores the UTF-8 byte length for each token ID, which is then used during evaluation.

A subtle caveat to understand here is that there are a few standard constants defined in the preparation phase. These include `MAX_SEQ_LEN`, `TIME_BUDGET` and `EVAL_TOKENS`. These are the fixed rules of the game where:

1. `MAX_SEQ_LEN` is the maximum number of tokens the model sees in one sequence at a time. This defines the context window for the system. Bigger sequence lengths mean more context is visible to the model, but they also demand more memory and compute. This is currently set to 2048 tokens.
2. `TIME_BUDGET` is the training duration for the model itself. Post this time limit, an evaluation run is conducted on the model to understand its current status. The goal of this is to keep experiments fair and directly comparable on the same machine. This is currently set to 5 minutes of wall clock time.
3. `EVAL_TOKENS` is the total number of tokens used in validation. This matters because if the evaluation dataset was too small, the results would be noisy, and if it was too large, then the evaluation stage itself would be too slow.

So these three systems act as the benchmark factors or limiters (***they are not the benchmark itself***).

Also keep in mind that language models **cannot** read raw strings, so encoders and decoders are used to convert text into token IDs. In this case, a `tiktoken` encoder-decoder is used. You can play around with `tiktoken` [here](https://github.com/openai/tiktoken).

## Model training

Now coming to the core section of this project, we have the model training phase. This section loads the tokenizer and the prepared data, builds a GPT style model on the fly, feeds it batches of tokenized text, makes it predict the next token, measures how wrong that prediction was, and then updates the weights to reduce that error. This loop is repeated for a total of 5 minutes and then a validation pipeline is run on it to print the final score.

This phase is seen in the `train.py` file where the training process begins with the initial model configuration such as the number of layers, embedding size, attention layout and the other core hyperparameters. These values together decide the overall model shape. One important thing to understand here is that this repo is not loading a pre-trained GPT model from somewhere else. It is building a custom GPT style model from scratch every time and then training it within a fixed wall clock budget.

Once the shape of the model is decided, the model is initialized and moved onto the GPU. The tokenizer built earlier is also loaded at this point, because the training loop does not work with raw text directly. The prepared data is read in batches, converted into token IDs and then passed through the model as input-target pairs. In simple terms, the model keeps seeing text and is repeatedly asked one question: given everything you have seen so far, what token should come next?

The training loop then runs continuously. For every batch, the model makes a prediction, calculates the loss, backpropagates that loss, and then updates its parameters using the optimizer. In this repo, the optimizer being used is **MuonAdamW**, which is a custom optimizer setup that combines Muon style updates for matrix parameters and AdamW style updates for the rest. This is important because in autoresearch the optimizer itself is part of the research space and not just a background implementation detail. You can read more about this optimizer [here](https://huggingface.co/blog/KingNish/optimizer-part1).

After the 5 minute training loop finishes, the model is switched into evaluation mode and is scored using the fixed validation metric defined in `prepare.py`, which is `val_bpb` or validation bits per byte. This is the main score the repo cares about. Lower values are better. This final score is what allows the agent to decide whether a code change in `train.py` was worth keeping or not.

## Evaluation

The evaluation phase is the final checkpoint of every experiment. Once training is complete, the repo does not ask a human whether the model looks better or worse. Instead, it runs the model against a fixed validation pipeline defined in `prepare.py` and produces a single score that can be compared across runs.

The metric being used here is `val_bpb`, which stands for validation bits per byte. In simple terms, this measures how efficiently the model predicts the held out validation text. Lower values are better. This is an important design choice because it makes comparisons more stable even if tokenization details or vocabulary choices differ. So instead of asking "how good did the loss look?", the system asks "how well did the model compress unseen text in terms of bits per byte?"

The validation data itself comes from the pinned shard discussed earlier through `VAL_SHARD`. This shard is never used during training, which makes it a clean held out reference point. During evaluation, batches are built from this validation split, passed through the model, and the prediction errors are accumulated across a fixed amount of validation tokens controlled by `EVAL_TOKENS`.

Another important point here is that the evaluation logic is intentionally fixed and is not meant to be modified during experiments. This keeps the benchmark trustworthy. In other words, the agent is free to change the model, the optimizer, the architecture and the training behavior, but it should not change the exam itself.

At the end of the run, this evaluation stage prints the final `val_bpb` score along with other useful signals such as training time, peak memory usage, number of steps, and total tokens processed. But among all of these outputs, `val_bpb` is the main number that decides whether an experiment was a success or not.
