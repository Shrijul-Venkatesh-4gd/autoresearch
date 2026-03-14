# What you need to know

Andrej Karpathy's autoresearch is, in layman's terms, a "tiny robot scientist" that helps you train an SLM; which is in this case, a custom GPT style model that is being trained from scratch. He argues that model training is often just a series of continuous experiments aimed at improving a final metric, and in this repo that learning process can be monitored by an AI agent. The model is not judged by a human during training; it is judged automatically by a fixed validation metric, while the human mainly designs the agent instructions in `program.md`.

## What does this imply in the future?

If in this scenario, an agent can truly govern the learning process of an SLM (or possibly in the future an LLM), then the all the researcher needs to do would be to modify the training/learning curve of the agent itself by just altering the `program.md`instruction set that will be given to the agent. This will ultimately lead to a 'self-evolving model' system where the agent, which is capable of governance will soon be able to govern itself given that it has the ability to pair this entire system with its own memory. So as the agent learns and updates its own memory with time, so will its ability to alter its own `program.md` thereby bringing in a concept of '**self-learning**'.

## A simple introduction to the architecture

Here is what the flow of the current system looks like :

1. *Preparing the [dataset](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)* : This phase downloads a massive text corpus from Hugging Face. The dataset already exists as parquet shards, and `prepare.py` downloads those shards and caches them locally on disk. It then builds a tokenizer which is needed in the training phase.
2. *The training phase* : This stage loads the tokenizer built in the previous stage and then builds the GPT like model. It reads batches of text from the prepared data, trains the model for 5 minutes exactly, and then evaluates it on a fixed validation metric. After that, an external agent such as Codex or Claude can update the training code and try again.

And thats it. The entire system is coded to achieve a very simple goal, train the model towards a certain benchmarking without any human intervention.


---



If you feel comfortable with your current understanding of this system you can move forward with the next stage and deep dive into this system.
