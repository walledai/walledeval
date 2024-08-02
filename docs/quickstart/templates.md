# Prompt Template Database

Beyond general 

With the advent of LLMs being used for mutation, inference and judges, prompt templates have become a recurring feature in various parts of the general evaluation framework. Sadly, since [`promptsource`](https://github.com/bigscience-workshop/promptsource), there hasn't been much effort made in compiling a large variety of prompt templates in one centralised system. This is where our **prompt database** comes in! WalledEval compiles prompts from many, many papers in prior literature to consolidate a huge database to choose from.

WalledEval strives to provide a method to build easy-to-use templates for researchers and testers alike to use for all kinds of different tasks. Whether it be a template to [automatically mutate prompts](auto-red-teaming.md) or to [prompt LLMs to perform like judges](judges.md), prompt templates take up a major portion of 


Beyond just loading data, our libray provides methods to load adversarial Prompt Templates like [DAN](https://github.com/verazuo/jailbreak_llms) and [DeepInception](https://github.com/tmlr-group/DeepInception). The team of WalledEval has compiled an extensive dataset of Prompt Templates from several papers, datasets and codebases, with more to come. We hope to use this to standardise a potential practice of keeping strings out of the codebase.