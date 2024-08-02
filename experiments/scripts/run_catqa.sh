echo "Running in $(pwd)"

echo "Beginning Llama 3.1 8B on CatQA"
python experiments/prompt_benchmarks.py -m llama3.1-8b -d catqa --verbose > ./experiments/runs/llama3.1-8b-catqa.log 2> ./experiments/runs/llama3.1-8b-catqa.log
echo "Beginning Llama 3 8B on CatQA"
python experiments/prompt_benchmarks.py -m llama3-8b -d catqa --verbose > ./experiments/runs/llama3-8b-catqa.log 2> ./experiments/runs/llama3-8b-catqa.log
echo "Beginning Llama 2 7B on CatQA"
python experiments/prompt_benchmarks.py -m llama2-7b -d catqa --verbose > ./experiments/runs/llama2-7b-catqa.log 2> ./experiments/runs/llama2-7b-catqa.log
echo "Experiments on Llama Completed"

# echo "Beginning Qwen 2 0.5B on CatQA"
# python experiments/prompt_benchmarks.py -m qwen2-0.5b -d catqa --verbose > ./experiments/runs/qwen2-0.5b-catqa.log 2> ./experiments/runs/qwen2-0.5b-catqa.log
# echo "Beginning Qwen 2 1.5B on CatQA"
# python experiments/prompt_benchmarks.py -m qwen2-1.5b -d catqa --verbose > ./experiments/runs/qwen2-1.5b-catqa.log 2> ./experiments/runs/qwen2-1.5b-catqa.log
# echo "Beginning Qwen 2 7B on CatQA"
# python experiments/prompt_benchmarks.py -m qwen2-7b -d catqa --verbose > ./experiments/runs/qwen2-7b-catqa.log 2> ./experiments/runs/qwen2-7b-catqa.log
# echo "Experiments on Qwen 2 Completed"

# echo "Beginning Phi 3 Mini on CatQA"
# python experiments/prompt_benchmarks.py -m phi3-mini -d catqa --verbose > ./experiments/runs/phi3-mini-catqa.log 2> ./experiments/runs/phi3-mini-catqa.log
# echo "Experiments on Phi 3 Mini Completed"

# echo "Beginning Mistral Nemo 12B on CatQA"
# python experiments/prompt_benchmarks.py -m mistral-nemo-12b -d catqa --verbose > ./experiments/runs/mistral-nemo-12b-catqa.log 2> ./experiments/runs/mistral-nemo-12b-catqa.log
# echo "Beginning Mixtral 8x7B on CatQA"
# python experiments/prompt_benchmarks.py -m mistral-8x7b -d catqa --verbose > ./experiments/runs/mistral-8x7b-catqa.log 2> ./experiments/runs/mixtral-8x7b-catqa.log
# echo "Beginning Mistral 7B on CatQA"
# python experiments/prompt_benchmarks.py -m mistral-7b -d catqa --verbose > ./experiments/runs/mistral-7b-catqa.log 2> ./experiments/runs/mistral-7b-catqa.log
# echo "Experiments on Mistral Completed"

# echo "Beginning Gemma 2 9B on CatQA"
# python experiments/prompt_benchmarks.py -m gemma2-9b -d catqa --verbose > ./experiments/runs/gemma2-9b-catqa.log 2> ./experiments/runs/gemma2-9b-catqa.log
# echo "Beginning Gemma 1.1 7B on CatQA"
# python experiments/prompt_benchmarks.py -m gemma-1.1-7b -d catqa --verbose > ./experiments/runs/gemma-1.1-7b-catqa.log 2> ./experiments/runs/gemma-1.1-7b-catqa.log
# echo "Beginning Gemma 7B on CatQA"
# python experiments/prompt_benchmarks.py -m gemma-7b -d catqa --verbose > ./experiments/runs/gemma-7b-catqa.log 2> ./experiments/runs/gemma-7b-catqa.log
# echo "Experiments on Gemma Completed"