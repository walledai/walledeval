echo "Running in $(pwd)"

# echo "Beginning Llama 3.1 8B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m llama3.1-8b --verbose > ./experiments/runs/llama3.1-8b-refusal.log 2> ./experiments/runs/llama3.1-8b-refusal.log
# echo "Beginning Llama 3 8B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m llama3-8b --verbose > ./experiments/runs/llama3-8b-refusal.log 2> ./experiments/runs/llama3-8b-refusal.log
# echo "Beginning Llama 2 7B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m llama2-7b --verbose > ./experiments/runs/llama2-7b-refusal.log 2> ./experiments/runs/llama2-7b-refusal.log
# echo "Refusal Experiments on Llama Completed"


# echo "Beginning Qwen 2 0.5B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m qwen2-0.5b --verbose > ./experiments/runs/qwen2-0.5b-refusal.log 2> ./experiments/runs/qwen2-0.5b-refusal.log
# echo "Refusal Experiments on Qwen 2 0.5B Completed"

# echo "Beginning Qwen 2 1.5B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m qwen2-1.5b --verbose > ./experiments/runs/qwen2-1.5b-refusal.log 2> ./experiments/runs/qwen2-1.5b-refusal.log
# echo "Refusal Experiments on Qwen 2 1.5B Completed"

# echo "Beginning Phi 3 Mini on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m phi3-mini --verbose > ./experiments/runs/phi3-mini-refusal.log 2> ./experiments/runs/phi3-mini-refusal.log
# echo "Refusal Experiments on Phi 3 Mini Completed"

# echo "Beginning Mistral 7B on XSTest x MCQ"
# python experiments/refusal_benchmarks.py -m mistral-7b --verbose > ./experiments/runs/mistral-7b-refusal.log 2> ./experiments/runs/mistral-7b-refusal.log
# echo "Refusal Experiments on Mistral 7B Completed"

echo "Beginning Gemma 7B on XSTest x MCQ"
python experiments/refusal_benchmarks.py -m gemma-7b --verbose > ./experiments/runs/gemma-7b-refusal.log 2> ./experiments/runs/gemma-7b-refusal.log
echo "Refusal Experiments on Gemma 7B Completed"