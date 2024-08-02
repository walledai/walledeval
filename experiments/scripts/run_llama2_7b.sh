echo "Running in $(pwd)"

echo "Beginning Llama 2 7B on HarmBench"
python experiments/prompt_benchmarks.py -m llama2-7b -d harmbench --verbose > ./experiments/runs/llama2-7b-harmbench.log 2> ./experiments/runs/llama2-7b-harmbench.log
echo "Beginning Llama 2 7B on AdvBench"
python experiments/prompt_benchmarks.py -m llama2-7b -d advbench --verbose > ./experiments/runs/llama2-7b-advbench.log 2> ./experiments/runs/llama2-7b-advbench.log
echo "Experiments on Llama 2 7B Completed"