echo "Beginning Mistral 7B v0.3 on HarmBench"
python experiments/prompt_benchmarks.py -m mistral-7b -d harmbench --verbose > mistral-7b-harmbench.log 2> mistral-7b-harmbench.log
echo "Beginning Mistral 7B v0.3 on AdvBench"
python experiments/prompt_benchmarks.py -m mistral-7b -d advbench --verbose > mistral-7b-advbench.log 2> mistral-7b-advbench.log
