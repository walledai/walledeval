echo "Beginning Qwen 2 0.5B on HarmBench"
python experiments/prompt_benchmarks.py -m qwen2-0.5b -d harmbench --verbose > qwen2-0.5b-harmbench.log 2> qwen2-0.5b-harmbench.log

echo "Beginning Qwen 2 0.5B on AdvBench"
python experiments/prompt_benchmarks.py -m qwen2-0.5b -d advbench --verbose > qwen2-0.5b-advbench.log 2> qwen2-0.5b-advbench.log

echo "Beginning Qwen 2 1.5B on HarmBench"
python experiments/prompt_benchmarks.py -m qwen2-1.5b -d harmbench --verbose > qwen2-1.5b-harmbench.log 2> qwen2-1.5b-harmbench.log
echo "Beginning Qwen 2 1.5B on AdvBench"
python experiments/prompt_benchmarks.py -m qwen2-1.5b -d advbench --verbose > qwen2-1.5b-advbench.log 2> qwen2-1.5b-advbench.log

echo "Beginning Yi 1.5 6B on HarmBench"
python experiments/prompt_benchmarks.py -m yi-1.5-6b -d harmbench --verbose > yi-1.5-6b-harmbench.log 2> yi-1.5-6b-harmbench.log
echo "Beginning Yi 1.5 6B on AdvBench"
python experiments/prompt_benchmarks.py -m yi-1.5-6b -d advbench --verbose > yi-1.5-6b-advbench.log 2> yi-1.5-6b-advbench.log