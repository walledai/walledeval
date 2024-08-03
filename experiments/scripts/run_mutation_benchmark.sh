echo "Running in $(pwd)"

echo "Beginning llama3.1-8b on harmbench" 
python experiments/mutation_benchmarks.py -m llama3.1-8b -d harmbench --verbose > ./experiments/runs/llama3.1-8b-mutated-harmbench.log 2> ./experiments/runs/llama3.1-8b-mutated-harmbench.log
echo "Ended llama3.1-8b on harmbench" 

echo "Beginning llama3-8b on harmbench" 
python experiments/mutation_benchmarks.py -m llama3-8b -d harmbench --verbose > ./experiments/runs/llama3-8b-mutated-harmbench.log 2> ./experiments/runs/llama3-8b-mutated-harmbench.log
echo "Ended llama3-8b on harmbench" 

echo "Beginning llama2-7b on harmbench" 
python experiments/mutation_benchmarks.py -m llama2-7b -d harmbench --verbose > ./experiments/runs/llama2-7b-mutated-harmbench.log 2> ./experiments/runs/llama2-7b-mutated-harmbench.log
echo "Ended llama2-7b on harmbench" 

echo "Beginning gemma2-9b on harmbench" 
python experiments/mutation_benchmarks.py -m gemma2-9b -d harmbench --verbose > ./experiments/runs/gemma2-9b-mutated-harmbench.log 2> ./experiments/runs/gemma2-9b-mutated-harmbench.log
echo "Ended gemma2-9b on harmbench" 

echo "Beginning gemma-1.1-7b on harmbench" 
python experiments/mutation_benchmarks.py -m gemma-1.1-7b -d harmbench --verbose > ./experiments/runs/gemma-1.1-7b-mutated-harmbench.log 2> ./experiments/runs/gemma-1.1-7b-mutated-harmbench.log
echo "Ended gemma-1.1-7b on harmbench" 

echo "Beginning gemma-7b on harmbench" 
python experiments/mutation_benchmarks.py -m gemma-7b -d harmbench --verbose > ./experiments/runs/gemma-7b-mutated-harmbench.log 2> ./experiments/runs/gemma-7b-mutated-harmbench.log
echo "Ended gemma-7b on harmbench" 

echo "Beginning mistral-nemo-12b on harmbench" 
python experiments/mutation_benchmarks.py -m mistral-nemo-12b -d harmbench --verbose > ./experiments/runs/mistral-nemo-12b-mutated-harmbench.log 2> ./experiments/runs/mistral-nemo-12b-mutated-harmbench.log
echo "Ended mistral-nemo-12b on harmbench" 

echo "Beginning mistral-7b on harmbench" 
python experiments/mutation_benchmarks.py -m mistral-7b -d harmbench --verbose > ./experiments/runs/mistral-7b-mutated-harmbench.log 2> ./experiments/runs/mistral-7b-mutated-harmbench.log
echo "Ended mistral-7b on harmbench" 

echo "Beginning mixtral-8x7b on harmbench" 
python experiments/mutation_benchmarks.py -m mixtral-8x7b -d harmbench --verbose > ./experiments/runs/mixtral-8x7b-mutated-harmbench.log 2> ./experiments/runs/mixtral-8x7b-mutated-harmbench.log
echo "Ended mixtral-8x7b on harmbench" 

echo "Beginning phi3-mini on harmbench" 
python experiments/mutation_benchmarks.py -m phi3-mini -d harmbench --verbose > ./experiments/runs/phi3-mini-mutated-harmbench.log 2> ./experiments/runs/phi3-mini-mutated-harmbench.log
echo "Ended phi3-mini on harmbench" 

echo "Beginning qwen2-7b on harmbench" 
python experiments/mutation_benchmarks.py -m qwen2-7b -d harmbench --verbose > ./experiments/runs/qwen2-7b-mutated-harmbench.log 2> ./experiments/runs/qwen2-7b-mutated-harmbench.log
echo "Ended qwen2-7b on harmbench" 

echo "Beginning qwen2-1.5b on harmbench" 
python experiments/mutation_benchmarks.py -m qwen2-1.5b -d harmbench --verbose > ./experiments/runs/qwen2-1.5b-mutated-harmbench.log 2> ./experiments/runs/qwen2-1.5b-mutated-harmbench.log
echo "Ended qwen2-1.5b on harmbench" 

echo "Beginning qwen2-0.5b on harmbench" 
python experiments/mutation_benchmarks.py -m qwen2-0.5b -d harmbench --verbose > ./experiments/runs/qwen2-0.5b-mutated-harmbench.log 2> ./experiments/runs/qwen2-0.5b-mutated-harmbench.log
echo "Ended qwen2-0.5b on harmbench" 


echo "Beginning llama3.1-8b on xstest" 
python experiments/mutation_benchmarks.py -m llama3.1-8b -d xstest --verbose > ./experiments/runs/llama3.1-8b-mutated-xstest.log 2> ./experiments/runs/llama3.1-8b-mutated-xstest.log
echo "Ended llama3.1-8b on xstest" 

echo "Beginning llama3-8b on xstest" 
python experiments/mutation_benchmarks.py -m llama3-8b -d xstest --verbose > ./experiments/runs/llama3-8b-mutated-xstest.log 2> ./experiments/runs/llama3-8b-mutated-xstest.log
echo "Ended llama3-8b on xstest" 

echo "Beginning llama2-7b on xstest" 
python experiments/mutation_benchmarks.py -m llama2-7b -d xstest --verbose > ./experiments/runs/llama2-7b-mutated-xstest.log 2> ./experiments/runs/llama2-7b-mutated-xstest.log
echo "Ended llama2-7b on xstest" 

echo "Beginning gemma2-9b on xstest" 
python experiments/mutation_benchmarks.py -m gemma2-9b -d xstest --verbose > ./experiments/runs/gemma2-9b-mutated-xstest.log 2> ./experiments/runs/gemma2-9b-mutated-xstest.log
echo "Ended gemma2-9b on xstest" 

echo "Beginning gemma-1.1-7b on xstest" 
python experiments/mutation_benchmarks.py -m gemma-1.1-7b -d xstest --verbose > ./experiments/runs/gemma-1.1-7b-mutated-xstest.log 2> ./experiments/runs/gemma-1.1-7b-mutated-xstest.log
echo "Ended gemma-1.1-7b on xstest" 

echo "Beginning gemma-7b on xstest" 
python experiments/mutation_benchmarks.py -m gemma-7b -d xstest --verbose > ./experiments/runs/gemma-7b-mutated-xstest.log 2> ./experiments/runs/gemma-7b-mutated-xstest.log
echo "Ended gemma-7b on xstest" 

echo "Beginning mistral-nemo-12b on xstest" 
python experiments/mutation_benchmarks.py -m mistral-nemo-12b -d xstest --verbose > ./experiments/runs/mistral-nemo-12b-mutated-xstest.log 2> ./experiments/runs/mistral-nemo-12b-mutated-xstest.log
echo "Ended mistral-nemo-12b on xstest" 

echo "Beginning mistral-7b on xstest" 
python experiments/mutation_benchmarks.py -m mistral-7b -d xstest --verbose > ./experiments/runs/mistral-7b-mutated-xstest.log 2> ./experiments/runs/mistral-7b-mutated-xstest.log
echo "Ended mistral-7b on xstest" 

echo "Beginning mixtral-8x7b on xstest" 
python experiments/mutation_benchmarks.py -m mixtral-8x7b -d xstest --verbose > ./experiments/runs/mixtral-8x7b-mutated-xstest.log 2> ./experiments/runs/mixtral-8x7b-mutated-xstest.log
echo "Ended mixtral-8x7b on xstest" 

echo "Beginning phi3-mini on xstest" 
python experiments/mutation_benchmarks.py -m phi3-mini -d xstest --verbose > ./experiments/runs/phi3-mini-mutated-xstest.log 2> ./experiments/runs/phi3-mini-mutated-xstest.log
echo "Ended phi3-mini on xstest" 

echo "Beginning qwen2-7b on xstest" 
python experiments/mutation_benchmarks.py -m qwen2-7b -d xstest --verbose > ./experiments/runs/qwen2-7b-mutated-xstest.log 2> ./experiments/runs/qwen2-7b-mutated-xstest.log
echo "Ended qwen2-7b on xstest" 

echo "Beginning qwen2-1.5b on xstest" 
python experiments/mutation_benchmarks.py -m qwen2-1.5b -d xstest --verbose > ./experiments/runs/qwen2-1.5b-mutated-xstest.log 2> ./experiments/runs/qwen2-1.5b-mutated-xstest.log
echo "Ended qwen2-1.5b on xstest" 

echo "Beginning qwen2-0.5b on xstest" 
python experiments/mutation_benchmarks.py -m qwen2-0.5b -d xstest --verbose > ./experiments/runs/qwen2-0.5b-mutated-xstest.log 2> ./experiments/runs/qwen2-0.5b-mutated-xstest.log
echo "Ended qwen2-0.5b on xstest" 
