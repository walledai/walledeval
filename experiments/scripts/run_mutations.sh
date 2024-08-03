echo "$(pwd)"

echo "Starting Generation on HarmBench with Mistral"
python experiments/mutation_generation.py -d harmbench -m mistral-7b -v > runs/mutations-harmbench-mistral.log 2> runs/mutations-harmbench-mistral.log
echo "Starting Generation on HarmBench with Llama 2"
python experiments/mutation_generation.py -d harmbench -m llama2-7b-uncensored -v > runs/mutations-harmbench-llama2.log 2> runs/mutations-harmbench-llama2.log

echo "Starting Generation on XSTest with Mistral"
python experiments/mutation_generation.py -d xstest -m mistral-7b -v > runs/mutations-xstest-mistral.log 2> runs/mutations-xstest-mistral.log
echo "Starting Generation on XSTest with Llama 2"
python experiments/mutation_generation.py -d xstest -m llama2-7b-uncensored -v > runs/mutations-xstest-llama2.log 2> runs/mutations-xstest-llama2.log

echo "Finished Generation"