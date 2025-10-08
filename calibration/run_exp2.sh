printf "\n\nAvg-Conf\n"
python run.py --model qwen7 --eval_method avg-conf
printf "\n\nPair-Rank with Demo\n"
python run.py --model qwen7 --eval_method pair-rank

# printf "\n\nAvg-Conf with Demo\n"
# python run.py --model qwen7 --add_demo --eval_method avg-conf
# printf "\n\nAvg-Conf with Traits\n"
# python run.py --model qwen7 --add_traits --eval_method avg-conf
# printf "\n\nAvg-Conf with Demo and Traits\n"
# python run.py --model qwen7 --add_demo --add_traits --eval_method avg-conf
# 
# printf "\n\nPair-Rank with Demo\n"
# python run.py --model qwen7 --add_demo --eval_method pair-rank
# printf "\n\nPair-Rank with Traits\n"
# python run.py --model qwen7 --add_traits --eval_method pair-rank
# printf "\n\nPair-Rank with Demo and Traits\n"
# python run.py --model qwen7 --add_demo --add_traits --eval_method pair-rank
