python mdpfuzz.py
# python test_gen.py --hour 1 --method generative 
# python test_gen.py --hour 1 --method generative+density 
# python test_gen.py --hour 1 --method generative+sensitivity
# python test_gen.py --hour 1 --method generative+performance
python test_gen.py --hour 1 --method generative+novelty 
python adafuzz.py
python curefuzz.py