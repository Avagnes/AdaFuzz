python mdpfuzz.py --terminate 1
python test_gen.py --hour 1 --method generative+novelty 
python curefuzz --terminate 1
python adafuzz --terminate 1