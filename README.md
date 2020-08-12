# chess-ai
Simple neural network implementation for chess in CLI

## Requirements 
```bash
pip3 install torch numpy python-chess
```
## How to play:
```bash
python3 play.py
```
## Implementation:
- state.py returns a 8x8x5 bitvector, add an axis to the vector using ```bash .unsqueeze(0)```  then input it into the neural network in nn.py. Use the output for alpha beta in play.py.
