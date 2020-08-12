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
Use UCI format such as: g1f3, from square:(g1), to square:(f3)
## Implementation:
- state.py returns a 8x8x5 bitvector, add an axis to the vector using ```.unsqueeze(0)```  then input it into the neural network in nn.py. Use the output for alpha beta in play.py.
