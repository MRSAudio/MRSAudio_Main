## calculate FD,FAD,IS,KL
install [audioldm_eval](https://github.com/haoheliu/audioldm_eval) by
```bash
git clone git@github.com:haoheliu/audioldm_eval.git
```
Then test with:
```bash
python scripts/test.py --pred_wavsdir {the directory that saves the audios you generated} --gt_wavsdir {the directory that saves audiocaps test set waves}
```