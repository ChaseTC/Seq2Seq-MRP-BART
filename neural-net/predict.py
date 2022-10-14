from transformers import pipeline, BartTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from pathlib import Path
from tqdm.auto import tqdm
import argparse

def predict(args):
    out_path = Path('data/predictions')

    test_dataset = load_dataset('json', data_dir='data/extracted/test')
    model_checkpoint = args.model
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
    translator = pipeline("translation", model=model_checkpoint, tokenizer=tokenizer)

    if not out_path.exists():
        out_path.mkdir()
    predictions = []
    print('starting predictions...')
    for out in tqdm(translator(KeyDataset(test_dataset['train'], "sentence"), max_length=1024, batch_size=8, num_beams=5)):
        predictions.append(out[0]['translation_text'])
    with open(out_path/args.model + '-pred.txt', 'w') as f:
        for p in predictions:
            f.write(p + '\n')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model', help='directory path to model to use for prediction')

    args = argparser.parse_args()

    predict(args)
    
if __name__ == '__main__':
    main()