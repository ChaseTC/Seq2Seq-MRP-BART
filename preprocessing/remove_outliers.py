from pathlib import Path
import json

in_path = Path('data/penman')
out_path = Path('data/subset')
splits = ['dev', 'train']

max_tokens = 512

def load_files():
    for split in splits:
        total = 0
        added = 0
        write_dir = out_path / split

        for file in (in_path / split).iterdir():
            j_lines = []

            with open(file, 'r') as f:
                for line in f.readlines():
                    total += 1
                    input = json.loads(line)
                    if len(input['penman'].split()) < max_tokens:
                        added += 1
                        j_lines.append(json.dumps(input))
            filename = file.name.split('\\')[-1]
            with open(write_dir/filename, 'w') as f:
                for line in j_lines:
                    if line != "":
                        f.write(line + "\n")
                        pass
        print('removed: {} from {}'.format(total-added, split))

def main():
    if not out_path.exists():
        out_path.mkdir()
    for split in splits:
        if not (out_path / split).exists():
            (out_path / split).mkdir()
    load_files()
    

if __name__ == '__main__':
    main()