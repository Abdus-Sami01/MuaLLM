import json
from pathlib import Path

out_file = Path('data/qa/github_edu_qa.jsonl')
out_file.parent.mkdir(parents=True, exist_ok=True)

files = list(Path('data/edu_dialogue').glob('conversations_train*.json'))
count = 0
limit = 2000  # We use 2000 dialogues for fine-tuning so it doesn't take days locally

with open(out_file, 'w', encoding='utf-8') as f_out:
    for file in files:
        data = json.load(open(file, 'r', encoding='utf-8'))
        for item in data:
            dialogue = ""
            for turn in item.get('conversation', []):
                # Standardize to User/Bot format for our SFT script
                role = "User" if turn["role"] == "Student" else "Bot"
                dialogue += f"{role}: {turn['text']}\n"
            
            # Add termination token
            dialogue = dialogue.strip() + " [SEP]"
            
            f_out.write(json.dumps({"text": dialogue}) + "\n")
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break

print(f"Successfully processed {count} highly-detailed educational dialogues!")
print(f"Saved to {out_file}")
