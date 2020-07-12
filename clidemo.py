import torch
from module import Tokenizer, init_model_by_key
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-s", "--stop_flag", default='q', type=str)
    parser.add_argument("-c", "--cuda", action='store_true')
    args = parser.parse_args()
    print("loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_info = torch.load(args.path)
    tokenizer = model_info['tokenzier']
    model = init_model_by_key(model_info['args'], tokenizer)
    model.load_state_dict(model_info['model'])
    while True:
        question = input("上联：")
        if question == args.stop_flag.lower():
            print("Thank you!")
            break
        input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        print(f"下联：{pred}")
if __name__ == "__main__":
    run()
