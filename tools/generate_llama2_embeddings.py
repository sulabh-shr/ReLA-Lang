from gres_model.utils.llama2 import Llama2TokenEncoder

if __name__ == "__main__":
    import os
    import json
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-2-13b-hf", help="model ID"
    )
    parser.add_argument(
        "--eos", action="store_true", help="add EOS token to the input text"
    )
    parser.add_argument(
        "--data", type=str, default="grefs(unc).json", help="path to the instances file"
    )
    parser.add_argument(
        "--output", type=str, default="embeddings", help="output directory"
    )
    parser.add_argument("--batch", type=int, default=50, help="batch size")
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing embeddings"
    )
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k:-<20s} : {v}")

    # Load the grefcoco instances.json file
    with open(args.data, "r") as file:
        data = json.load(file)

    tokenizer = Llama2TokenEncoder(model_id=args.model_id, add_eos=args.eos)

    os.makedirs(args.output, exist_ok=True)

    # Assuming JSON structure is a list of dictionaries
    # each dictionary has a key 'sentences' with list of sentence dictionaries
    # each sentence dictionary has a key 'sent_id' and 'raw'
    all_sentences = {}

    for ann_idx, ann in enumerate(data, start=1):
        sentences = ann["sentences"]
        for sentence in sentences:
            id_ = sentence["sent_id"]
            text = sentence["raw"]
            all_sentences[id_] = text

    sentence_ids = sorted(list(all_sentences.keys()))
    num_sentences = len(sentence_ids)
    print(f"Total sentences: {num_sentences}")

    if not args.overwrite:
        existing_files = os.listdir(args.output)
        existing_ids = set(
            [int(file.split(".")[0]) for file in existing_files if file.endswith(".pt")]
        )
        sentence_ids = [id_ for id_ in sentence_ids if id_ not in existing_ids]
        print(f"Skipping existing sentences: {len(existing_ids)}")
        print(f"Remaining sentences: {len(sentence_ids)}")
        num_sentences = len(sentence_ids)

    start_time = last_print_time = time.time()
    for idx in range(0, num_sentences, args.batch):
        batch_ids = sentence_ids[idx: idx + args.batch]
        batch_texts = [all_sentences[id_] for id_ in batch_ids]
        batch_decoded_tokens, batch_embeddings = tokenizer.encode(batch_texts)
        names = [os.path.join(args.output, f"{id_}.pt") for id_ in batch_ids]
        tokenizer.save(
            batch_decoded_tokens,
            batch_embeddings,
            names=names,
        )

        if time.time() - last_print_time >= 60:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (idx + 1) * (num_sentences - idx - 1) / 60
            print(
                f"Processed [{idx} / {num_sentences}] sentences | "
                f"Remaining time: {remaining_time:.2f} minutes",
                flush=True,
            )
            last_print_time = time.time()
