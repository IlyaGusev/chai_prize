import fire
import ctranslate2
import sentencepiece as spm

from tqdm import tqdm
from nltk import sent_tokenize
from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.util.dl import gen_batch


def segment_text(text, text_id):
    segments = []
    if "\n" in text:
        fragments = [f for f in text.split("\n") if f.strip()]
        for fragment in fragments:
            if len(fragment) < 300:
                segments.append({"sentence": fragment, "text_id": text_id, "delimiter": "\n"})
            else:
                for sentence in sent_tokenize(fragment):
                    segments.append({"sentence": sentence, "text_id": text_id, "delimiter": " "})
            segments[-1]["delimiter"] = "\n"
    else:
        if len(text) < 300:
            segments.append({"sentence": text, "text_id": text_id, "delimiter": ""})
        else:
            for sentence in sent_tokenize(text):
                segments.append({"sentence": sentence, "text_id": text_id, "delimiter": " "})
    return segments


def translate(
    ct_model_path,
    sp_model_path,
    input_path,
    output_path
):
    # Load the source SentecePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    translator = ctranslate2.Translator(ct_model_path, "cuda")

    # Source and target langauge codes
    src_lang = "eng_Latn"
    tgt_lang = "rus_Cyrl"

    records = read_jsonl(input_path)
    for batch in tqdm(list(gen_batch(records, batch_size=512))):
        entries = []
        for record in batch:
            text_id = record["text_id"]
            text = record["text"].strip()
            entries.extend(segment_text(text, text_id))

        target_prefix = [[tgt_lang]] * len(entries)
        source_sents_subworded = sp.encode_as_pieces([r["sentence"] for r in entries])
        source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

        translations_subworded = translator.translate_batch(
            source_sents_subworded,
            batch_type="tokens",
            max_batch_size=4096,
            beam_size=5,
            target_prefix=target_prefix
        )
        translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
        for translation in translations_subworded:
            if tgt_lang in translation:
                translation.remove(tgt_lang)

        translations = sp.decode(translations_subworded)
        for entry, target in zip(entries, translations):
            if entry["sentence"]:
                entry["target"] = target
            else:
                entry["target"] = ""

        target_texts = []
        current_sentences = []
        current_text_id = None
        for entry in entries:
            text_id = entry["text_id"]
            if current_text_id is not None and text_id != current_text_id:
                target_texts.append("".join(current_sentences).strip())
                current_sentences = []
            current_text_id = text_id
            current_sentences.append(entry["target"])
            current_sentences.append(entry["delimiter"])
        target_texts.append(" ".join(current_sentences))
        assert len(batch) == len(target_texts), str(batch) + "\n\n" + str(target_texts)
        for record, target_text in zip(batch, target_texts):
            record["translation"] = target_text
    write_jsonl(records, output_path)


if __name__ == "__main__":
    fire.Fire(translate)
