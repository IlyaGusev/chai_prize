import fire

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.tools.translate import Translator


def translate_file(
    sp_model_path: str,
    ct_model_path: str,
    input_path: str,
    output_path: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl"
):
    records = read_jsonl(input_path)
    translator = Translator(sp_model_path, ct_model_path)
    outputs = translator.translate_records(records, src_lang, tgt_lang)
    write_jsonl(outputs, output_path)


if __name__ == "__main__":
    fire.Fire(translate_file)
