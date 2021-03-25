
import argparse
import os
import _pickle as pickle
import jsonlines
import jieba
from utils.parse_data import *
from tqdm import tqdm
def stopword_remover(line, total_stop_words_set):

    tokens = [w for w in line if not (w in total_stop_words_set
                    or is_number(w) or ((not is_ascii(w)) and len(w) <= 1))]
    return tokens

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default='../rawdata/final_all_data/first_stage/train.json',
                        help='The path to the input data.')
    parser.add_argument('--output', default='data/instances/train',
                        help='The path to output directory.')
    parser.add_argument('--relation', default='./data/my_dict.txt')
    parser.add_argument('--stop_words', default='./data/stop_words.txt',
                        help='The path to stop words.')
    parser.add_argument('--word_dict', default='./data/word_dict_10w.pkl')
    parser.add_argument('--id', required= True,
                        help='The prefix of generate json file.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    # Read child-parent label relations
    with open(args.relation) as f:
        c_to_p = eval(f.readline())
    # Read parent label dictionary
    pd = open('/Users/jasonchan/PycharmProjects/NLP/HMN/data/parent_dict.pkl', 'rb')
    parent_dict = pickle.load(pd)
    pd.close()
    # Read stop words
    with open(args.stop_words, 'r') as stops:
        s = stops.read()
        stop_words_from_file = s.split()
    stop_words_from_file = set(stop_words_from_file)
    total_stop_words_set = stop_words_from_file

    wordHelper = data_helper.Vocab("./data/word_dict_10w.pkl")
    # Read json file
    ind = 0
    with open(args.input , 'r') as js:
        JSON = jsonlines.Reader(js)
        for it in tqdm(JSON):
            parent_class = []
            text = it['fact']
            laws = it['meta']['relevant_articles']
            # Find corresponding parent labels
            for i in laws:
                if parent_dict['id2word'][c_to_p[str(i)]] in parent_class:
                    continue
                parent_class.append(parent_dict['id2word'][c_to_p[str(i)]])
            seg_fact = stopword_remover(jieba.lcut(text), stop_words_from_file)
            textIds = wordHelper.transform_raw(seg_fact)
            text_len = len(textIds)
            instance = {
                "text_len": text_len,
                "laws": laws,
                "textIds": textIds,
                "parent_class": parent_class
            }
            jsondata = json.dump(instance, open(os.path.join(args.output, ''.join([args.id, '_', str(ind), '.json'])), 'a'), ensure_ascii=False)
            ind+=1