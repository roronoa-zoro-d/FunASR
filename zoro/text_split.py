import sys
import re



def is_chinese(char):

    if '\u4e00' <= char <= '\u9fff':
        return True
    else:
        return False



def normalize_string(text):
    text = " ".join(text.strip().split())   # 合并连续空格
    s = re.sub(r'[^\w\s]', '', text)        # 删除标点

    if len(s) <= 1:
        return [s]

    tokens = ""
    last_state = is_chinese(s[0])
    tokens += s[0]
    for i in range(1, len(s)):
        cur_state = is_chinese(s[i])
        # if last_state or cur_state or s[i].isdigit():
        if last_state:
            tokens += " "
        elif last_state != cur_state:
            tokens += " "
        tokens += s[i]
        last_state = cur_state
    cleaned_text = ' '.join(tokens.split())


    # tokens = ""
    # for ch in s:
    #     if ch.isalpha():
    #         ch = ch.lower()
    #     tokens += " " + ch
    # cleaned_text = ' '.join(tokens.split())
    cleaned_text = cleaned_text.lower()


    return cleaned_text.split()


def read_text(filename):
    utts = []
    utt2text = {}
    
    with open(filename, 'r') as f:
        for line in f:
            buff = line.strip().split("\t")
            utt = buff[0]
            text = ""
            if len(buff) == 2:
                text = buff[1]
            utt2text[utt] = text
            utts.append(utt)
    return utts, utt2text


if __name__ == '__main__':
    in_text_file = sys.argv[1]
    out_text_file = sys.argv[2]
    
    utts, utt2text = read_text(in_text_file)
    
    with open(out_text_file, 'w') as f:
        for utt in utts:
            text = utt2text[utt]
            tokens = normalize_string(text)
            f.write(utt + "\t" + " ".join(tokens) + "\n")
        