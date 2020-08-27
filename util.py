import datetime
import re


def index_of_sentence(query, sentence):
    """Known q_list in k_list, find index(first time) of q_list in k_list"""
    q_list_length = len(query)
    s_list_length = len(sentence)
    idx_starts = []
    for idx in range(s_list_length - q_list_length + 1):
        if query[0] != sentence[idx]:  # quicker
            continue
        t = [q == k for q, k in zip(query, sentence[idx: idx + q_list_length])]
        if all(t):
            idx_starts.append(idx)
    return idx_starts


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flush_text(text: str) -> str:
    """
    在英文与中文中间添加空格
    """
    return re.sub("([a-zA-Z0-9]+\\s)*[a-zA-Z0-9]+", lambda x: " " + x.group() + " ", text)


def flush_token(token: str):
    if token.startswith("##"):
        return token[2:]
    return token


def get_showed_word(sentence: str, word: str):
    """
    从句子中抽取token组成word的原词
    eg: sentence为"a apple tree" word为"appletree", 返回 "apple tree"
    :param sentence:
    :param word:
    :return:
    """
    re_pattern = re.compile(r'([' + word[0] + r'].*[' + word[-1] + r'])', re.S)  # 贪婪匹配
    candidates = re.findall(re_pattern, sentence)
    for candidate in candidates:
        if candidate.replace(" ", "") == word:
            return candidate
    for candidate in candidates:
        if len(candidate) > len(word):
            res = get_showed_word(candidate[:len(candidate) - 1], word)
            if res is not None:
                return res
            res = get_showed_word(candidate[1:], word)
            if res is not None:
                return res
    return None


if __name__ == '__main__':
    sent = "急性支气管炎（acute bronchitis)是指由于各种致病原引起的支气管黏膜感染，由于气管常同时受累,故称为急性气管支气管炎(acute tracheobronchitis)。 【治疗】 一般治疗同上呼吸道感染，经常变换体位，多饮水，保持适当的湿度,使呼吸道分泌物易于咳出。"
    print(get_showed_word(sent, "acutetracheobronchitis"))
