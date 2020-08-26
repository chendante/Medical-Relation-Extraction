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
