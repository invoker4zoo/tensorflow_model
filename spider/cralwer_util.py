# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: cralwer_util.py
@ time: $17-10-20 上午11:50
"""

import sys
import os
import random
import requests
from pprint import pprint
import threading
import datetime


USER_AGENTS = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0',
    'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11',
    'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11',
]


def random_user_agent():
    return random.choice(USER_AGENTS)


class BaseCrawler(object):
    def __init__(self, **kwargs):
        logger = kwargs.get('logger')
        if logger:
            self._log = logger.info
            self._exception = logger.exception
            del kwargs['logger']
        else:
            self._exception = self._log = pprint
        self.__dict__ = dict(self.__dict__, **kwargs)
        self._request = requests.Session()
        self._request.headers['User-Agent'] = random_user_agent()

    def get_raw(self, url, timeout=10, times=3):
        if times == 0:
            return None
        try:
            return self._request.get(url, timeout=timeout)
        except Exception, e:
            self._exception(e)
            return self.get_raw(url, timeout=timeout, times=times - 1)

    def get(self, url, timeout=10, times=3):
        raw = self.get_raw(url, timeout=timeout, times=times)
        if raw:
            return raw.content
        return None

    def post_raw(self, url, data, headers=None, timeout=10, times=3):
        if times == 0:
            return None
        try:
            if headers:
                headers = dict(self._request.headers, **headers)
                return self._request.post(url, data=data, headers=headers, timeout=timeout)
            return self._request.post(url, data=data, timeout=timeout)
        except Exception, e:
            self._exception(e)
            return self.post_raw(url, data, headers=headers, timeout=timeout, times=times - 1)

    def post(self, url, data, headers=None, timeout=10, times=3):
        raw = self.post_raw(url, data, headers=headers, timeout=timeout, times=times)
        if raw:
            return raw.content
        return None


def Handler(start, end, url, filename):
    headers= {'Range': 'bytes=%d-%d'%(start, end)}
    res = requests.get(url, headers=headers, stream=True)
    with open(filename, 'r+b') as f:
        # 移动到起始位置
        f.seek(start)
        var = f.tell()
        f.write(res.content)


def download_multi_thread(url, file_path, num_thread=5):
    res = requests.head(url)
    try:
        file_name = res.headers['Content-Disposition']
        # rewrite part for different url
        # this for shooter
        file_name = file_name.split(';')[-1].split('="')[-1][:-1]
        file_size = int(res.headers['Content-Length'])

    except Exception, e:
        print 'get url file failed for: %s'%str(e)
        return 0
    file_name = os.path.join(file_path, file_name)
    # 创建一个和要下载文件一样大小的文件
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()

    # 启动多线程写文件
    part = file_size // num_thread  # 如果不能整除，最后一块应该多几个字节
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:  # 最后一块
            end = file_size
        else:
            end = start + part

        t = threading.Thread(target=Handler, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
        t.setDaemon(True)
        t.start()
    # 等待所有线程下载完成
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()
    print('%s 下载完成' % file_name)
    return 1

# download_multi_thread('http://shooter.zimuku.net/download/NTcwNjV8Z2xhZGlhdG9yXzIwMDBfZXh0ZW5kZWRfc3Vic19yMy5yYXI%3D')