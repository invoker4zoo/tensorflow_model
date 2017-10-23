# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: crawler_zimu.py
@ time: $17-10-20 上午11:43
"""

from bs4 import BeautifulSoup
import requests
from logger import logger
from cralwer_util import BaseCrawler


# logger = logger.logger
SAVING_PATH = '/home/showlove/cc/shooter'
BASE_URL = 'http://www.zimuku.net'
SEARCHING_URL = 'http://www.zimuku.net/search?q=&t=onlyst&p={}'


class ShooterCrawler(BaseCrawler):
    def __init__(self):
        super(ShooterCrawler, self).__init__()

    def save(self, down_load_link):
        pass

    def get_down_load_url(self, link_url):
        return ''

    def run(self):
        try:
            for page_num in range(1,12994):
                logger.info()
                url = SEARCHING_URL.format(page_num)
                content = self.get(url)
                soup = BeautifulSoup(content)
                div_list = soup.find_all('div',attrs={'class':'persub clearfix'})
                for index, div_item in enumerate(div_list):
                    _link = div_item.find('h1').find('a').attrs('href')
                    link = BASE_URL + _link
                    down_load_link = self.get_down_load_url(link)
                    self.save(down_load_link)
        except Exception, e:
            logger.error('searching download page failed for : %s'%str(e))


    def save(self, saving_path=SAVING_PATH):
        pass
