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
from cralwer_util import BaseCrawler, download_multi_thread
import requests
import time

# logger = logger.logger
SAVING_PATH = '/home/showlove/cc/shooter'
BASE_URL = 'http://www.zimuku.net'
SEARCHING_URL = 'http://www.zimuku.net/search?q=&t=onlyst&p={}'


class ShooterCrawler(BaseCrawler):
    def __init__(self):
        super(ShooterCrawler, self).__init__()

    def save(self, down_load_link, saving_path=SAVING_PATH):
        if download_multi_thread(down_load_link, saving_path):
            logger.info('success download file')
        else:
            logger.error('Fun [download_multi_thread] failed')

    def get_down_load_url(self, link_url):
        try:
            download_content = self.get(link_url)
            download_soup = BeautifulSoup(download_content)
            download_a = download_soup.find('a', attrs={'id': 'down1'})
            download_link = download_a.attrs['href']
            # download_url = BASE_URL + download_link
            if 'http' in download_link:
                return download_link
            else:
                return BASE_URL + download_link
        except Exception, e:
            logger.error('[get_down_load_url] failed for: %s'%str(e))
            return None

    def run(self):
        try:
            # for page_num in range(1,12994):
            for page_num in range(195, 12994):
                logger.info('searching shooter in page %d'%page_num)
                url = SEARCHING_URL.format(page_num)
                content = self.get(url)
                soup = BeautifulSoup(content)
                div_list = soup.find_all('div', attrs={'class': 'persub clearfix'})
                for index, div_item in enumerate(div_list):
                    logger.info('page %d ---- downloading No.%d file....'%(page_num, index))
                    _link = div_item.find('h1').find('a').attrs['href']
                    link = BASE_URL + _link
                    down_load_link = self.get_down_load_url(link)
                    if down_load_link:
                        self.save(down_load_link)
                    else:
                        logger.warn('failed get download url in page %s'%link)

                    time.sleep(5)
        except Exception, e:
            logger.error('searching download page failed for : %s'%str(e))


crawler = ShooterCrawler().run()
