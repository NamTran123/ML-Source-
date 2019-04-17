import urllib3.request as b
from bs4 import BeautifulSoup
page =  b.urlopen('https://www.skype.com/en/thank-you/').read()
soup = BeautifulSoup(page)
soup.prettify()
for anchor in soup.findAll('a', href=True):
    print (anchor['href'])