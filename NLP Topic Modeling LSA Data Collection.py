# from https://www.youtube.com/watch?v=iQ1bfDMCv_c&t=533s
# from https://github.com/adashofdata/nlp-in-python-tutorial/blob/master/1-Data-Cleaning.ipynb
import requests
from bs4 import BeautifulSoup
import pickle


def url_to_transcript(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="markdown-converter__text--rendered").find_all('p')]
    return text


# urls = ["https://www.kaggle.com/navinmundhra/daily-power-generation-in-india-20172020"]
# transcripts = [url_to_transcript(u) for u in urls]

# print(transcripts)

url = "https://www.kaggleusercontent.com/services.dataview.v1.DataViewer/GetDataView"
result = requests.get(url).text
# soup = BeautifulSoup(page, "lxml")
# print(soup.find(class_="content-box"))
# print(result.status_code)
# print(result.headers)
print(result)
soup = BeautifulSoup(result, "html.parser")

links = soup.find_all("p")
print(links)

