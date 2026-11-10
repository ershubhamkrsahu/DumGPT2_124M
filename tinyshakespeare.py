import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
destination = "input.txt"

urllib.request.urlretrieve(url, destination)

print("Download complete!")