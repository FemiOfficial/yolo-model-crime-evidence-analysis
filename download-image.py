import urllib.request

count = 1
for line in open("images.txt").readlines():
    lineArr = line.split("\n")
    print()
    name = "images-table\\" + str(count) + ".jpg"
    urllib.request.urlretrieve(lineArr[0], name)
    count += 1
    break
