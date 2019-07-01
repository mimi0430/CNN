from PIL import Image
import glob


files=glob.glob('信号青/*')
for i in range(len(files)):
    picture=files[i]
    picture = Image.open(picture ,'r')
    if picture.mode != "RGB":
        picture = picture.convert("RGB")
    picture.save('拡張子/sample'+ str(i) + '.jpg', 'JPEG')
