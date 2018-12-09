
class Config(object):
    def __init__(self):

        self.content_weight = 1
        self.style_weight = 50
        self.tv_weight = 1e-4
        self.batch_size = 8
        self.img_size = 256
        self.style_root = './data/WikiArt'  #风格数据集存放路径
        self.content_root = './data/COCO'   #内容数据集存放路径

        self.epoches = 2
        self.lr = 1e-3      #学习速率 (Learning rate)
        self.use_gpu = True

        self.content_path = 'content.png'
        self.style_path = 'style.png'
        self.result_path = 'result.png'


