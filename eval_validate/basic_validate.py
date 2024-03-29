# 评测中的validate基类


class BasicValidate(object):
    def __init__(self):
        self.epoch = 0
        self.model = None
        pass

    def load_config(self):
        pass

    # 必须实现
    def load_data(self):
        pass

    # model
    def load_model(self, epoch):
        """
        如果epoch为0，新建model
        如果model==None，epoch不为0，尝试载入resume的model
        """
        return self.model
        pass

    # validate
    def validate(self, model, epoch):
        pass
