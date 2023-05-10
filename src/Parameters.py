class Parameters:
    def __init__(self):
        self.PROJECT_DIR = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.batch_size = None
        self.epoch = None
        self.SaveModelDir = None
        self.checkpoint = None
        self.cp_callback = None
        self.current_model = None
        self.wanted_model = None
        self.x_test = None
        self.y_test = None
        self.inf_model = None

    def Print(self):
        print("PROJ_DIR : ", self.PROJECT_DIR)
