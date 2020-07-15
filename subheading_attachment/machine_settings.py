class _MachineConfig:
    def __init__(self):

        root_dir = 'home/****'
        self.data_dir = '/{}/input-data/subheading-attachment'.format(root_dir)
        self.database_host = 'localhost'
        self.database_port = 3306

        self.train_limit = 1000000000                           
        self.dev_limit   = 1000000000
        self.test_limit = 1000000000

        self.runs_dir = '/{}/runs/subheading-attachment'.format(root_dir)               
        self.use_multiprocessing = True
        self.workers = 10
        self.max_queue_size = 10