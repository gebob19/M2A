import neptune 

def _update_keys(d, prefix):
    keys = list(d.keys())
    for k in keys: 
        d['{}_{}'.format(prefix, k)] = d.pop(k)

class NeptuneWriter:
    def __init__(self, proj_name):
        self.project = neptune.init(proj_name)
        self.has_started = False

    def start(self, args, **kwargs):
        self.experiment = self.project.create_experiment(
            name=args['experiment_name'], params=args, **kwargs)
        self.has_started = True 

    def fin(self):
        if self.has_started:
            # will finish when all data has been sent
            self.experiment.stop()
            self.has_started = False

    def write(self, data, step):
        if self.has_started:
            for k in data.keys():
                self.experiment.log_metric(k, step, data[k])
        else: 
            print('Warning: Writing to dead writer - call .start({}) first')

    def id(self):
        return self.experiment.id