import matplotlib.pyplot as plt
import numpy as np
import random
import phyre
import torch
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

tier = 'ball'
eval_setup = 'ball_within_template'
fold_id = 0
random.seed(0)

train, dev, test = phyre.get_fold(eval_setup, fold_id)
print('Size of resulting splits:\n train:', len(train), '\n dev:',
      len(dev), '\n test:', len(test))

cache = phyre.get_default_100k_cache(tier)
print('cache.action_array shape:', cache.action_array.shape)

from dqn import DQNAgent

agent = DQNAgent()

#model = agent.build_model()
#new_model = TestModel()
#model.load_state_dict(torch.load("./model/test_model.pth"))
#state = dict(model =model, cache = cache)

state, statistic = agent.train(cache, train, tier, dev)

model = state['model']

#save
savePath = "./model/16_512_within_auccess.pth"
torch.save(model.state_dict(), savePath)

file=open("./model/16_512_within_auccess_stt","wb") 
pickle.dump(statistic,file) 
file.close()

evaluator = agent.eval(state, test, tier)

print('AUCESS after 100 attempts on test set', evaluator.get_aucess())
