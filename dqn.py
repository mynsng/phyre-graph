from neural_model import *
import phyre
import torch
import numpy as np
import pdb

default_params = {
    'network_type'        :'resnet18',
    'train_batch_size'    : 64,
    'max_train_actions'   : 100000,
    'balance_classes'     : True,   # balancing is critical for the 2B-tasks
    'updates'             : 100000, # number of total updates to perform
    'cosine_scheduler'    : True,
    'learning_rate'       : 3e-4,
    'action_hidden_size'  : 512,
    'report_every'        : 1000,
    'eval_every'          : 20000,
    'eval_size'           : 10000,  # evaluation size of the train & dev data    # check later
    'num_auccess_actions' : 10000,  # number of actions to evaluate in AUCCESS
    'num_auccess_tasks'   : 200,
    'qa_coef'             : 1,
    'max_predict_actions' : 1,
    'embed_size'          : 128,
    'hidden_size'         : 128,
    'report_statistic'    : 99900,
    'test_size'           : 10000,
    
    'rank_size'            : 10000,
    'eval_batch_size'      : 128,
    'max_attempts_per_task': 100
    
}

class DQNAgent():

    def __init__(self, params = default_params):
        self.params = default_params
        self.neural_model = NeuralModel()
        
    def build_model(self):
        
        model  = self.neural_model._build_model(network_type = self.params['network_type'],
                                           action_space_dim = 3,
                                           action_hidden_size = self.params['action_hidden_size'],
                                           embed_size = self.params['embed_size'],
                                           hidden_size = self.params['hidden_size'])
        
        return model
    
    def train(self, cache, task_ids, tier, dev_task_ids):
        model, statistic = self.neural_model.train(cache, 
                                        task_ids, 
                                        tier, 
                                        dev_task_ids, 
                                        default_params)
        
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data.shape)
        #        
        #total_params = sum(p.numel() for p in model.parameters())
        #print(total_params)
        #pdb.set_trace()
    
        state = dict(model =model, cache = cache)
        #torch.cuda.empty_cache()
        return state, statistic
    
    def eval(self, state, task_ids, tier):
        model  = state['model']
        cache  = state['cache']
        # NOTE: Current agent is only using the actions that are seen in the training set,
        #       though agent has the ability to rank the actions that are not seen in the training set
        actions = state['cache'].action_array[:self.params['rank_size']]
        
        model.cuda()
        simulator = phyre.initialize_simulator(task_ids, tier)
        observations = simulator.initial_scenes
        evaluator = phyre.Evaluator(task_ids)
        
        for task_index in range(len(task_ids)):
            task_id = simulator.task_ids[task_index]
            observation = observations[task_index]
            scores  = self.neural_model.eval_actions(model,
                                                     actions,
                                                     self.params['eval_batch_size'],
                                                     observation)
            # Rank of the actions in descending order
            action_order = np.argsort(-scores)
            # Result of the actions are already stored in cache
            statuses = cache.load_simulation_states(task_id)
            
            for action_id in action_order:
                if evaluator.get_attempts_for_task(task_index) >= self.params['max_attempts_per_task']:
                    break
                status = phyre.SimulationStatus(statuses[action_id])
                evaluator.maybe_log_attempt(task_index, status)
        return evaluator
    
    def predict_qa(self, state, task_ids, tier, action):
        #torch.cuda.empty_cache()
        model  = state['model']
        cache  = state['cache']
        #task_ids = [task_ids[140]]
        #pdb.set_trace()
        # NOTE: Current agent is only using the actions that are seen in the training set,
        #       though agent has the ability to rank the actions that are not seen in the training set
        #actions = state['cache'].action_array[:self.params['rank_size']]
        
        model.cuda()
        obs, obs_predict = self.neural_model.predict_qa(model, cache, task_ids, tier, action)
        return obs, obs_predict
    
    def get_test_loss(self, state, task_ids, tier):
        
        model = state['model']
        cache = state['cache']
        
        model.cuda()
        loss = self.neural_model.get_test_loss(model, cache, task_ids, tier)
        
        return loss
    
if __name__=='__main__':
    agent = DQNAgent()