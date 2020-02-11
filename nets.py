import torch
import torch.nn as nn
import torchvision
import phyre
import numpy as np
import pdb
import cv2
from layers import build_pre_act

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=2)
            
class RewardFCNet(nn.Module):
            
    def __init__(self):
        super().__init__()
        
        self.pooling = nn.Linear(5,1)
        self.reason = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, embedding):
        
        batch_size, n, input_size = embedding.size()
        embedding= embedding.view(-1, n)
        graph = self.pooling(embedding)
        graph = graph.view(batch_size, -1)
        reward = self.reason(graph)
        reward = reward.squeeze(1)
        
        return reward
    
    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(decisions, targets)
        
class ResNet18FilmAction(nn.Module):

    def __init__(self,
                 action_size,
                 action_hidden_size,
                 embed_size,
                 hidden_size):
        super().__init__()
        # output_size is designated as a number of channel in resnet
        output_size = 256
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

        self.qna_networks = LightGraphQA(embed_size, hidden_size) #both 256
        # number of channel in the last resnet is 256
        self.reason = RewardFCNet()

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):

        batch_size = observations.size(0)
        image = self._image_colors_to_onehot(observations)
        green = image[:, 2, :, :].unsqueeze(1)
        blue1 = image[:, 3, :, :]
        blue2 = image[:, 4, :, :]
        blue = blue1 + blue2
        blue = blue.unsqueeze(1)
        gray = image[:, 5, :, :].unsqueeze(1)
        black = image[:, 6, :, :].unsqueeze(1)
        entities = [green, blue, gray, black]
        entity = torch.empty((batch_size, 4, 1, 256, 256)).cuda()
        node_features = torch.empty((batch_size, 5, 512)).cuda()
        t = 0
        for obj in entities:
            entity[:, t, :, :, : ] = obj
            t = t+1
        entity = entity.view(-1, 1, 256, 256)
        
        features = self.stem(entity)
        for stage in self.stages:
            features = stage(features)
        features = nn.functional.adaptive_max_pool2d(features, 1)
        features = features.flatten(1)
        features = features.view(batch_size, -1, 512)
        node_features[:, 1:5, :] = features
            
        return node_features

    def forward(self, observations, action, preprocessed = None):
        if preprocessed is None:
            features= self.preprocess(observations)
        else:
            features = preprocessed
            
        if features.shape[0] == 1 and action.shape[0] != 1:
            features = features.expand(action.shape[0], 5, 512) 
        action = self._apply_action(action)
        action = torch.from_numpy(action).float().to(features.device)
        action = action.unsqueeze(1)
        action = self.stem(action)
        for stage in self.stages:
            action = stage(action)
        action = nn.functional.adaptive_max_pool2d(action, 1)
        action = action.flatten(1)
        features[:, 0, :] = action
        #pdb.set_trace()
        return features

    def predict_location(self, embedding, edges):

        outputs, _ = self.qna_networks(embedding, edges)
      
        return outputs

    def compute_loss(self, embedding, edges, label_batch, targets):

        label_batch = torch.from_numpy(label_batch).float().to(embedding.device)
        predict_location, last_hidden = self.qna_networks(embedding, edges)

        targets = targets.to(dtype=torch.float, device=embedding.device)
        qa_loss = self.qna_networks.MSE_loss(label_batch, predict_location)
        qa_loss = torch.mean(qa_loss, 1)
        qa_loss = torch.mean(qa_loss, 1)
        
        ce_loss = self.reason.ce_loss(self.get_score(last_hidden), targets)

        return qa_loss, ce_loss
    
    def get_score(self, embedding):
        
        score = self.reason(embedding)
        
        return score
    
    def compute_16_loss(self, embedding, edges, label_batch, targets):

        label_batch = torch.from_numpy(label_batch).float().to(embedding.device)
        predict_location, last_hidden = self.qna_networks(embedding, edges)

        targets = targets.to(dtype=torch.float, device=embedding.device)
        qa_loss = self.qna_networks.MSE_loss(label_batch, predict_location)
        qa_loss = qa_loss[:, -1, :]
        qa_loss = torch.mean(qa_loss, 1)
        #last_hidden = nn.functional.adaptive_max_pool2d(last_hidden, 1)
        #last_hidden = last_hidden.flatten(1)
        #decision = self.reason(last_hidden).squeeze(-1)
        #ce_loss = nn.functional.binary_cross_entropy_with_logits(decisions, targets, reduce = False)
        #pdb.set_trace()
        #qa_loss + ce_loss

        return qa_loss

    def last_hidden(self, embedding, edges):

        _, last_hidden = self.qna_networks(embedding, edges)

        #last_hidden = nn.functional.adaptive_max_pool2d(last_hidden, 1)
        #last_hidden = last_hidden.flatten(1)
        #decision = self.reason(last_hidden).squeeze(-1)

        return last_hidden

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot

    def _apply_action(self, action):
        
        batch_size = action.size(0)
        img = np.zeros((batch_size, 256, 256))
        
        for t in range(batch_size):
            t_action= [action[t][0]*256//1, action[t][1]*256//1, action[t][2]*32//1]
            action_img = np.zeros((256,256))
            action_img = cv2.circle(action_img, (int(t_action[0]), int(t_action[1])), int(t_action[2]*2), (1), -1)
            img[t, :, :] = action_img
        
        return img

    #def auccess_loss(self, embedding, label_batch, targets)
    
class LightGraphQA(nn.Module):

    def __init__(self,
                 entity_dim,
                 edge_dim):
        super().__init__()
        # output_size is designated as a number of channel in resnet
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS)) #이거 왜있는거죠?

        self.graph_net = InteractionNetwork(512,128)
        self.location = nn.Linear(512, 2)
        #self.loss_fn = nn.MSELoss()

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'


    def forward(self, entity, edges):  #label = (batch, time, location)


        batch_size = entity.size(0)
        #edges = edges.to(entity.device)
        outputs = torch.empty((batch_size, 16, 8)).cuda()
        # If multiple actions are provided with a given image, shape should be adjusted
        #if features.shape[0] == 1 and actions.shape[0] != 1:
        #    features = features.expand(actions.shape[0], -1)

        for t in range(16):

            entity = self.graph_net(entity, edges)

            red_out = self.location(entity[:, 0, :])
            green_out = self.location(entity[:, 1, :])
            blue_out = self.location(entity[:, 2, :])
            gray_out = self.location(entity[:, 3, :])
            out = torch.cat((red_out, green_out, blue_out, gray_out), 1)

            if t == 15:
                last_location = entity
            outputs[:, t, :] = out

        return outputs, last_location

    def MSE_loss(self, labels, targets):

        #pdb.set_trace()
        loss = nn.functional.mse_loss(labels, targets, reduce = False)

        return loss

class InteractionNetwork(nn.Module):
    
    def __init__(self, object_dim, effect_dim):
        super(InteractionNetwork, self).__init__()
        
        self.object_dim = object_dim
        self.relational_model = RelationalModel(2*object_dim, effect_dim, 512)
        self.object_model     = ObjectModel(object_dim + effect_dim, object_dim, 512)
    
    def forward(self, obj_vecs, edges):
        
        dtype, device = obj_vecs.dtype, obj_vecs.device
        #pdb.set_trace()
        obj_vecs = obj_vecs.transpose(1,2)
        Rs = edges['Rs']
        Rr = edges['Rr']
        Rs = torch.from_numpy(Rs).float().to(device)
        Rr = torch.from_numpy(Rr).float().to(device)

        #Sender, Receiver Node Representation
        src_obj = torch.matmul(obj_vecs, Rs).transpose(1,2)
        dst_obj = torch.matmul(obj_vecs, Rr).transpose(1,2)
        
        # Node -> Edge, Massage Passing
        node_obj = torch.cat([src_obj, dst_obj], dim=-1)
        edge_obj = self.relational_model(node_obj)
        
        aggregation_node = torch.matmul(Rr, edge_obj)
        obj_vecs = obj_vecs.transpose(1,2)
        update_node = torch.cat([obj_vecs, aggregation_node], dim=-1)
        predicted = self.object_model(update_node)
        
        
        return predicted
    

class ObjectModel(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        batch_size, n, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n, self.output_size)
        
        return x
    
class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x


