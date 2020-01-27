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
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.graphlayer1, net.graphlayer2])

        self.qna_networks = LightGraphQA(embed_size, hidden_size) #both 256

        # number of channel in the last resnet is 256
        self.reason = nn.Linear(128, 1)

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
        node_features = torch.empty((batch_size, 5, 128)).cuda()
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
        features = features.view(batch_size, -1, 128)
        node_features[:, 1:5, :] = features
            
        return node_features

    def forward(self, observations, action, preprocessed = None):
        if preprocessed is None:
            features= self.preprocess(observations)
        else:
            features = preprocessed
            
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
        #pdb.set_trace()
        #last_hidden = nn.functional.adaptive_max_pool2d(last_hidden, 1)
        #last_hidden = last_hidden.flatten(1)
        #decision = self.reason(last_hidden).squeeze(-1)
        #ce_loss = nn.functional.binary_cross_entropy_with_logits(decisions, targets, reduce = False)
        #pdb.set_trace()
        #qa_loss + ce_loss

        return qa_loss#, ce_loss

    def compute_reward(self, embedding, edges):

        _, last_hidden = self.qna_networks(embedding, edges)

        #last_hidden = nn.functional.adaptive_max_pool2d(last_hidden, 1)
        #last_hidden = last_hidden.flatten(1)
        decision = self.reason(last_hidden).squeeze(-1)

        return decision

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

class GraphQA(nn.Module):

    def __init__(self,
                 entity_dim,
                 hidden_size):
        super().__init__()
        # output_size is designated as a number of channel in resnet
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS)) #이거 왜있는거죠?

        self.graph_net = BypassFactorGCNet(entity_dim)
        self.location = nn.Linear(hidden_size, 2)
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
            pdb.set_trace()
            out = torch.cat((red_out, green_out, blue_out, gray_out), 1)
            #pdb.set_trace()
            if t == 15:
                last_location = entity
            outputs[:, t, :] = out

        return outputs, last_location

    def MSE_loss(self, labels, targets):

        #pdb.set_trace()
        loss = nn.functional.mse_loss(labels, targets, reduce = False)

        return loss


class BypassFactorGCNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, entity_dim, num_blocks=4, num_units=2,
                 pooling='avg', preact_normalization='batch', spatial=1, stop_grad=True):
        super().__init__()
        self.spatial = spatial
        dim_layers = [entity_dim * spatial * spatial] * num_blocks
        self.entity_dim = entity_dim * spatial * spatial

        self.num_layers = len(dim_layers) - 1
        self.gblocks = nn.ModuleList()

        self.stop_grad = stop_grad  ##??

        for n in range(self.num_layers):
            gblock_kwargs = {
                'input_dim': dim_layers[n],
                'output_dim': dim_layers[n + 1],
                'num_units': num_units,
                'pooling': pooling,
                'preact_normalization': preact_normalization,
            }
            self.gblocks.append(GraphResBlock(**gblock_kwargs))

    def forward(self, entity, edges, stop_grad=None):
        """
        :param pose: (Batch_size, N_o, C, H, W)
        :param edges:
        :return:
        """
        out = {}

        if stop_grad and self.stop_grad:
            entity = entity.clone()
            entity= entity.detach()
        ## check later! update vs not
        Batch_size = entity.size(0)
        N_o = entity.size(1)
        entity = entity.view(Batch_size, N_o, -1)

        for i in range(self.num_layers):
            net = self.gblocks[i]
            obj_vecs = net(entity, edges)

        return obj_vecs

class GraphResBlock(nn.Module):
    """ A residual block of 2 Graph Conv Layer with one skip conection"""

    def __init__(self, input_dim, output_dim, num_units=2, pooling='avg', preact_normalization='batch'):
        super().__init__()
        self.num_units = num_units
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'pooling': pooling,
            'preact_normalization': preact_normalization,
        }
        GraphUnit = GraphEdgeConv

        for n in range(self.num_units):
            if n == self.num_units - 1:
                gconv_kwargs['output_dim'] = output_dim
            else:
                gconv_kwargs['output_dim'] = input_dim
            self.gconvs.append(GraphUnit(**gconv_kwargs))

    def forward(self, entity, edges):

        for i in range(self.num_units):
            gconv = self.gconvs[i]
            obj_vecs = gconv(entity, edges)

        return obj_vecs


class GraphEdgeConv(nn.Module):
    """
    Single Layer of graph conv: node -> edge -> node
    """
    def __init__(self, input_dim, output_dim=None, edge_dim=128,
                 pooling='avg', preact_normalization='batch'):
        super().__init__()
        if output_dim is None:
            output_dim = 128
        if edge_dim is None:
            edge_dim = 128
        self.input_dim = 128
        self.output_dim = 128
        self.edge_dim = 128
        # Node, edge 개수 정하기

        assert pooling in ['sum', 'avg', 'softmax'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        self.net_node2edge = build_pre_act(2*self.input_dim, edge_dim, 20, batch_norm=preact_normalization)
        self.net_edge2node = build_pre_act(self.input_dim + edge_dim, self.input_dim, 5, batch_norm=preact_normalization)
        self.net_node2edge.apply(_init_weights)
        self.net_edge2node.apply(_init_weights)

    def forward(self, obj_vecs, edges):
        """
        Inputs:
          + obj_vecs: (Batch_size, N_o, F)
          + edges:

        Outputs:
          + new_obj_vecs: (Batch_size, N_o, F)

        Alg:
          relu(AXW), new_AX = AX, mlp = relu(new_AX, W)
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        #pdb.set_trace()
        obj_vecs = obj_vecs.transpose(1,2)
        Rs = edges['Rs']
        Rr = edges['Rr']
        Rs = torch.from_numpy(Rs).float().to(device)
        Rr = torch.from_numpy(Rr).float().to(device)
        #pdb.set_trace()
        V = obj_vecs.size(0)
        N_o = obj_vecs.size(1)
        #pdb.set_trace()
        N_e = Rs.size(1)
        #pdb.set_trace()


        #Sender, Receiver Node Representation
        src_obj = torch.matmul(obj_vecs, Rs)
        dst_obj = torch.matmul(obj_vecs, Rr)
        
        # Node -> Edge, Massage Passing
        node_obj = torch.cat([src_obj, dst_obj], dim=-1).view(-1, N_e, 256)
        #pdb.set_trace()
        if node_obj.size(0) != 1:
            edge_obj = self.net_node2edge[0](node_obj)
        else:
            edge_obj = node_obj
        edge_obj = self.net_node2edge[1](edge_obj)
        edge_obj = self.net_node2edge[2](edge_obj)
        #pdb.set_trace()
        
        # Edge - > Node, Massage Aggregation
        aggregation_node = torch.matmul(Rr, edge_obj)
        obj_vecs = obj_vecs.transpose(1,2)
        update_node = torch.cat([obj_vecs, aggregation_node], dim=-1)
        #pdb.set_trace()
        if aggregation_node.size(0) != 1:
            new_obj_vecs = self.net_edge2node[0](update_node)
            #pdb.set_trace()
        else:
            new_obj_vecs = update_node
            #pdb.set_trace()
        #pdb.set_trace()
        new_obj_vecs = self.net_edge2node[1](new_obj_vecs)
        #pdb.set_trace()
        new_obj_vecs = self.net_edge2node[2](new_obj_vecs)
        #pdb.set_trace()
        
        return new_obj_vecs
    
class LightGraphQA(nn.Module):

    def __init__(self,
                 entity_dim,
                 edge_dim):
        super().__init__()
        # output_size is designated as a number of channel in resnet
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS)) #이거 왜있는거죠?

        self.graph_net = InteractionNetwork(128,128)
        self.location = nn.Linear(128, 2)
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
        self.relational_model = RelationalModel(2*object_dim, effect_dim, 256)
        self.object_model     = ObjectModel(object_dim + effect_dim, object_dim, 128)
    
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


