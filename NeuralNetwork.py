import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

torch.manual_seed(0)

#Super Class
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, x):
        assert self.net is not None, "Sub class needs to define self.net"
        return self.net(x)

    def visualize(self, title="Network", show_edge_labels=True, show_biases=True,
                  weight_fmt="{:.2f}", bias_fmt="{:+.2f}"):
        
        weights, biases, activations, layer_shapes = [], [], [], []

        for module in self.net:
            if isinstance(module, nn.Linear):
                w = module.weight.detach().cpu().numpy()
                weights.append(w)
                biases.append(module.bias.detach().cpu().numpy() if module.bias is not None else None)
                layer_shapes.append((w.shape[1], w.shape[0]))  
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.ELU)):
                activations.append(module.__class__.__name__)

        layers = [{"name": "Input", "size": layer_shapes[0][0]}]
        for i, (_, out_dim) in enumerate(layer_shapes):
            lname = "Output" if i == len(layer_shapes) - 1 else f"Hidden{i+1}"
            layers.append({"name": lname, "size": out_dim})

        G = nx.DiGraph()
        pos = {}
        layer_x_spacing = 3
        neuron_y_spacing = 2
        node_labels = {}
        layer_positions = []

        for l, layer in enumerate(layers):
            ids = []
            size = layer["size"]
            y_off = (size - 1) * neuron_y_spacing / 2
            for n in range(size):
                node = f"L{l}_N{n}"
                G.add_node(node)
                pos[node] = (l * layer_x_spacing, -(n * neuron_y_spacing - y_off))
                label = f"{layer['name'][0]}{n+1}"
                if 0 < l < len(layers) - 1:
                    act_fn = activations[l - 1] if l - 1 < len(activations) else "None"
                    label += f"\n({act_fn})"
                node_labels[node] = label
                ids.append(node)
            layer_positions.append(ids)

        for l in range(len(weights)):
            w = weights[l] 
            for i, src in enumerate(layer_positions[l]):
                for j, tgt in enumerate(layer_positions[l + 1]):
                    weight_val = float(w[j, i])
                    G.add_edge(src, tgt, weight=round(weight_val, 2))

        plt.figure(figsize=(14, 8))
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="mediumpurple")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white", font_weight="bold")
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=35)

        if show_edge_labels:
            edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="darkgreen", font_size=17)

        if show_biases:
            for l, b_vec in enumerate(biases):
                if b_vec is None:
                    continue
                tgt_nodes = layer_positions[l + 1]
                for j, node in enumerate(tgt_nodes):
                    bx, by = pos[node]
                    bval = float(b_vec[j])
                    plt.text(bx + 0.35, by + 0.35, f"b={bias_fmt.format(bval)}",
                             fontsize=17, color="darkgreen")

        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def get_θ(self):
        theta = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().clone()
                W.requires_grad_(True)
                theta.append(W)
                if module.bias is not None:
                    b = module.bias.detach().clone()
                    b.requires_grad_(True)
                    theta.append(b)
        return theta

    def replace_θ(self, theta_new):
        idx = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    W_new = torch.as_tensor(theta_new[idx], dtype=module.weight.dtype, device=module.weight.device)
                    module.weight.copy_(W_new)
                    idx += 1
                    if module.bias is not None:
                        b_new = torch.as_tensor(theta_new[idx], dtype=module.bias.dtype, device=module.bias.device)
                        module.bias.copy_(b_new)
                        idx += 1

    def visualize_fx(self, sample_input, title="Computational Graph (fx)"):
        gm = symbolic_trace(self)
        
        print(gm.graph)           
        gm.graph.print_tabular()  

        G = nx.DiGraph()
        for node in gm.graph.nodes:
            G.add_node(str(node))
            for user in node.users:
                G.add_edge(str(node), str(user))

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=800)
        plt.title(title)
        plt.show()


#Sub Class for Multi Layer Perceptron
class MLP(NN):
    def __init__(self, sizes=(1, 3, 3, 3, 1), activation=nn.ReLU, add_softmax=False, softmax_dim=-1):
        super().__init__()

        self.sizes = sizes
        
        layers = []
        for i in range(len(sizes) - 1):
            in_f, out_f = sizes[i], sizes[i + 1]
            layers.append(nn.Linear(in_f, out_f))
            if i < len(sizes) - 2:          # nach allen außer der letzten Linear
                layers.append(activation())
        if add_softmax:
            layers.append(nn.Softmax(dim=softmax_dim))
        self.net = nn.Sequential(*layers)
    def getSizes():
        return self.sizes()

#Sub Class for Convolutional Neural Network
class TinyCNN(NN):
    def __init__(self, in_ch=1, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )
