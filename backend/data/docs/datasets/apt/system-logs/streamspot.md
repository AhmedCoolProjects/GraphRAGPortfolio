# StreamSpot

This dataset is a collection of **600** graphs.

The **attack scenario** graphs are:

- Drive-by-download attack (graph ID's 300 - 399)

The **benign scenario** graphs are:

- YouTube (graph ID's 0 - 99)
- GMail (graph ID's 100 - 199)
- VGame (graph ID's 200 - 299)
- Download (graph ID's 400 - 499)
- CNN (graph ID's 500 - 599)

The graph ID's are sorted by scenario as follows:

   1. YouTube (graph ID's 0 - 99)
   2. GMail (graph ID's 100 - 199)
   3. VGame (graph ID's 200 - 299)
   4. Drive-by-download attack (graph ID's 300 - 399)
   5. Download (graph ID's 400 - 499)
   6. CNN (graph ID's 500 - 599)

## Format

Each line in the dataset represents an edge in the graph. The format is as follows:

```
source-id (int)	source-type (char)	destination-id (int)	destination-type (char)	   edge-type (char)	   graph-id (int)
```

A snapshot of the dataset is as follows:

```
4       a       5       c       p       0
4       a       6       c       p       0
4       a       7       c       p       0
4       a       8       c       p       0
4       a       9       c       j       0
4       a       0       d       t       0
4       a       10      c       j       0
4       a       11      c       u       0
4       a       11      c       q       0
4       a       11      c       t       0
4       a       11      c       n       0
4       a       9       c       j       0
4       a       12      c       u       0
```

## Preprocessing

A simple script to prepare the data from the `all.tsv` file with the format mentioned above.

```python title="Generate NetworkX Node_Link Graphs in JSON Format"
import networkx as nx
import json
import dgl
from tqdm import tqdm
import os

NUM_GRAPHS = 600
node_type_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
edge_type_dict = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                  'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'C', 'D', 'E', 'G']

node_type_set = set(node_type_dict)
edge_type_set = set(edge_type_dict)

class StreamspotDataset(dgl.data.DGLDataset):
    '''
    1. Use numbers/indices for nodes/edges instead of characters
    2. Instead of the default streamspot format, we use networkx's node_link_graph format: {nodes: [{type, id}], links: [{type, source, target}], directed: True, multigraph: False}
    3. We save each graph in a separate json file
    '''
    def __init__(self, name, src_file_path, dst_folder_path):
        super(StreamspotDataset, self).__init__(name=name)
        self.name = name
        self.src_file_path = src_file_path
        self.dst_folder_path = dst_folder_path
        self.graphs = []
        self.labels = []
        # methods
        self.generate()
    
    def _save_graph(self, g, graph_id):
        for node in g.nodes():
            g.nodes[node]['type'] = node_type_dict.index(g.nodes[node]['type'])
        for edge in g.edges():
            g.edges[edge]['type'] = edge_type_dict.index(g.edges[edge]['type'])
        f = open(self.dst_folder_path + f'/g_{graph_id}.json', 'w', encoding='utf-8')
        json.dump(nx.node_link_data(g), f)
        f.close()

    def generate(self):
        count_graph = 0
        with open(self.src_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            g = nx.DiGraph()
            node_map = {}
            count_node = 0
            for line in tqdm(lines):
                src, src_type, dst, dst_type, e_type, graph_id = line.strip('\n').split('\t')
                graph_id = int(graph_id)
                src = int(src)
                dst = int(dst)
                # check if the extracted nodes are the ones in my type_dict
                if src_type not in node_type_set or dst_type not in node_type_set:
                    continue
                if e_type not in edge_type_set:
                    continue
                # in case we moved to another graph now
                if graph_id != count_graph:
                    count_graph += 1
                    self._save_graph(g, count_graph)
                    assert count_graph == graph_id
                    g = nx.DiGraph()
                    count_node = 0
                # check if src and dst are in the map, if not add them to the graph then to the map
                if src not in node_map:
                    node_map[src] = count_node
                    g.add_node(count_node, type=src_type)
                    count_node += 1
                if dst not in node_map:
                    node_map[dst] = count_node
                    g.add_node(count_node, type=dst_type)
                    count_node += 1
                # check if there's an edge between the src and dst, if not add it
                if not g.has_edge(node_map[src], node_map[dst]):
                    g.add_edge(node_map[src], node_map[dst], type=e_type)
            count_graph += 1
            self._save_graph(g, count_graph)
        
        print(f"Generated {count_graph} graphs")

    def process(self):
        pass

```

We will have **600** json files, each for a graph with the following format:

```json title="Each Processed Graph"
{
    "directed": true,
    "multigraph": false,
    "graph": {},
    "nodes": [
        {
            "type": 0,
            "id": 0
        },
        {
            "type": 2,
            "id": 1
        },
        ...
    "links": [
        {
            "type": 7,
            "source": 0,
            "target": 1
        },
        {
            "type": 7,
            "source": 0,
            "target": 2
        },
        ...
    ]
}
```

Now we need to prepare our **Benign vs. Malicious** dataset.

```python title="Build a DGL Dataset with Labels"
class StreamspotDataset(dgl.data.DGLDataset):
    '''
    1. Use numbers/indices for nodes/edges instead of characters
    2. Instead of the default streamspot format, we use networkx's node_link_graph format: {nodes: [{type, id}], links: [{type, source, target}], directed: True, multigraph: False}
    3. We save each graph in a separate json file
    '''
    def __init__(self, name, src_file_path, dst_folder_path):
        super(StreamspotDataset, self).__init__(name=name)
        self.name = name
        self.src_file_path = src_file_path
        self.dst_folder_path = dst_folder_path
        self.graphs = []
        self.labels = []
        # methods
        self.generate()
        self.load()
    
    def _save_graph(self, g, graph_id):
        # ...

    def generate(self):
        # ...

    def process(self):
        pass

    def load(self):
        print(f"Loading {self.name} dataset...")
        for i in tqdm(range(1, NUM_GRAPHS + 1), desc=f"Loading {self.name} dataset"):
            g = dgl.from_networkx(
                nx.node_link_graph(json.load(open(f'{self.dst_folder_path}/g_{i}.json'))),
                node_attrs=['type'],
                edge_attrs=['type']
            )
            self.graphs.append(g)
            if 301 <= i <= 400:
                self.labels.append(1)
            else:
                self.labels.append(0)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)
        
```



## References

You can find it on [GitHub](https://github.com/sbustreamspot/sbustreamspot-data) in In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

