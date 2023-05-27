import re
import requests
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from logging import getLogger
from tqdm import tqdm

logger = getLogger(__name__)


@dataclass
class ReactionGraph:
    graph_data: List[Tuple[str, str]]

    def plot_multigraph(self, ax: plt.axes=None):
        """
        Plots a multigraph using the NetworkX and Matplotlib libraries.

        :return: Matplotlib axis object
        """
        # Create an empty multigraph
        graph = nx.MultiGraph()

        # Add nodes (KEGG IDs and reaction IDs)
        logger.info('Adding nodes to graph...')
        all_ids = [i for j in self.graph_data for i in j]
        kegg_ids = set([i for i in all_ids if re.match(r'^K\d{5}$', i)])
        reaction_ids = set([i for i in all_ids if re.match(r'^R\d{5}$', i)])

        graph.add_nodes_from(all_ids)
        graph.add_edges_from(self.graph_data)

        # Set up plot settings
        logger.info('Setting up plot settings...')
        pos = nx.spring_layout(graph,  k=0.15, iterations=20)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Draw edges with alpha of 0.5
        # for u, v, d in tqdm(graph.edges(data=True)):
        #     ax.annotate("", xy=pos[u], xytext=pos[v], arrowprops=dict(arrowstyle="->", alpha=0.5))

        # Draw KEGG IDs (blue color)
        nx.draw_networkx_nodes(graph, pos, nodelist=list(kegg_ids), node_color='blue', ax=ax)

        # Draw reaction IDs (green color)
        nx.draw_networkx_nodes(graph, pos, nodelist=list(reaction_ids), node_color='green', ax=ax)

        # # Draw edges (black color)
        nx.draw_networkx_edges(graph, pos, edgelist=self.graph_data, width=1, alpha=0.5, edge_color='black', ax=ax)

        # Hide axis and labels
        ax.axis('off')

        # Create a custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reaction IDs'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='KEGG IDs')
        ]
        plt.legend(handles=legend_elements)

        # Show the plot
        return ax



class KEGGPathwayAPI:
    BASE_URL = 'https://rest.kegg.jp/link'

    def __init__(self):
        self.session = requests.Session()

    def get_pathway_kos(self, pathway_id):
        """
        Retrieves the KO identifiers (KOs) for a given pathway.

        :param pathway_id: KEGG pathway ID (e.g., "ko00010" for pathway "Metabolic pathways")
        :return: List of KO identifiers (KOs)
        """
        url = f'{self.BASE_URL}/ko/{pathway_id}'
        response = self.session.get(url)
        if response.status_code == 200:
            kos = self.extract_kos(response.text)
            return kos
        else:
            response.raise_for_status()

    def extract_kos(self, pathway_text):
        """
        Extracts the KO identifiers (KOs) from the pathway text.

        :param pathway_text: Text content of the KEGG pathway response
        :return: List of KO identifiers (KOs)
        """
        kos = []
        for line in pathway_text.split('\n'):
            if line:
                _, ko_ids = line.split('\t')
                ko_ids = re.findall(r'K\d+', ko_ids)
                kos.extend(ko_ids)
        return kos

    def get_ko_reaction_links(self, ko_ids):
        """
        Retrieves the links between KO identifiers and reaction IDs.

        :param ko_ids: List of KO identifiers (KOs)
        :return: List of tuples where the first element is a KO and the second element is a Reaction ID
        """
        links = []
        url = f"{self.BASE_URL}/reaction/{'+'.join(iter(ko_ids))}"
        response = self.session.get(url)
        if response.status_code == 200:
            for row in response.text.split('\n'):
                if row:
                    ko_id, reaction_id = row.split('\t')
                    ko_id = re.findall(r'K\d+', ko_id)[0]
                    reaction_id = re.findall(r'R\d+', reaction_id)[0]
                    links.extend([(ko_id, reaction_id)])
        else:
            response.raise_for_status()
        return links


if __name__ == "__main__":
    ko_ids = KEGGPathwayAPI().get_pathway_kos('ko00330')
    master_reaction_links = []
    if len(ko_ids) > 100:
        # Split the KO IDs into batches of 100. This is done as the KEGG API URL can only be so long.
        iter_batches = len(ko_ids) // 100
        for i in tqdm(range(0, len(ko_ids), 100), total=iter_batches):
            ko_ids_batch = ko_ids[i:i+100]
            reaction_links = KEGGPathwayAPI().get_ko_reaction_links(ko_ids_batch)
            master_reaction_links.extend(reaction_links)
    else:
        master_reaction_links = KEGGPathwayAPI().get_ko_reaction_links(ko_ids)

    graph = ReactionGraph(master_reaction_links)
    graph.plot_multigraph()
    plt.savefig('./output_images/hw5_multigraph.png')
