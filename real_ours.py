import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import alphashape
from collections import defaultdict
from tqdm import tqdm
import random
import os
import pandas as pd
import networkx as nx
from matplotlib import patches as mpatches
from sklearn import metrics
from itertools import combinations
from compute_onmi import compute_onmi


def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G

class ExplainableCommunityDetection:
    def __init__(self, name, k, G, a=0.5, alpha=0.8, beta=0, max_iter=100):
        self.G = G
        self.k = k
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.start_time = time.time()
        self.end_time = 0.0
        self.time_taken = 0.0
        self.overlap_ratio = 0                              # 重复覆盖率
        self.uncovered_ratio = 0                            # 未覆盖率
        self.num_overlap_nodes = 0                          # 重复节点数量
        self.num_uncovered_nodes = 0                        # 未覆盖节点数量
        self.overlap_nodes = []                             # 重复节点
        self.uncovered_nodes = []                           # 未覆盖节点
        self.objective = 0                                  # 目标函数值
        self.over_modularity = 0
        self.ori_NMI = 0
        self.ori_ri = 0
        self.ori_f_measure = 0
        self.ori_ONMI = 0
        self.ori_recall = 0
        self.ori_precision = 0
        self.ori_purity = 0
        self.modularity = 0                                 # 模块度
        self.NMI = 0                                        # NMI
        self.ONMI = 0
        self.ri = 0
        self.f_measure = 0
        self.recall = 0
        self.precision = 0
        self.purity = 0
        self.max_iter = max_iter                            # 最大循环次数
        self.n_nodes = len(G)                               # 节点数量
        self.node_list = list(G.nodes())                    # 节点列表
        self.coverage_dict = {}                             # 覆盖节点列表
        self.current_communities = defaultdict(list)        # 当前中心点和半径划分下的社区情况
        self.distance_matrix = self._precompute_distances() # 最短路径矩阵

    # 重新选择社区的中心点，根据度数选择
    def _select_new_center(self, comm_idx):
        community_nodes = self.current_communities[comm_idx]
        max_score = -1
        best_node = None
        for node in community_nodes:
            # 计算度中心性
            degree = self.G.degree(node)
            degree_centrality = degree / (self.n_nodes - 1)
            # 计算接近中心性
            c_idx = self.node_list.index(node)
            row = self.distance_matrix[c_idx]
            # 创建掩码：排除自身且距离有效
            mask = (row != np.iinfo(int).max) & (np.arange(self.n_nodes) != c_idx)
            reachable_indices = np.where(mask)[0]
            num_reachable = len(reachable_indices)
            if num_reachable == 0:
                closeness_centrality = 0.0
            else:
                sum_dist = row[reachable_indices].sum()
                closeness_centrality = (num_reachable / (self.n_nodes - 1)) * (num_reachable / sum_dist)
            # 总得分
            total_score = self.alpha * degree_centrality + (1 - self.alpha) * closeness_centrality
            if total_score > max_score:
                max_score = total_score
                best_node = node
        return best_node if best_node is not None else community_nodes[0]

    # 检查当前社区的半径扩展是否会覆盖到其他社区的中心点
    def _check_for_center_conflict(self, centers, comm_idx, radii):
        current_center = centers[comm_idx]
        current_radius = radii[comm_idx]
        # 获取当前社区中心的覆盖节点
        c_idx = self.node_list.index(current_center)
        covered_nodes = np.where(self.distance_matrix[c_idx] <= current_radius)[0].tolist()
        # 检查当前社区的覆盖节点是否包含其他社区的中心点
        for other_comm_idx, other_center in enumerate(centers):
            if other_comm_idx != comm_idx and other_center in [self.node_list[node] for node in covered_nodes]:
                return True  # 有冲突，返回 True
        return False  # 没有冲突，返回 False

    # 协调式半径搜索
    def _coordinated_radius_search(self, centers, coverage_dict):
        improved = False        #几率是否有优化
        community_order = random.sample(range(len(centers)), len(centers))  #打乱节点顺序，增加随机性
        current_obj = self._compute_global_objective(coverage_dict)
        # 遍历所有社区尝试增加半径
        for comm_idx in community_order:
            # 更新完半径后重新选定中心点
            new_center = self._select_new_center(comm_idx)
            new_centers = centers
            new_centers[comm_idx] = new_center
            # 检查是否有其他社区的中心点被覆盖
            if not self._check_for_center_conflict(new_centers, comm_idx, self.current_radii):
                # 看中心点会不会改善分割
                temp_coverage = self._calculate_coverage(new_centers, self.current_radii.copy())
                new_obj = self._compute_global_objective(temp_coverage)
                # 如果有改善，更新中心点
                if new_obj < current_obj:
                    self._update_current_communities(new_centers, self.current_radii)
                    self.centers[comm_idx] = new_center
                    centers[comm_idx] = new_center
                    improved = True
            # 看半径会不会改善分割
            original_r = self.current_radii[comm_idx]   #初始半径
            # 找最大半径长度
            sp_lengths = nx.single_source_shortest_path_length(self.G, new_centers[comm_idx])
            max_search_radius = max(sp_lengths.values())
            # 如果最大半径没有比原始半径大，不继续搜索
            if max_search_radius <= original_r:
                continue
            # 对半径进行局部搜索
            r = original_r + 1
            temp_radii = self.current_radii.copy()
            temp_radii[comm_idx] = r
            temp_coverage = self._calculate_coverage(new_centers, temp_radii)
            # 检查是否有其他社区的中心点被覆盖
            if self._check_for_center_conflict(new_centers, comm_idx, temp_radii):
                continue  # 如果有冲突，跳过当前半径
            new_obj = self._compute_global_objective(temp_coverage)
            # 如果有改善，更新半径
            if new_obj < current_obj:
                self.current_radii[comm_idx] = r
                self._update_current_communities(new_centers, self.current_radii)
                self.centers[comm_idx] = new_center
                centers[comm_idx] = new_center
                improved = True
        return improved

    # 预计算所有节点之间的最短路径距离
    def _precompute_distances(self):
        print("Precomputing distance matrix...")
        # 使用 np.iinfo(int).max 作为无穷大的替代值，数据类型为 int
        dist = np.full((self.n_nodes, self.n_nodes), np.iinfo(int).max, dtype=int)
        for i, u in enumerate(tqdm(self.node_list,desc='Construct distance_matrix')):
            lengths = nx.single_source_shortest_path_length(self.G, u)
            for j, v in enumerate(self.node_list):
                # 如果存在路径，将最短路径长度赋值给矩阵元素；否则使用无穷大替代值
                dist[i, j] = lengths.get(v, np.iinfo(int).max)
        return dist

    # 选取初始迭代中心
    def _select_initial_center(self, k):
        centers = []
        available = set(range(self.n_nodes))  # 节点索引集合
        if not available:
            return []
        # 按度数排序
        degrees = [self.G.degree(self.node_list[i]) for i in available]
        sorted_indices = sorted(available, key=lambda x: degrees[x], reverse=True)
        # 只考虑前80%的节点
        eighty_percent_indices = sorted_indices[:int(len(sorted_indices) * 0.8)]
        # 按度数排序选择首个中心
        first_center = eighty_percent_indices[0]
        centers.append(first_center)
        available.remove(first_center)
        eighty_percent_indices.remove(first_center)
        # 后续中心选择
        for _ in range(1, k):
            if not eighty_percent_indices:
                break
            max_separation = -1
            valid_candidates = []
            # 寻找距离已选中心最远的节点
            for node_idx in eighty_percent_indices:
                dists = [self.distance_matrix[node_idx, center] for center in centers]
                min_dist = np.min(dists)
                if min_dist > max_separation:
                    max_separation = min_dist
                    valid_candidates = [node_idx]
                elif min_dist == max_separation:
                    valid_candidates.append(node_idx)
            # 从候选节点中最终选择
            if valid_candidates:
                # 计算候选节点度数
                candidate_degrees = [self.G.degree(self.node_list[idx]) for idx in valid_candidates]
                max_degree = max(candidate_degrees)
                # 筛选最大度数候选
                max_degree_candidates = [valid_candidates[i] for i, d in enumerate(candidate_degrees) if
                                         d == max_degree]
                if len(max_degree_candidates) == 1:
                    selected = max_degree_candidates[0]
                else:
                    # 计算全局最短路径总距离
                    total_distances = [np.sum(self.distance_matrix[idx]) for idx in max_degree_candidates]
                    min_total = min(total_distances)
                    # 筛选最小总距离候选
                    min_distance_candidates = [max_degree_candidates[i] for i, d in enumerate(total_distances) if
                                               d == min_total]
                    selected = min_distance_candidates[0]  # 也可以随机选择
                # 更新数据结构
                centers.append(selected)
                available.remove(selected)
                eighty_percent_indices.remove(selected)
        return [self.node_list[i] for i in centers]

    # 全局目标函数
    def _compute_global_objective(self, coverage_dict):
        # 计算重复覆盖比例
        num_overlap_nodes = sum(1 for count in coverage_dict.values() if count > 1)
        overlap_ratio = num_overlap_nodes / self.n_nodes
        # 计算未覆盖比例
        num_uncovered_nodes = self.n_nodes - len(coverage_dict)
        uncovered_ratio = num_uncovered_nodes / self.n_nodes
        # 计算社区规模平衡因子
        community_sizes = [len(comm) for comm in self.current_communities.values()]
        if community_sizes:
            balance_term = (max(community_sizes) - min(community_sizes)) / self.n_nodes
        else:
            balance_term = 0
        obj = self.a * overlap_ratio + (1 - self.a) * uncovered_ratio + self.beta * balance_term
        # 目标函数
        return obj

    # 最终全局目标函数
    def _final_compute_global_objective(self, coverage_dict):
        # 计算重复覆盖节点（即覆盖次数大于 1 的节点）
        overlap_nodes = [node for node, count in coverage_dict.items() if count > 1]
        num_overlap_nodes = len(overlap_nodes)
        overlap_ratio = num_overlap_nodes / self.n_nodes

        # 计算未覆盖节点（即没有被任何中心覆盖的节点）
        uncovered_nodes = [node for node in self.node_list if node not in coverage_dict]
        num_uncovered_nodes = len(uncovered_nodes)
        uncovered_ratio = num_uncovered_nodes / self.n_nodes

        # 计算社区规模平衡因子
        community_sizes = [len(comm) for comm in self.current_communities.values()]
        if community_sizes:
            balance_term = (max(community_sizes) - min(community_sizes)) / self.n_nodes
        else:
            balance_term = 0
        # 计算目标函数值
        obj = self.a * overlap_ratio + (1 - self.a) * uncovered_ratio + self.beta * balance_term
        # 返回目标函数值和相关的统计信息
        return obj, overlap_ratio, uncovered_ratio, num_overlap_nodes, num_uncovered_nodes, overlap_nodes, uncovered_nodes

    # 计算当前覆盖状态
    def _calculate_coverage(self, centers, radii):
        coverage = defaultdict(int)
        for c, r in zip(centers, radii):
            c_idx = self.node_list.index(c)
            covered = np.where(self.distance_matrix[c_idx] <= r)[0].tolist()
            for node_idx in covered:
                node_value = self.node_list[node_idx]  # 获取实际节点值
                coverage[node_value] += 1
        return coverage

    # 保存社区指标信息
    def _save_community_metrics(self, communities):
        self.community_details = []
        for com_id in sorted(communities.keys()):
            detail = {
                "id": com_id,
                "size": len(communities[com_id]),
                "center": self.community_info[com_id][0],
                "radius": self.community_info[com_id][1],
            }
            self.community_details.append(detail)

    # 更新当前社区划分情况
    def _update_current_communities(self, centers, radii):
        self.current_communities.clear()
        for i, (c, r) in enumerate(zip(centers, radii)):
            c_idx = self.node_list.index(c)
            members = np.where(self.distance_matrix[c_idx] <= r)[0].tolist()
            self.current_communities[i] = [self.node_list[m] for m in members]

    # 主算法
    def detect_communities(self):
        # 初始化
        self.time = time.time()
        self.centers = self._select_initial_center(self.k)
        self.current_radii = [0] * self.k    # [0,0,0]
        best_obj = float('inf')
        coverage_dict = self._calculate_coverage(self.centers, self.current_radii)  #保存覆盖节点列表
        # 迭代优化中心点和半径,tqdm用于显示控制台的进度条
        for _ in tqdm(range(self.max_iter), desc="Optimizing"):
            # 每次迭代前更新社区状态
            self._update_current_communities(self.centers, self.current_radii)
            # 半径搜索
            improved = self._coordinated_radius_search(self.centers.copy(), coverage_dict)
            # 有优化执行更新操作
            if improved:
                coverage_dict = self._calculate_coverage(self.centers, self.current_radii)
                # 更新最优解
                current_obj, self.overlap_ratio, self.uncovered_ratio, self.num_overlap_nodes, self.num_uncovered_nodes, self.overlap_nodes, self.uncovered_nodes = self._final_compute_global_objective(coverage_dict)
                if current_obj < best_obj:
                    best_obj = current_obj
                    self.coverage_dict = coverage_dict
            # 早停机制
            else:
                break
        # 应用最优解
        communities = self.current_communities
        self.coverage_dict = self.coverage_dict
        self.objective = best_obj
        # 计算最终指标
        self.community_info = {i: (c, r) for i, (c, r) in enumerate(zip(self.centers, self.current_radii))}
        self._save_community_metrics(communities)
        self.end_time = time.time()
        self.time_taken = self.end_time - self.start_time
        return dict(communities)

def save_community_partition(dataset_name, output_folder, k, readPath):
    """
    执行 CPM 社区检测并保存结果。
    :param dataset_name: 数据集名称
    :param output_folder: 输出文件夹
    :param k: 团的大小
    :param readPath: 数据集路径
    :return: 数据集名称和处理时间（毫秒）
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 记录开始时间
    print(f"Performing CPM community detection for {dataset_name} with k={k}...")

    # 加载图数据
    G = load_graph(readPath)

    a = 0.5
    alpha = 0.8
    beta = 0

    start_time = time.time()
    # 执行社区检测
    ecd = ExplainableCommunityDetection(name=dataset_name, k=k, G=G, a=a, alpha=alpha, beta=beta, max_iter=50)  # 初始化
    detected_communities = ecd.detect_communities()

    # 保存结果，将整数节点转换为字符串写入
    txt_filename = f'{dataset_name}_communities.txt'
    with open(os.path.join(output_folder, txt_filename), 'w') as f:
        for community in detected_communities.values():
            f.write(" ".join(map(str, community)) + "\n")
    print(f"Saved community partition for {dataset_name} in {output_folder}.")

    # 返回时间（毫秒）
    end_time = time.time()
    duration_ms = int(round((end_time - start_time) * 1000))
    return (dataset_name, duration_ms)


if __name__ == '__main__':
    # 数据集列表
    names = ['karate', 'football', 'personal', 'polblogs', 'railways', 'email', 'polbooks']
    # names = ['polblogs']

    # 为每个数据集指定不同的 k 值
    k_values = {
        'karate': 2,
        'football': 12,
        'personal': 8,
        'polblogs': 2,
        'railways': 21,
        'email': 42,
        'polbooks': 3
    }

    # 输出文件夹
    output_folder = '../comparable/Ours-result/real'

    # 处理时间记录
    processing_times = []
    for name in names:
        k = k_values.get(name, 3)  # 若未指定，则默认 k=3
        readPath = f"../data/real/graph/{name}.txt"
        current_name, time_ms = save_community_partition(name, output_folder, k, readPath)
        processing_times.append((current_name, time_ms))

    # 将时间记录到 CSV 文件（保存到 'real' 子目录）
    csv_filename = f'{output_folder}/processing_times.csv'
    fieldnames = ['Dataset Name', 'ours Time (ms)']

    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        # 检查文件是否为空
        if os.stat(csv_filename).st_size == 0:
            writer.writerow(fieldnames)
        # 写入数据
        for name, time_ms in processing_times:
            writer.writerow([name, time_ms])

    print(f"\n所有社区检测完成！时间记录已保存至 '{csv_filename}' 文件中。")
